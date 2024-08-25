# generation.py

import os
import openai
import logging
import ast
import re
from typing import List, Dict, Any, Tuple
from radon.metrics import mi_visit
from transformers import GPT2TokenizerFast, GPT2LMHeadModel
import torch
from torch.nn import functional as F
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.meteor_score import meteor_score
from rouge import Rouge
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lsa import LsaSummarizer
from ast_analyzer import ASTAnalyzer

class CodeGenerator:
    def __init__(self, api_key: str, cfg: Dict[str, Any], embedding_model):
        self.memory: Dict[str, Dict[str, Any]] = {}
        self.client = openai.OpenAI(api_key=api_key)
        self.cfg = cfg
        self.embedding_model = embedding_model
        self.tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
        self.generator = GPT2LMHeadModel.from_pretrained("gpt2")
        self.rouge = Rouge()
        self.summarizer = LsaSummarizer()
        self.ast_analyzer = ASTAnalyzer()

    def generate_code(
        self,
        name: str,
        desc: str,
        context: str,
        n: int = 1,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 40,
        repetition_penalty: float = 1.2,
    ) -> str:
        prompt = self._generate_prompt(name, desc, context)
        generated_code = self._generate_text(
            prompt,
            n=n,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
        )[0].strip()
        return generated_code

    def _generate_prompt(self, name: str, desc: str, context: str) -> str:
        return f"'''\nModule: {name}\nDescription: {desc}\nContext: {context}\n'''\n"

    def _generate_text(
        self,
        prompt: str,
        n: int = 1,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 40,
        repetition_penalty: float = 1.2,
    ) -> List[str]:
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt")
        output = self.generator.generate(
            input_ids,
            max_length=1024,
            num_return_sequences=n,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
            do_sample=True,
            pad_token_id=self.tokenizer.eos_token_id,
        )
        generated_text = [
            self.tokenizer.decode(
                output[i], skip_special_tokens=True
            )
            for i in range(n)
        ]
        return generated_text

    def _create_file(self, filepath: str, content: str = "") -> None:
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(content)

    def _analyze_code(self, code: str) -> Dict[str, Any]:
        try:
            analysis = self.ast_analyzer.analyze(code)
            analysis["maintainability_index"] = mi_visit(code, True)
            return analysis
        except SyntaxError as e:
            logging.error(f"SyntaxError: {e}")
            return {
                "imports": [],
                "functions": {},
                "classes": {},
                "complexity": 0,
                "maintainability_index": 0,
                "control_flow_graph": {},
            }

    def _summarize_code(self, code: str) -> str:
        try:
            parser = PlaintextParser.from_string(code, Tokenizer("english"))
            summary = self.summarizer(parser.document, 3)
            return " ".join([str(sentence) for sentence in summary])
        except Exception as e:
            logging.error(f"SummarizationError: {e}")
            return "Error summarizing code"

    def _refine_code(
        self,
        code: str,
        summary: str,
        analysis: Dict[str, Any],
        context: str,
        iterations: int = 3,
    ) -> Tuple[str, str, Dict[str, Any]]:
        for _ in range(iterations):
            prompt = self._generate_refinement_prompt(
                code, summary, analysis, context
            )
            refined_code = self._generate_text(prompt)[0].strip()
            if refined_code:
                analysis = self._analyze_code(refined_code)
                summary = self._summarize_code(refined_code)
                code = refined_code
        return code, summary, analysis

    def _generate_refinement_prompt(
        self, code: str, summary: str, analysis: Dict[str, Any], context: str
    ) -> str:
        return f"```python\n{code}\n```\n\nSummary: {summary}\n\nAnalysis: {analysis}\n\nContext: {context}\n\nRefined code:"

    def _generate_test_cases(self, code: str, summary: str, n: int = 3) -> List[str]:
        prompt = self._generate_test_case_prompt(code, summary)
        test_cases = self._generate_text(prompt, n=n, stop=["```"])
        return test_cases

    def _generate_test_case_prompt(self, code: str, summary: str) -> str:
        return f"```python\n{code}\n```\n\nSummary: {summary}\n\nTest cases:"

    def _evaluate_test_cases(self, code: str, test_cases: List[str]) -> str:
        prompt = self._generate_test_evaluation_prompt(
            code, test_cases
        )
        evaluation = self._generate_text(prompt, n=1)[0]
        return evaluation

    def _generate_test_evaluation_prompt(
        self, code: str, test_cases: List[str]
    ) -> str:
        return f"```python\n{code}\n```\n\nTest cases:\n{chr(10).join(test_cases)}\n\nEvaluation:"

    def _find_similar_code(self, code: str, top_k: int = 3) -> List[str]:
        try:
            code_embedding = self.embedding_model.encode([code])[0]
            similarities = []
            for code_id, code_data in self.memory.items():
                code_embedding_mem = code_data["embedding"]
                similarity = cosine_similarity(
                    [code_embedding], [code_embedding_mem]
                )[0][0]
                similarities.append((code_id, similarity))
            similarities = sorted(
                similarities, key=lambda x: x[1], reverse=True
            )
            return [
                self.memory[code_id]["code"] for code_id, _ in similarities[:top_k]
            ]
        except Exception as e:
            logging.error(f"SimilarityError: {e}")
            return []

    def _cluster_code(self, n_clusters: int = 5) -> None:
        try:
            code_embeddings = [
                code_data["embedding"] for code_data in self.memory.values()
            ]
            kmeans = KMeans(n_clusters=n_clusters, random_state=42).fit(
                code_embeddings
            )
            labels = kmeans.labels_
            for i, (code_id, code_data) in enumerate(self.memory.items()):
                code_data["cluster"] = labels[i]
        except Exception as e:
            logging.error(f"ClusteringError: {e}")

    def _visualize_code_space(self) -> None:
        try:
            code_embeddings = [
                code_data["embedding"] for code_data in self.memory.values()
            ]
            pca = PCA(n_components=2, random_state=42).fit_transform(
                code_embeddings
            )
            import matplotlib.pyplot as plt

            plt.figure(figsize=(8, 6))
            plt.scatter(
                pca[:, 0],
                pca[:, 1],
                c=[code_data["cluster"] for code_data in self.memory.values()],
            )
            plt.xlabel("PCA Component 1")
            plt.ylabel("PCA Component 2")
            plt.title("Code Space Visualization")
            plt.colorbar(
                ticks=range(
                    max([code_data["cluster"] for code_data in self.memory.values()])
                    + 1
                )
            )
            plt.show()
        except Exception as e:
            logging.error(f"VisualizationError: {e}")

    def _store_generated_code(self, code: str) -> None:
        try:
            code_embedding = self.embedding_model.encode([code])[0]
            code_id = len(self.memory)
            self.memory[code_id] = {"code": code, "embedding": code_embedding}
        except Exception as e:
            logging.error(f"CodeStorageError: {e}")

    def generate_and_store_code(
        self, name: str, desc: str, context: str
    ) -> Tuple[str, str, Dict[str, Any]]:
        generated_code = self.generate_code(name, desc, context)
        if not generated_code:
            logging.error(f"Failed to generate code for {name} module")
            return "", "", {}
        analysis = self._analyze_code(generated_code)
        score = analysis["maintainability_index"] / (analysis["complexity"] + 1)
        summary = self._summarize_code(generated_code)
        test_cases = self._generate_test_cases(generated_code, summary)
        evaluation = self._evaluate_test_cases(generated_code, test_cases)
        logging.info(f"Test case evaluation for {name} module: {evaluation}")
        refined_code, refined_summary, refined_analysis = self._refine_code(
            generated_code, summary, analysis, context
        )
        similar_codes = self._find_similar_code(refined_code)
        if similar_codes:
            logging.info(f"Similar code found for {name} module:")
            for code in similar_codes:
                logging.info(code)
        self._store_generated_code(refined_code)
        self._cluster_code()
        self._visualize_code_space()
        return refined_code, refined_summary, refined_analysis