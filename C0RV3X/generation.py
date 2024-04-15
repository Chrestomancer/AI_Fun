# generation.py

import os
import openai
import logging
import ast
import re
from radon.metrics import mi_visit
from transformers import GPT2TokenizerFast
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

class InnovativeCodeGenerator:
    def __init__(self, api_key, cfg, embedding_model):
        self.memory = {}
        self.client = openai.OpenAI(api_key=api_key)
        self.cfg = cfg
        self.embedding_model = embedding_model
        self.tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
        self.rouge = Rouge()
        self.summarizer = LsaSummarizer()

    def _generate(self, prompt, task, n=1, stop=None, temperature=0.7, top_p=0.9, top_k=40, repetition_penalty=1.2):
        logging.debug(f"Generating code for task: {task}")
        try:
            response = self.client.chat.completions.create(
                model=self.cfg["models"][task],
                messages=[{"role": "system", "content": self.cfg['roles'][task]}, {"role": "user", "content": prompt}],
                n=n,
                stop=stop,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                repetition_penalty=repetition_penalty
            )
            generated_codes = [choice.message.content.strip() for choice in response.choices]
            logging.debug(f"Generated code for task {task}: {generated_codes}")
            return generated_codes
        except Exception as e:
            logging.error(f"Generate Error: {e}")
            return []

    def _create_file(self, filepath, content=""):
        logging.debug(f"Creating file: {filepath}")
        directory = os.path.dirname(filepath)
        if not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)

        with open(filepath, "w", encoding="utf-8") as f:
            logging.debug(f"Writing content to file: {filepath}")
            f.write(content)

    def _analyze_code(self, code):
        logging.debug("Analyzing code")
        try:
            code = self._preprocess_code(code)
            tree = ast.parse(code)
            analyzer = ASTAnalyzer()
            analyzer.visit(tree)
            analysis = analyzer.get_analysis()
            analysis['maintainability_index'] = mi_visit(code, True)
            logging.debug(f"Code analysis: {analysis}")
            return analysis
        except SyntaxError as e:
            logging.error(f"SyntaxError: {e}")
            return {'imports': [], 'functions': [], 'classes': [], 'complexity': 0, 'maintainability_index': 0}

    def _preprocess_code(self, code):
        logging.debug("Preprocessing code")
        return re.sub(r'""".*?"""', '', code, flags=re.DOTALL)

    def _summarize_code(self, code):
        logging.debug("Summarizing code")
        try:
            parser = PlaintextParser.from_string(code, Tokenizer("english"))
            summary = self.summarizer(parser.document, 3)
            summary = " ".join([str(sentence) for sentence in summary])
            logging.debug(f"Code summary: {summary}")
            return summary
        except Exception as e:
            logging.error(f"SummarizationError: {e}")
            return "Error summarizing code"

    def _refine_code(self, code, summary, analysis, context, iterations=3):
        for i in range(iterations):
            logging.debug(f"Refining code, iteration {i + 1}")
            prompt = self.cfg['prompt_template']['refine'].format(
                code=code, summary=summary, analysis=analysis, context=context
            )
            try:
                refined_codes = self._generate(prompt, "refine", n=3, stop=["```"])
                if not refined_codes:
                    break

                # Select the best refined code based on BLEU score
                bleu_scores = [sentence_bleu([code], refined_code) for refined_code in refined_codes]
                best_idx = bleu_scores.index(max(bleu_scores))
                code = refined_codes[best_idx]

                analysis = self._analyze_code(code)
                summary = self._summarize_code(code)
            except Exception as e:
                logging.error(f"RefinementError: {e}")
                break
        return code, summary, analysis

    def _generate_test_cases(self, code, summary, n=3):
        logging.debug("Generating test cases")
        prompt = self.cfg['prompt_template']['test_cases'].format(code=code, summary=summary)
        test_cases = self._generate(prompt, "test_cases", n=n, stop=["```"])
        return test_cases

    def _evaluate_test_cases(self, code, test_cases):
        logging.debug("Evaluating test cases")
        prompt = self.cfg['prompt_template']['evaluate_tests'].format(code=code, test_cases="\n".join(test_cases))
        evaluation = self._generate(prompt, "evaluate_tests", n=1)[0]
        return evaluation

    def _find_similar_code(self, code, top_k=3):
        logging.debug(f"Finding similar code")
        try:
            code_embedding = self.embedding_model.encode([code])[0]
            similarities = []
            for code_id, code_data in self.memory.items():
                code_embedding_mem = code_data['embedding']
                similarity = cosine_similarity([code_embedding], [code_embedding_mem])[0][0]
                similarities.append((code_id, similarity))
            similarities = sorted(similarities, key=lambda x: x[1], reverse=True)
            return [self.memory[code_id]['code'] for code_id, _ in similarities[:top_k]]
        except Exception as e:
            logging.error(f"SimilarityError: {e}")
            return []

    def _cluster_code(self, n_clusters=5):
        logging.debug("Clustering code")
        try:
            code_embeddings = [code_data['embedding'] for code_data in self.memory.values()]
            kmeans = KMeans(n_clusters=n_clusters, random_state=42).fit(code_embeddings)
            labels = kmeans.labels_
            for i, (code_id, code_data) in enumerate(self.memory.items()):
                code_data['cluster'] = labels[i]
        except Exception as e:
            logging.error(f"ClusteringError: {e}")

    def _visualize_code_space(self):
        logging.debug("Visualizing code space")
        try:
            code_embeddings = [code_data['embedding'] for code_data in self.memory.values()]
            pca = PCA(n_components=2, random_state=42).fit_transform(code_embeddings)
            # Visualize the code space using a scatter plot
            import matplotlib.pyplot as plt
            plt.figure(figsize=(8, 6))
            plt.scatter(pca[:, 0], pca[:, 1], c=[code_data['cluster'] for code_data in self.memory.values()])
            plt.xlabel("PCA Component 1")
            plt.ylabel("PCA Component 2")
            plt.title("Code Space Visualization")
            plt.colorbar(ticks=range(max([code_data['cluster'] for code_data in self.memory.values()]) + 1))
            plt.show()
        except Exception as e:
            logging.error(f"VisualizationError: {e}")

    def _store_generated_code(self, code):
        logging.debug("Storing generated code")
        try:
            code_embedding = self.embedding_model.encode([code])[0]
            code_id = len(self.memory)
            self.memory[code_id] = {'code': code, 'embedding': code_embedding}
        except Exception as e:
            logging.error(f"CodeStorageError: {e}")

    def generate_and_store_code(self, name, desc, context):
        prompt = self.cfg['prompt_template']['code'].format(
            name=name, desc=desc, context=context
        )
        generated_codes = self._generate(prompt, "code", n=3, stop=["```"])
        if not generated_codes:
            logging.error(f"Failed to generate code for {name} module")
            return "", "", {}

        # Select the best generated code based on code analysis metrics
        analyzed_codes = []
        for code in generated_codes:
            if self._is_valid_code(code):
                analysis = self._analyze_code(code)
                score = analysis['maintainability_index'] / (analysis['complexity'] + 1)
                analyzed_codes.append((code, score))
            else:
                logging.warning(f"Invalid code generated for {name} module")
        
        if not analyzed_codes:
            logging.error(f"No valid code generated for {name} module")
            return "", "", {}

        best_code, _ = max(analyzed_codes, key=lambda x: x[1])
        summary = self._summarize_code(best_code)
        test_cases = self._generate_test_cases(best_code, summary)
        evaluation = self._evaluate_test_cases(best_code, test_cases)
        logging.info(f"Test case evaluation for {name} module: {evaluation}")

        refined_code, refined_summary, refined_analysis = self._refine_code(best_code, summary, analysis, context)

        similar_codes = self._find_similar_code(refined_code)
        if similar_codes:
            logging.info(f"Similar code found for {name} module:")
            for code in similar_codes:
                logging.info(code)

        self._store_generated_code(refined_code)
        self._cluster_code()
        self._visualize_code_space()

        return refined_code, refined_summary, refined_analysis

    def _is_valid_code(self, code):
        try:
            ast.parse(code)
            return True
        except SyntaxError:
            return False

class ASTAnalyzer(ast.NodeVisitor):
    def __init__(self):
        self.imports = []
        self.functions = []
        self.classes = []
        self.complexity = 1

    def visit_Import(self, node):
        for alias in node.names:
            self.imports.append(alias.name)
        self.complexity += 1

    def visit_ImportFrom(self, node):
        self.imports.append(node.module)
        self.complexity += 1

    def visit_FunctionDef(self, node):
        self.functions.append(node.name)
        self.complexity += 1

    def visit_ClassDef(self, node):
        self.classes.append(node.name)
        self.complexity += 1

    def visit_If(self, node):
        self.complexity += len(node.body) + 1

    def visit_For(self, node):
        self.complexity += len(node.body) + 1

    def visit_While(self, node):
        self.complexity += len(node.body) + 1

    def visit_Try(self, node):
        self.complexity += len(node.body) + len(node.handlers) + 1

    def get_analysis(self):
        return {'imports': self.imports, 'functions': self.functions, 'classes': self.classes,
                'complexity': self.complexity}