# generation.py

import networkx as nx
import time
import os
import openai
from pathlib import Path
import logging
import ast
import re
from sklearn.metrics.pairwise import cosine_similarity
from radon.metrics import mi_visit

class CodeGenerator:
    def __init__(self, api_key, cfg, embedding_model):
        self.graph = nx.DiGraph()
        self.memory = {}
        self.client = openai.OpenAI(api_key=api_key)
        self.cfg = cfg
        self.embedding_model = embedding_model
        self.models = cfg.get("models", {"default": ("gpt-4-turbo-preview", 0.7, 0.95)})
        self.project_context = ""

    def _generate(self, prompt, task):
        logging.debug(f"Generating code for task: {task}")
        try:
            model, temp, top_p = self.models.get(task, self.models["default"])
            response = self.client.chat.completions.create(
                model=model, messages=[{"role": "system", "content": self.cfg['roles'][task]}, {"role": "user", "content": prompt}],
                temperature=temp, top_p=top_p, max_tokens=self.cfg['max_tokens'], stream=False
            )
            logging.debug(f"Generated code for task {task}: {response.choices[0].message.content.strip()}")
            return response.choices[0].message.content.strip()
        except Exception as e:
            if "maximum context length" in str(e):
                logging.error(f"OpenAI API context length exceeded: {e}")
            else:
                logging.error(f"Generate Error: {e}")
            return ""

    def _create_file(self, filepath, content="", max_retries=3, retry_delay=1):
        logging.debug(f"Creating file: {filepath}")
        directory = os.path.dirname(filepath)
        if not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)

        for attempt in range(max_retries):
            try:
                with open(filepath, "w", encoding="utf-8") as f:
                    logging.debug(f"Writing content to file: {filepath}")
                    f.write(content)
            
                if os.path.exists(filepath):
                    return True
                else:
                    logging.warning(f"File '{filepath}' not found after creation. Retrying...")
            except Exception as e:
                logging.error(f"Error while creating file '{filepath}': {str(e)}. Retrying...")
        
            time.sleep(retry_delay)
    
        logging.error(f"Failed to create file '{filepath}' after {max_retries} attempts.")
        return False


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
            if "EOL while scanning string literal" in str(e):
                return self._analyze_code(code.replace("'", "\\'").replace('"', '\\"'))
            logging.error(f"SyntaxError: {e}")
            return {'imports': [], 'functions': [], 'classes': [], 'complexity': 0, 'maintainability_index': 0}

    def _update_graph(self, module_path, code):
        logging.debug(f"Updating graph for module: {module_path}")
        try:
            tree = ast.parse(code)
            imports = [node.names[0].name for node in ast.walk(tree) if isinstance(node, ast.Import)]
            from_imports = [f"{node.module}.{node.names[0].name}" for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)]
        except SyntaxError:
            logging.warning(f"Failed to parse code for module: {module_path}")
            return

        self.graph.add_node(module_path)

        for imported_module in imports + from_imports:
            imported_module_path = os.path.join(os.path.dirname(module_path), f"{imported_module}.py")
            if imported_module_path in self.memory:
                self.graph.add_edge(module_path, imported_module_path)

        try:
            tree = ast.parse(code)
            analyzer = FunctionCallAnalyzer()
            analyzer.visit(tree)
            for called_func, called_from in analyzer.function_calls:
                logging.debug(f"Function call: {called_func} called from {called_from}")
                if called_from == module_path:
                    caller_module = module_path
                else:
                    caller_module = os.path.join(os.path.dirname(module_path), f"{called_from}.py")

                if called_func in self.memory:
                    called_module = os.path.join(os.path.dirname(module_path), f"{called_func}.py")
                    self.graph.add_edge(caller_module, called_module)
        except SyntaxError:
            logging.warning(f"Failed to analyze function calls for module: {module_path}")

    def _preprocess_code(self, code):
        logging.debug("Preprocessing code")
        return re.sub(r'""".*?"""', '', code, flags=re.DOTALL)

    def _find_similar_code(self, desc, top_k=3):
        logging.debug(f"Finding similar code for description: {desc}")
        try:
            embedding = self.embedding_model.encode([desc])
            similarities = []
            for mod in self.memory['modules']:
                summary_similarity = cosine_similarity(embedding, self.memory['modules'][mod]['summary_embedding'].reshape(1, -1))[0][0]
                dependency_similarity = self._calculate_dependency_similarity(mod)
                combined_similarity = 0.7 * summary_similarity + 0.3 * dependency_similarity
                similarities.append((mod, combined_similarity))
            logging.debug(f"Similar modules: {similarities}")
            return sorted(similarities, key=lambda x: x[1], reverse=True)[:top_k]
        except Exception as e:
            logging.error(f"SimilarityError: {e}")
            return []
        
    def _calculate_dependency_similarity(self, module_path):
        module_dependencies = set(self.graph.predecessors(module_path))
        return len(module_dependencies.intersection(self.memory['modules'].keys())) / len(module_dependencies)

    def _summarize_code(self, code):
        logging.debug("Summarizing code")
        try:
            prompt = self.cfg['prompt_template']['summarize'].format(code=code)
            summary = self._generate(prompt, "summarize")
            logging.debug(f"Code summary: {summary}")
            summary_embedding = self.embedding_model.encode([summary])[0]
            return summary, summary_embedding
        except Exception as e:
            logging.error(f"SummarizationError: {e}")
            return "Error summarizing code", None

    def _get_module_context(self, module_path):
        related_modules = self.graph.neighbors(module_path)
        related_summaries = [self.memory['modules'][mod]['summary'] for mod in related_modules]
        logging.debug(f"Related module summaries: {related_summaries}")
        return f"Related module summaries:\n" + "\n".join(related_summaries)

    def _refine_code(self, code, summary, analysis, context, similar_context, iterations=3):
        for i in range(iterations):
            logging.debug(f"Refining code, iteration {i + 1}")
            prompt = self.cfg['prompt_template']['refine'].format(
                code=code, summary=summary, analysis=analysis, context=self.project_context, similar_context=similar_context
            )
            try:
                code = self._generate(prompt, "refine")
                if not code:
                    break
                analysis = self._analyze_code(code)
                summary, _ = self._summarize_code(code)
            except Exception as e:
                logging.error(f"RefinementError: {e}")
                break
        return code, summary, analysis

    def relate_modules(self, module_paths):
        prompt = self.cfg['prompt_template']['relate_modules'].format(
            summaries=self._get_module_summaries(module_paths)
        )
        relation = self._generate(prompt, "relate_modules")
        for path in module_paths:
            self.memory['modules'][path]['relation'] = relation

    def set_project_context(self, context):
        logging.debug(f"Setting project context: {context}")
        self.project_context = context

    def _get_module_summaries(self, module_paths):
        return "\n\n".join([f"{path}:\n{self.memory['modules'][path]['summary']}" for path in module_paths])
        logging.debug(f"Module summaries: {summaries}")

    def _generate_module(self, name, desc, project_context, max_retries=3):
        for attempt in range(max_retries):
            prompt = self.cfg['prompt_template']['code'].format(
                name=name, desc=desc, project_context=project_context
            )
            code = self._generate(prompt, "code")
            if not code:
                logging.error(f"Failed to generate code for {name} module")
                continue
            if self._is_valid_code(code):
                analysis = self._analyze_code(code)
                logging.debug(f"Code analysis for {name} module: {analysis}")
                summary, summary_embedding = self._summarize_code(code)
                module_path = os.path.join(os.getcwd(), name + ".py")
                self.memory['modules'][module_path] = {'code': code, 'summary': summary, 'analysis': analysis, 'summary_embedding': summary_embedding}
            
                # Find similar modules and pass them as context for refinement
                logging.debug(f"Finding similar modules for {name} module")
                similar_modules = self._find_similar_code(desc)
                similar_context = "\n".join([f"{mod}: {self.memory['modules'][mod]['summary']}" for mod, _ in similar_modules])
                refined_code, refined_summary, refined_analysis = self._refine_code(code, summary, analysis, project_context, similar_context)
            
                return refined_code, refined_summary, refined_analysis
        
            logging.info(f"Retry generating {name} module (attempt {attempt + 1})")
        return "", "", {}


    def _is_valid_code(self, code):
        try:
            ast.parse(code)
            return True
        except SyntaxError:
            return False

class FunctionCallAnalyzer(ast.NodeVisitor):
    def __init__(self):
        self.function_calls = []

    def visit_Call(self, node):
        if isinstance(node.func, ast.Attribute):
            called_func = node.func.attr
            called_from = node.func.value.id
        elif isinstance(node.func, ast.Name):
            called_func = node.func.id
            called_from = None
        else:
            return

        self.function_calls.append((called_func, called_from))
        self.generic_visit(node)