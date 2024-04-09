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
        try:
            model, temp, top_p = self.models.get(task, self.models["default"])
            response = self.client.chat.completions.create(
                model=model, messages=[{"role": "system", "content": self.cfg['roles'][task]}, {"role": "user", "content": prompt}],
                temperature=temp, top_p=top_p, max_tokens=self.cfg['max_tokens'], stream=False
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            if "maximum context length" in str(e):
                logging.error(f"OpenAI API context length exceeded: {e}")
            else:
                logging.error(f"Generate Error: {e}")
            return ""

    def _create_file(self, filepath, content=""):
        try:
            Path(filepath).parent.mkdir(parents=True, exist_ok=True)
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(content)
        except Exception as e:
            if "maximum context length" in str(e):
                logging.error(f"OpenAI API context length exceeded: {e}")
            else:
                logging.error(f"Error while creating file '{filepath}': {e}")

    def _analyze_code(self, code):
        try:
            code = self._preprocess_code(code)
            tree = ast.parse(code)
            analyzer = ASTAnalyzer()
            analyzer.visit(tree)
            analysis = analyzer.get_analysis()
            analysis['maintainability_index'] = mi_visit(code, True)
            return analysis
        except SyntaxError as e:
            if "EOL while scanning string literal" in str(e):
                return self._analyze_code(code.replace("'", "\\'").replace('"', '\\"'))
            logging.error(f"SyntaxError: {e}")
            return {'imports': [], 'functions': [], 'classes': [], 'complexity': 0, 'maintainability_index': 0}

    def _update_graph(self, module_path, code):
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
        return re.sub(r'""".*?"""', '', code, flags=re.DOTALL)

    def _find_similar_code(self, code_snippet, top_k=3):
        try:
            embedding = self.embedding_model.encode([code_snippet])
            similarities = [(mod, cosine_similarity(embedding, emb.reshape(1, -1))[0][0]) for mod, emb in
                            self.memory['modules'].items()]
            return sorted(similarities, key=lambda x: x[1], reverse=True)[:top_k]
        except Exception as e:
            logging.error(f"SimilarityError: {e}")
            return []

    def _summarize_code(self, code):
        try:
            prompt = self.cfg['prompt_template']['summarize'].format(code=code)
            return self._generate(prompt, "summarize")
        except Exception as e:
            logging.error(f"SummarizationError: {e}")
            return "Error summarizing code"

    def _get_module_context(self, module_path):
        related_modules = self.graph.neighbors(module_path)
        related_summaries = [self.memory['modules'][mod]['summary'] for mod in related_modules]
        return f"Related module summaries:\n" + "\n".join(related_summaries)

    def _refine_code(self, code, summary, analysis, context, iterations=3):
        for i in range(iterations):
            prompt = self.cfg['prompt_template']['refine'].format(
                code=code, summary=summary, analysis=analysis, context=self.project_context
            )
            try:
                code = self._generate(prompt, "refine")
                if not code:
                    break
                analysis = self._analyze_code(code)
                summary = self._summarize_code(code)
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
        self.project_context = context

    def _get_module_summaries(self, module_paths):
        return "\n\n".join([f"{path}:\n{self.memory['modules'][path]['summary']}" for path in module_paths])

    def _generate_module(self, name, desc, project_context, max_retries=3):
        """Generate a module with the given name, description, and project context."""
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
                summary = self._summarize_code(code)
                return code, summary, analysis
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
