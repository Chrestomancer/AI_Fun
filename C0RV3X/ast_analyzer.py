# ast_analyzer.py

import ast
import logging
from collections import defaultdict
from typing import List, Dict, Any
import networkx as nx
from nltk.sentiment.vader import SentimentIntensityAnalyzer

class ASTAnalyzer:
    def __init__(self):
        self.imports: List[str] = []
        self.functions: Dict[str, Dict[str, Any]] = {}
        self.classes: Dict[str, Dict[str, Any]] = {}
        self.complexity: int = 1
        self.control_flow_graph: nx.DiGraph = nx.DiGraph()
        self.sentiment_analyzer = SentimentIntensityAnalyzer()

    def analyze(self, code: str) -> Dict[str, Any]:
        self._reset()
        tree = ast.parse(code)
        self.visit(tree)
        return {
            "imports": self.imports,
            "functions": self.functions,
            "classes": self.classes,
            "complexity": self.complexity,
            "control_flow_graph": nx.to_dict_of_lists(self.control_flow_graph),
        }

    def _reset(self):
        self.imports = []
        self.functions = {}
        self.classes = {}
        self.complexity = 1
        self.control_flow_graph = nx.DiGraph()

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            self.imports.append(alias.name)
        self.complexity += 1

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        self.imports.append(node.module)
        self.complexity += 1

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._analyze_function(node)

    def _analyze_function(self, node: ast.FunctionDef, class_name: str = None) -> None:
        function_name = node.name
        if class_name:
            function_name = f"{class_name}.{function_name}"
        self.functions[function_name] = {
            "docstring": ast.get_docstring(node),
            "arguments": [arg.arg for arg in node.args.args],
            "complexity": 0,
            "control_flow": [],
            "sentiment": [],
        }
        self._analyze_block(
            node.body, parent_block=function_name, block_type="Function"
        )

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        class_name = node.name
        self.classes[class_name] = {
            "docstring": ast.get_docstring(node),
            "methods": [],
            "complexity": 0,
            "control_flow": [],
            "sentiment": [],
        }
        for stmt in node.body:
            if isinstance(stmt, ast.FunctionDef):
                self.classes[class_name]["methods"].append(stmt.name)
                self._analyze_function(stmt, class_name=class_name)
        self._analyze_block(
            node.body, parent_block=class_name, block_type="Class"
        )

    def _analyze_block(
        self, body: List[ast.AST], parent_block: str, block_type: str
    ) -> None:
        current_block = parent_block
        for i, stmt in enumerate(body):
            block_name = f"{current_block} - Block {i+1}"
            self.control_flow_graph.add_node(block_name, type=type(stmt).__name__)
            self.control_flow_graph.add_edge(current_block, block_name)
            self._update_complexity(block_type, parent_block, stmt)
            self._analyze_statement(stmt, block_name, parent_block)
            current_block = block_name

    def _analyze_statement(
        self, stmt: ast.AST, block_name: str, parent_block: str
    ) -> None:
        if isinstance(stmt, ast.If):
            self._analyze_block(stmt.body, block_name, "If")
            if stmt.orelse:
                self._analyze_block(stmt.orelse, block_name, "Else")
        elif isinstance(stmt, (ast.For, ast.While)):
            self._analyze_block(
                stmt.body, block_name, "Loop"
            )
        elif isinstance(stmt, ast.Try):
            self._analyze_block(stmt.body, block_name, "Try")
            for handler in stmt.handlers:
                self._analyze_block(
                    handler.body, block_name, f"Handler: {handler.type}"
                )
            if stmt.finalbody:
                self._analyze_block(stmt.finalbody, block_name, "Finally")
        elif isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Constant):
            self._analyze_docstring(stmt.value.value, parent_block)

    def _analyze_docstring(self, docstring: str, parent_block: str) -> None:
        if docstring is not None:
            sentiment = self.sentiment_analyzer.polarity_scores(docstring)
            if parent_block in self.functions:
                self.functions[parent_block]["sentiment"].append(sentiment)
            elif parent_block in self.classes:
                self.classes[parent_block]["sentiment"].append(sentiment)

    def _update_complexity(
        self, block_type: str, parent_block: str, stmt: ast.AST
    ) -> None:
        if block_type == "Function":
            self.complexity += 1
            self.functions[parent_block]["complexity"] += 1
        elif block_type == "Class":
            self.complexity += 1
            self.classes[parent_block]["complexity"] += 1
        if isinstance(stmt, (ast.If, ast.For, ast.While, ast.Try)):
            self.complexity += 1
            if block_type == "Function":
                self.functions[parent_block]["complexity"] += 1
            elif block_type == "Class":
                self.classes[parent_block]["complexity"] += 1