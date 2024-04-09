# ast_analyzer.py

import ast
import logging

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
        if not node.handlers:
            logging.warning("Try statement without exception handlers detected")
        self.complexity += len(node.body) + len(node.handlers) + 1

    def get_analysis(self):
        return {'imports': self.imports, 'functions': self.functions, 'classes': self.classes,
                'complexity': self.complexity}