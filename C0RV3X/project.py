import os
import logging
import re
import yaml
import networkx as nx
from typing import List, Dict, Tuple
from sentence_transformers import SentenceTransformer
from code_generator import CodeGenerator
from ast_analyzer import ASTAnalyzer
from utils import load_config, load_db, save_db

class ProjectManager:
    def __init__(self, cfg_file: str = "config.yaml"):
        self.cfg = load_config(cfg_file)
        self.api_key = os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise EnvironmentError(
                "OPENAI_API_KEY environment variable is not set. Please set it or provide it in the config file."
            )
        self.api_key = self.cfg.get("api_key", self.api_key)
        self.db_file = self.cfg["db_file"]
        self.db = load_db(self.db_file)
        self.memory = {"projects": {}, "modules": {}}
        self.graph = nx.DiGraph()
        self.embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
        self.code_generator = CodeGenerator(
            self.api_key, self.cfg, self.embedding_model
        )
        self.ast_analyzer = ASTAnalyzer()

    def generate_project(self, idea: str) -> None:
        logging.info(f"Generating project for idea: {idea}")
        project_id = re.sub(r"\W+", "_", idea)[:50]
        self._initialize_project(project_id, idea)
        self.project_dir = os.path.join(os.getcwd(), project_id)
        logging.info(f"Project directory: {self.project_dir}")
        structure_prompt = self.cfg["prompt_template"]["struct"].format(
            idea=idea
        )
        struct = self.code_generator._generate_text(
            structure_prompt, n=1
        )[0].strip()
        self.db[project_id]["structure"] = struct
        self.memory["projects"][project_id] = {"structure": struct}
        logging.debug("Generated project structure:\n%s", struct)
        try:
            modules = self._parse_project_structure(struct)
            if not modules:
                logging.error(
                    "Failed to parse project structure. No modules found."
                )
                return
            project_context = self.db[project_id]["structure"]
            if not project_context:
                logging.error(
                    "Failed to generate project structure. Unable to generate modules."
                )
                return
            self._generate_modules(modules, project_id, project_context)
            self._generate_main_module(project_id, idea, struct)
        except Exception as e:
            logging.error(f"Error generating project: {e}")
            raise

    def _initialize_project(self, project_id: str, idea: str) -> None:
        if project_id not in self.db:
            self.db[project_id] = {"idea": idea, "files": {}}

    def _parse_project_structure(
        self, struct: str
    ) -> List[Tuple[str, str, str]]:
        modules: List[Tuple[str, str, str]] = []
        try:
            struct = struct.strip("```")
            loaded_struct = yaml.safe_load(struct)
            if not isinstance(loaded_struct, dict):
                loaded_struct = {"project": loaded_struct}
            project_data = loaded_struct.get("project", {})
            for module_info in project_data.get("modules", []):
                if isinstance(module_info, dict):
                    module_name = module_info.get("name")
                    description = module_info.get("description", "")
                    file_path = module_info.get(
                        "file", f"{module_name}.py"
                    )
                    module_path = os.path.join(
                        self.project_dir, file_path
                    )
                    os.makedirs(
                        os.path.dirname(module_path), exist_ok=True
                    )
                    self.code_generator._create_file(module_path)
                    modules.append(
                        (module_name, description, module_path)
                    )
                else:
                    logging.warning(
                        f"Invalid module format: {module_info}"
                    )
        except Exception as e:
            logging.error(
                f"Failed to parse project structure: {e}"
            )
        return modules

    def _generate_modules(
        self,
        modules: List[Tuple[str, str, str]],
        project_id: str,
        project_context: str,
    ) -> None:
        for name, desc, module_path in tqdm(
            modules, desc="Generating modules", unit="module"
        ):
            logging.info(f"Processing module: {name}")
            (
                code,
                summary,
                analysis,
            ) = self._generate_module_code(
                name, desc, project_id, project_context
            )
            if code:
                self._save_module_code(
                    project_id, module_path, code, summary, analysis
                )
                self._update_project_graph(
                    project_id, module_path, analysis
                )

    def _generate_module_code(
        self, name: str, desc: str, project_id: str, project_context: str
    ) -> Tuple[str, str, Dict[str, Any]]:
        (
            code,
            summary,
            analysis,
        ) = self.code_generator.generate_and_store_code(
            name, desc, project_context
        )
        if not code:
            logging.error(f"Failed to generate {name} module")
        logging.debug(f"Generated code for {name} module")
        return code, summary, analysis

    def _save_module_code(
        self,
        project_id: str,
        module_path: str,
        code: str,
        summary: str,
        analysis: Dict[str, Any],
    ) -> None:
        try:
            self.code_generator._create_file(module_path, code)
            module_name = os.path.basename(module_path).replace(
                ".py", ""
            )
            self.db[project_id]["files"][module_name] = {
                "code": code,
                "summary": summary,
                "analysis": analysis,
            }
            self.memory["modules"][module_path] = {
                "code": code,
                "summary": summary,
                "analysis": analysis,
            }
            logging.info(f"Generated module: {module_name}")
        except Exception as e:
            logging.error(
                f"Failed to save module code: {module_path}. Error: {e}"
            )

    def _generate_main_module(
        self, project_id: str, idea: str, struct: str
    ) -> None:
        logging.info("Generating main module")
        main_prompt = self.cfg["prompt_template"][
            "main"
        ].format(
            idea=idea,
            struct=struct,
            summaries=self._get_module_summaries(project_id),
        )
        main_code = self.code_generator._generate_text(
            main_prompt, n=1
        )[0].strip()
        main_path = os.path.join(self.project_dir, "main.py")
        self._save_module_code(
            project_id, main_path, main_code, "Main module", {}
        )
        logging.info(f"Generated main script: {main_path}")
        logging.info(f"Generated project in {self.project_dir}")

    def _get_module_summaries(self, project_id: str) -> str:
        summaries = []
        for module_path, module_data in self.memory[
            "modules"
        ].items():
            if project_id in module_path:
                summary = module_data["summary"]
                summaries.append(
                    f"{os.path.basename(module_path).replace('.py', '')}: {summary}"
                )
        return "\n".join(summaries)

    def _update_project_graph(
        self, project_id: str, module_path: str, analysis: Dict[str, Any]
    ) -> None:
        module_name = os.path.basename(module_path).replace(
            ".py", ""
        )
        self.graph.add_node(module_name)
        for import_stmt in analysis.get("imports", []):
            self.graph.add_node(import_stmt)
            self.graph.add_edge(module_name, import_stmt)

    def visualize_project_graph(self) -> None:
        import matplotlib.pyplot as plt

        plt.figure(figsize=(12, 8))
        pos = nx.spring_layout(self.graph, seed=42)
        nx.draw(
            self.graph,
            pos,
            with_labels=True,
            node_size=500,
            node_color="skyblue",
            font_size=10,
            font_weight="bold",
        )
        plt.title("Project Module Dependency Graph")
        plt.show()