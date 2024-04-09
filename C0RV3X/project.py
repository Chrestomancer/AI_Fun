import os
import logging
import re
from pathlib import Path
import networkx as nx
from sentence_transformers import SentenceTransformer
from utils import load_config, load_db, save_db
from generation import CodeGenerator

class ProjectManager:
    def __init__(self, cfg_file="genius_config.json"):
        self.cfg = load_config(cfg_file)
        self.api_key = os.getenv('OPENAI_API_KEY')
        if not self.api_key:
            logging.error("OPENAI_API_KEY environment variable is not set.")
            raise EnvironmentError("Set OPENAI_API_KEY environment variable or provide it in the config file.")
        if 'api_key' in self.cfg:
            self.api_key = self.cfg['api_key']
        elif not self.api_key:
            logging.error("OPENAI_API_KEY is not set in the environment or the config file.")
            raise EnvironmentError("Set OPENAI_API_KEY environment variable or provide it in the config file.")
        self.db_file, self.state, self.project_dir = self.cfg["db_file"], "idle", None
        self.memory, self.graph = {"projects": {}, "modules": {}}, nx.MultiDiGraph()
        self.db = load_db(self.db_file)
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.code_generator = CodeGenerator(self.api_key, self.cfg, self.embedding_model)
        self.code_generator.graph = self.graph

    def generate_project(self, idea):
        project_id = re.sub(r'\W+', '_', idea)[:50]
        if project_id not in self.db:
            self.db[project_id] = {'idea': idea, 'files': {}}
        self.project_dir = os.path.join(os.getcwd(), project_id)
        print(f"Generating project: {project_id}")
        # Generate project structure
        structure_prompt = self.cfg['prompt_template']['struct'].format(idea=idea)
        struct = self.code_generator._generate(structure_prompt, "struct")
        self.db[project_id]['structure'] = struct
        self.memory['projects'][project_id] = {'structure': struct}
        print(f"Project structure:\n{struct}")

        modules = []  # Store module information
        for name, desc in re.findall(r'(\w+):\s*(.*?)(?=\s+\w+:|$)', struct, re.DOTALL):
            modules.append((name, desc))
            module_path = os.path.join(self.project_dir, f"{name}.py")
            self.code_generator._create_file(module_path)  # Create empty file
        project_context = self.db[project_id]['structure']
        if not project_context:
            logging.error("Failed to generate project structure. Unable to generate modules.")
            return
        self.complete_modules(modules, project_id)  # Complete modules iteratively

        # Generate main module
        main_prompt = self.cfg['prompt_template']['main'].format(
            idea=idea, struct=struct, summaries=self.code_generator._get_module_summaries(list(self.memory['modules'].keys()))
        )
        main_code = self.code_generator._generate(main_prompt, "code")
        main_path = os.path.join(self.project_dir, "main.py")
        self.code_generator._create_file(main_path, main_code)
        self.db[project_id]['files']['main'] = main_code
        save_db(self.db, self.db_file)
        print(f"Generated main script.")
        logging.info(f"Generated project in {self.project_dir}")

    def _complete_module_generation(self, name, desc, project_id, project_context):
        """
        Generate, analyze, summarize, refine, and create a module.

        Args:
            name (str): The name of the module.
            desc (str): The description of the module.
            project_id (str): The ID of the project.
            project_context (str): The context of the project.

        Returns:
            bool: True if the module was generated successfully, False otherwise.
        """
        logging.info(f"Generating {name} module...")
        module_path = os.path.join(self.project_dir, f"{name}.py")
        code, summary, analysis = self.code_generator._generate_module(name, desc, project_context)
        if not code:
            logging.error(f"Failed to generate {name} module")
            return False
        self.db[project_id]['files'][name] = code
        self.memory['modules'][module_path] = {'code': code, 'summary': summary, 'analysis': analysis, 'relation': ""}
        context = self.code_generator._get_module_context(module_path)
        code, summary, analysis = self.code_generator._refine_code(code, summary, analysis, context)
        self.code_generator._create_file(module_path, code)
        self.memory['modules'][module_path] = {'code': code, 'summary': summary, 'analysis': analysis,
                                              'relation': ""}
        print(f"Generated module: {name}")
        return True

    def complete_modules(self, modules, project_id):
        project_context = self.db[project_id]['structure']
        for name, desc in modules:
            if not self._complete_module_generation(name, desc, project_id, project_context):
                break

        if not self.memory['modules']:
            logging.error("No modules were successfully generated")
        # Relate modules
        module_paths = list(self.memory['modules'].keys())
        self.code_generator.relate_modules(module_paths)
        print("Module relations established.")

        for module_path in self.code_generator.memory:
            code = self.code_generator.memory[module_path]['code']
            self.code_generator._update_graph(module_path, code)
