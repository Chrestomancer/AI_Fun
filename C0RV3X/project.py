# project.py

import os
import logging
import re
import yaml
import concurrent.futures
from pathlib import Path
import networkx as nx
from sentence_transformers import SentenceTransformer
from utils import load_config, load_db, save_db
from generation import CodeGenerator

class LineLoader(yaml.SafeLoader):
    def construct_mapping(self, node, deep=False):
        mapping = super().construct_mapping(node, deep=deep)
        for key in mapping:
            if isinstance(key, yaml.nodes.ScalarNode):
                if key.value.startswith('**'):
                    mapping[key.value[2:].strip()] = mapping.pop(key)
        return mapping

def line_constructor(loader, node):
    if isinstance(node, yaml.MappingNode):
        mapping = loader.construct_mapping(node)
        for key in mapping:
            if isinstance(key, str) and key.startswith('**'):
                mapping[key[2:].strip()] = mapping.pop(key)
        return mapping
    else:
        return {}

LineLoader.add_constructor('!mapping', line_constructor)
yaml.add_constructor('!mapping', line_constructor)

class ProjectManager:
    def __init__(self, cfg_file="genius_config.json"):
        self.cfg = load_config(cfg_file)
        self.api_key = os.getenv('OPENAI_API_KEY')
        if not self.api_key:
            raise EnvironmentError("OPENAI_API_KEY environment variable is not set. Please set it or provide it in the config file.")
        self.api_key = self.cfg.get('api_key', self.api_key)
        self.db_file = self.cfg["db_file"]
        self.db = load_db(self.db_file)
        self.memory = {"projects": {}, "modules": {}}
        self.graph = nx.MultiDiGraph()
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.code_generator = CodeGenerator(self.api_key, self.cfg, self.embedding_model)
        self.code_generator.graph = self.graph

    def generate_project(self, idea):
        logging.info(f"Generating project for idea: {idea}")
        project_id = re.sub(r'\W+', '_', idea)[:50]
        if project_id not in self.db:
            self.db[project_id] = {'idea': idea, 'files': {}}
        self.project_dir = os.path.join(os.getcwd(), project_id)
        logging.info(f"Project directory: {self.project_dir}")

        # Generate project structure
        structure_prompt = self.cfg['prompt_template']['struct'].format(idea=idea)
        struct = self.code_generator._generate(structure_prompt, "struct")
        self.db[project_id]['structure'] = struct
        self.memory['projects'][project_id] = {'structure': struct}
        logging.debug(f"Generated project structure:\n{struct}")

        try:
            structure_data, error = self._parse_project_structure(struct)
            if error:
                logging.error(f"Failed to parse project structure: {error}")
                return
            if not structure_data:
                logging.error("Project structure is empty or invalid. Unable to generate modules.")
                return

            modules = []
            for module_name, module_info in structure_data.items():
                if isinstance(module_info, dict):
                    description = module_info.get('description', '')
                    file_path = module_info.get('file', f"{module_name}.py")
                    module_path = os.path.join(self.project_dir, file_path)
                    os.makedirs(os.path.dirname(module_path), exist_ok=True)
                    self.code_generator._create_file(module_path)
                    modules.append((module_name, description, module_path))
                else:
                    logging.warning(f"Invalid module format: {module_name}")

            project_context = self.db[project_id]['structure']
            if not project_context:
                logging.error("Failed to generate project structure. Unable to generate modules.")
                return

            with concurrent.futures.ThreadPoolExecutor(max_workers=self.cfg.get('max_workers', os.cpu_count())) as executor:
                futures = []
                for name, desc, module_path in modules:
                    logging.info(f"Processing module: {name}")
                    similar_modules = self.code_generator._find_similar_code(desc)
                    if similar_modules and similar_modules[0][1] >= 0.9:  # Adjust the similarity threshold as needed
                        logging.info(f"Skipping generation of {name} module due to high similarity with {similar_modules[0][0]}")
                        continue

                    future = executor.submit(self._generate_module_code, name, desc, project_id, project_context)
                    futures.append(future)

                for future in concurrent.futures.as_completed(futures):
                    module_data = future.result()
                    if module_data:
                        code, summary, analysis = module_data
                        module_name = Path(module_data[2]).stem
                        logging.debug(f"Saving module code: {module_name}")
                        self._save_module_code(project_id, module_data[2], code, summary, analysis)

            # Generate main module
            logging.info("Generating main module")
            main_prompt = self.cfg['prompt_template']['main'].format(
                idea=idea,
                struct=struct,
                summaries=self.code_generator._get_module_summaries(list(self.memory['modules'].keys()))
            )
            main_code = self.code_generator._generate(main_prompt, "code")
            main_path = os.path.join(self.project_dir, "main.py")
            self.code_generator._create_file(main_path, main_code)
            self.db[project_id]['files']['main'] = main_code
            save_db(self.db, self.db_file)
            logging.info("Generated main script.")
            logging.info(f"Generated project in {self.project_dir}")

        except Exception as e:
            logging.error(f"Error generating project: {str(e)}")
            raise

    def _parse_project_structure(self, struct):
        struct = struct.strip('```')
        struct = struct.strip('`')
        try:
            if struct.startswith('```') or struct.endswith('```'):
                struct = struct.strip('```')
            structure_data = {}
            for doc in yaml.load_all(struct, LineLoader):
                if isinstance(doc, dict):
                    structure_data.update(data)
                else:
                    logging.warning(f"Skipping invalid document: {doc}")
            return structure_data, None
        except yaml.YAMLError as e:
            error_msg = str(e)
            if hasattr(e, 'problem_mark'):
                mark = e.problem_mark
                error_msg += f" at line {mark.line + 1}, column {mark.column + 1}"
            return None, error_msg
        except Exception as e:
            error_msg = str(e)
            return None, error_msg

    def _generate_module_code(self, name, desc, project_id, project_context):
        module_path = os.path.join(self.project_dir, f"{name}.py")
        code, summary, analysis = self.code_generator._generate_module(name, desc, project_context)
        if not code:
            logging.error(f"Failed to generate {name} module")
            return None

        logging.debug(f"Generated code for {name} module")
        return code, summary, analysis, module_path

    def _save_module_code(self, project_id, module_path, code, summary, analysis):
        try:
            self.code_generator._create_file(module_path, code)
            module_name = Path(module_path).stem
            self.db[project_id]['files'][module_name] = code
            self.memory['modules'][module_path] = {'code': code, 'summary': summary, 'analysis': analysis, 'relation': ""}
            logging.info(f"Generated module: {module_name}")
            return True
        except Exception as e:
            logging.error(f"Failed to save module code: {module_path}. Error: {str(e)}")
            return False

    def complete_modules(self, modules, project_id):
        logging.info("Completing modules")
        project_context = self.db[project_id]['structure']
        for name, desc, module_path in modules:
            if not self._complete_module_generation(name, desc, project_id, project_context):
                break

        if not self.memory['modules']:
            logging.error("No modules were successfully generated")

        # Relate modules
        module_paths = list(self.memory['modules'].keys())
        self.code_generator.relate_modules(module_paths)
        logging.info("Module relations established.")

        for module_path in self.code_generator.memory:
            code = self.code_generator.memory[module_path]['code']
            self.code_generator._update_graph(module_path, code)