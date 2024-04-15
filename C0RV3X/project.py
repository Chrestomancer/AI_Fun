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
from generation import InnovativeCodeGenerator
from tqdm import tqdm
import openai
from tenacity import retry, stop_after_attempt, wait_exponential

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

class AdvancedProjectManager:
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
        self.code_generator = InnovativeCodeGenerator(self.api_key, self.cfg, self.embedding_model)
        self.code_generator.graph = self.graph

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    def generate_project(self, idea):
        logging.info(f"Generating project for idea: {idea}")
        project_id = re.sub(r'\W+', '_', idea)[:50]
        if project_id not in self.db:
            self.db[project_id] = {'idea': idea, 'files': {}}
        self.project_dir = os.path.join(os.getcwd(), project_id)
        logging.info(f"Project directory: {self.project_dir}")

        # Generate project structure
        structure_prompt = self.cfg['prompt_template']['struct'].format(idea=idea)
        struct = self.code_generator._generate(structure_prompt, "struct", n=1)[0]
        self.db[project_id]['structure'] = struct
        self.memory['projects'][project_id] = {'structure': struct}
        logging.debug("Generated project structure:\n%s", struct)

        try:
            modules = self._parse_project_structure(struct)
            if not modules:
                logging.error("Failed to parse project structure. No modules found.")
                return

            project_context = self.db[project_id]['structure']
            if not project_context:
                logging.error("Failed to generate project structure. Unable to generate modules.")
                return

            self._generate_modules(modules, project_id, project_context, struct)
            self._generate_main_module(project_id, idea, struct)

        except openai.error.RateLimitError as e:
            logging.warning(f"OpenAI API rate limit exceeded: {e}. Retrying...")
            raise
        except Exception as e:
            logging.error(f"Error generating project: {e}")
            raise

    def _parse_project_structure(self, struct):
        struct = struct.strip('```')
        try:
            loaded_struct = yaml.load(struct, LineLoader)
            if not isinstance(loaded_struct, dict):
                loaded_struct = {'project': loaded_struct}

            project_data = loaded_struct.get('project', {})
            modules = []
            for module_info in project_data.get('modules', []):
                if isinstance(module_info, dict):
                    module_name = module_info.get('name')
                    description = module_info.get('description', '')
                    file_path = module_info.get('file', f"{module_name}.py")
                    module_path = os.path.join(self.project_dir, file_path)
                    os.makedirs(os.path.dirname(module_path), exist_ok=True)
                    self.code_generator._create_file(module_path)
                    modules.append((module_name, description, module_path))
                else:
                    logging.warning(f"Invalid module format: {module_info}")
            return modules
        except Exception as e:
            logging.error(f"Failed to parse project structure: {e}")
            return []

    def _generate_modules(self, modules, project_id, project_context, struct):
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.cfg.get('max_workers', os.cpu_count())) as executor:
            futures = []
            for name, desc, module_path in modules:
                logging.info(f"Processing module: {name}")
                future = executor.submit(self._generate_module_code, name, desc, project_id, project_context)
                futures.append(future)

            with tqdm(total=len(futures), desc="Generating modules", unit="module") as pbar:
                for future in concurrent.futures.as_completed(futures):
                    module_data = future.result()
                    if module_data:
                        code, summary, analysis = module_data[:3]
                        module_path = module_data[3]
                        self._save_module_code(project_id, module_path, code, summary, analysis)
                    pbar.update(1)

    def _generate_module_code(self, name, desc, project_id, project_context):
        module_path = os.path.join(self.project_dir, f"{name}.py")
        code, summary, analysis = self.code_generator.generate_and_store_code(name, desc, project_context)
        if not code:
            logging.error(f"Failed to generate {name} module")
            return None

        logging.debug(f"Generated code for {name} module")
        return code, summary, analysis, module_path

    def _save_module_code(self, project_id, module_path, code, summary, analysis):
        try:
            self.code_generator._create_file(module_path, code)
            module_name = Path(module_path).stem
            self.db[project_id]['files'][module_name] = {
                'code': code,
                'summary': summary,
                'analysis': analysis
            }
            self.memory['modules'][module_path] = {
                'code': code,
                'summary': summary,
                'analysis': analysis,
                'relation': ""
            }
            logging.info(f"Generated module: {module_name}")
            return True
        except Exception as e:
            logging.error(f"Failed to save module code: {module_path}. Error: {e}")
            return False

    def _generate_main_module(self, project_id, idea, struct):
        logging.info("Generating main module")
        main_prompt = self.cfg['prompt_template']['main'].format(
            idea=idea,
            struct=struct,
            summaries=self._get_module_summaries(project_id)
        )
        main_code = self.code_generator._generate(main_prompt, "code", n=1)[0]
        main_path = os.path.join(self.project_dir, "main.py")
        self._save_main_module_code(project_id, main_path, main_code)
        logging.info(f"Generated main script: {main_path}")
        logging.info(f"Generated project in {self.project_dir}")

    def _save_main_module_code(self, project_id, main_path, main_code):
        try:
            self.code_generator._create_file(main_path, main_code)
            self.db[project_id]['files']['main'] = main_code
            save_db(self.db, self.db_file)
            return True
        except Exception as e:
            logging.error(f"Failed to save main module code: {main_path}. Error: {e}")
            return False

    def _get_module_summaries(self, project_id):
        summaries = []
        for module_path, module_data in self.memory['modules'].items():
            if project_id in module_path:
                summary = module_data['summary']
                summaries.append(f"{Path(module_path).stem}: {summary}")
        return "\n".join(summaries)

    def complete_modules(self, modules, project_id):
        logging.info("Completing modules")
        project_context = self.db[project_id]['structure']
        for name, desc, module_path in modules:
            if not self._complete_module_generation(name, desc, project_id, project_context):
                break

        if not self.memory['modules']:
            logging.error("No modules were successfully generated")

        self._relate_modules(project_id)
        self._update_module_graphs(project_id)

    def _complete_module_generation(self, name, desc, project_id, project_context):
        module_path = os.path.join(self.project_dir, f"{name}.py")
        code, summary, analysis = self.code_generator.generate_and_store_code(name, desc, project_context)
        if not code:
            logging.error(f"Failed to generate {name} module")
            return False

        logging.debug(f"Generated code for {name} module")
        self._save_module_code(project_id, module_path, code, summary, analysis)
        return True

    def _relate_modules(self, project_id):
        module_paths = [path for path in self.memory['modules'].keys() if project_id in path]
        self.code_generator.relate_modules(module_paths)
        logging.info("Module relations established.")

    def _update_module_graphs(self, project_id):
        for module_path in self.memory['modules'].keys():
            if project_id in module_path:
                code = self.memory['modules'][module_path]['code']
                self.code_generator._update_graph(module_path, code)