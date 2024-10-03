import yaml
class Config:
    def __init__(self, config_path: str):
        with open(config_path, 'r') as file:
            self.config = yaml.safe_load(file)
        self.model_name = self.config['model_name']
        self.chunk_size = self.config['chunk_size']
        self.batch_size = self.config['batch_size']
        self.documents_path = self.config['documents_path']
        self.db_path = self.config['db_path']
        self.index_path = self.config['index_path']
        self.processed_files_path = self.config['processed_files_path']