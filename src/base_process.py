class BaseProcess():
    def __init__(self, _id, logger):
        self._logger  = logger
        self.id = _id

    def load_general_config(path=None,mode=None):
        config_ini = configparser.ConfigParser()
        path = '../ini/config.ini' if path is None else path
        self.general_config = config_ini.read(path, encoding='utf-8')[mode]
        self.save_dir = os.path.join(self.general_config.get("MODEL_SAVE_PATH"), self.id)
        self._logger.info('[DONE]Load General Config.')
        
    def load_model_config(path=None,model_name=None):
        model_config_ini = configparser.ConfigParser()
        path = '../ini/model_config.ini' if path is None else path
        self.model_config = model_config_ini.read(path, encoding='utf-8')[model_name]
        self._logger.info('[DONE]Load Model Config.')
        