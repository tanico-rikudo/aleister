class BaseProcess():
    def __init__(self, _id, logger):
        self._logger  = logger
        self.id = _id

    def load_general_config(source="ini", path=None,dict_obj=None, mode=None):
        config_ini = configparser.ConfigParser()
        if source == "ini":
            path = '../ini/config.ini' if path is None else path
            self.general_config = config_ini.read(path, encoding='utf-8')[mode]
        elif source == "dict":
            self.general_config = parser.read_dict(dict_obj)[mode]
        self.save_dir = os.path.join(self.general_config.get("MODEL_SAVE_PATH"), self.id)
        self._logger.info('[DONE]Load General. Source={0}'.format(source))
        
    def load_model_config(source="ini", path=None,dict_obj=None, odel_name=None):
        config_ini = configparser.ConfigParser()
        if source == "ini":
            model_config_ini = configparser.ConfigParser()
            path = '../ini/model_config.ini' if path is None else path
            self.model_config = model_config_ini.read(path, encoding='utf-8')[model_name]
        elif source == "dict":          
            self.general_config = parser.read_dict(dict_obj)[mode]
        self._logger.info('[DONE]Load Model Config. Source={0}'.format(source))
        

        
    