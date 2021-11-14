from util.config import ConfigManager

cm = ConfigManager(os.environ['ALEISTER_INI'])

class BaseProcess():
    def __init__(self, _id, logger):
        self._logger  = logger
        self.id = _id

    def load_general_config(source="ini", path=None,dict_obj=None, mode=None):
        if source == "ini":
            self.general_config = cm.load_ini_config(path=path,config_name="general", mode=mode)
        elif source == "dict":
            self.general_config = cm.load_dict_config(dict_obj)[mode]
        self.save_dir = os.path.join(self.general_config.get("MODEL_SAVE_PATH"), self.id)
        self._logger.info('[DONE]Load General. Source={0}'.format(source))
        
    def load_model_config(source="ini", path=None,dict_obj=None, odel_name=None):
        if source == "ini":
            self.general_config = cm.load_ini_config(path=path,config_name="model", mode=mode)
        elif source == "dict":
            self.general_config = cm.load_dict_config(dict_obj)[mode]
        self._logger.info('[DONE]Load Model Config. Source={0}'.format(source))
        

        
    