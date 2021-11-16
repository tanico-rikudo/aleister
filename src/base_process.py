import os
from util.config import ConfigManager

cm = ConfigManager(os.environ['ALEISTER_INI'])
LOGDIR=os.environ['LOGDIR']
class BaseProcess():
    def __init__(self, _id):
        self._logger = cm.load_log_config(os.path.join(LOGDIR,'logging.log'),log_name="ALEISTER")
        self.id = _id
        

    def load_general_config(self, source="ini", path=None,dict_obj=None, mode=None):
        if source == "ini":
            self.general_config = cm.load_ini_config(path=path,config_name="general", mode=mode)
        elif source == "dict":
            self.general_config = cm.load_dict_config(dict_obj)[mode]
        self.save_dir = os.path.join(self.general_config.get("MODEL_SAVE_PATH"), self.id)
        self._logger.info('[DONE]Load General. Source={0}'.format(source))
        
    def load_model_config(self, source="ini", path=None,dict_obj=None, model_name=None):
        if source == "ini":
            self.model_config = cm.load_ini_config(path=path,config_name="model", mode=model_name)
        elif source == "dict":
            self.model_config = cm.load_dict_config(dict_obj)[model_name]
        self._logger.info('[DONE]Load Model Config. Source={0}'.format(source))
        

        
    