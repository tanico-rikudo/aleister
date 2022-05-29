import os
from util.config import ConfigManager

cm = ConfigManager(os.environ['ALEISTER_INI'])
LOGDIR = os.environ['ALEISTER_LOGDIR']


class BaseProcess():
    def __init__(self, _id):
        self._logger = cm.load_log_config(os.path.join(LOGDIR, 'logging.log'), log_name="ALEISTER")
        self.id = _id

    def load_general_config(self, source="ini", path=None, dict_obj=None, mode=None):
        if source == "ini":
            self.general_config = cm.load_ini_config(path=path, config_name="general", mode=mode)
        elif source == "dict":
            self.general_config = cm.load_dict_config(dict_obj)[mode]
        self.save_dir = os.path.join(self.general_config.get("MODEL_SAVE_PATH"), self.id)
        self.tz = self.general_config.get("TIMEZONE")
        self._logger.info('[DONE]Load General. Source={0}'.format(source))


    def load_model_config(self, source="ini", path=None, dict_obj=None, model_name=None, omegaconf=None, replace=True):
        if source == "ini":
            model_config = cm.load_ini_config(path=path, config_name="model", mode=model_name)
        elif source == "dict":
            model_config = cm.load_dict_config(dict_obj)[model_name]
        elif source == 'yaml':
            model_config = cm.load_yaml_config(path=path, config_name="hparams", mode=model_name)
        else:
            raise Exception("Fail to load model config. Source={0}".format(source))
        self._logger.info('[DONE]Load Model Config. Source={0}'.format(source))

        if replace:
            self.model_config = model_config
        else:
            return model_config

    def load_postgres_ini(self):
        self.postgres_ini = cm.load_ini_config(
            path=None, config_name="postgres", mode=None
        )



