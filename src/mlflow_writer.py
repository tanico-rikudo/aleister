import  os
import torch
import mlflow
mlflow.set_registry_uri(os.environ['MLFLOW_TRACKING_URI'] )
mlflow.set_tracking_uri(os.environ['MLFLOW_ARTIFACTS_URI'] )
from mlflow.tracking import MlflowClient
from mlflow.utils.mlflow_tags import MLFLOW_RUN_NAME,MLFLOW_USER,MLFLOW_SOURCE_NAME
    

class MlflowWriter():
    """
    MLflow tracking wrapper
    """

    def __init__(self,logger, kwargs):
        self.client = MlflowClient(**kwargs)
        self._logger = logger
        
    def create_experiment(self, experiment_name, run_tags):
        try:
            self.experiment_id = self.client.create_experiment(experiment_name)
        except:
            self.experiment_id = self.client.get_experiment_by_name(experiment_name).experiment_id

        self.run_id = self.client.create_run(self.experiment_id,tags=run_tags).info.run_id
        self.experiment = self.client.get_experiment(self.experiment_id)
        self._logger.info(f"New run started: {run_tags[MLFLOW_RUN_NAME]}")
        self._logger.info(f"Experiment Name: {self.experiment.name}")
        self._logger.info(f"Experiment id: {self.experiment.experiment_id}")
        self._logger.info(f"Artifact Location: {self.experiment.artifact_location}")
        self._logger.info("[DONE] Set up MLflow tracking")
        
    def register_model(self, model_name):
        """
        Register model into model  registory
        Args:
            model_name (str): uninque name 
        """
        try:
            self.client.create_registered_model(model_name)
        except Exception as e:
            self._logger.warn("[Failure] Cannot regist model. Name=f'{model_name}'. :{e}")
        pass
        
    def print_registered_model_info(self, rm):
        s = f"name: {rm.name}, \
            tags: {rm.tags}, \
            description: {rm.description}"
        print(s)
        return  s

        
    def create_model_version(self, model_name, desc=""):
        mv = self.client.create_model_version(
            name=model_name,
            source=os.path.join(self.experiment.artifact_location, self.run_id, "artifacts", "model"),
            run_id=self.run_id,
            description=desc
        )
        self._logger.info("[DONE] Create modek version")
        self._logger.info("Name: {}".format(mv.name))
        self._logger.info("Version: {}".format(mv.version))
        self._logger.info("Description: {}".format(mv.description))
        self._logger.info("Status: {}".format(mv.status))
        self._logger.info("Stage: {}".format(mv.current_stage))
        

    def log_params_from_omegaconf_dict(self, params):
        for param_name, element in params.items():
            self._explore_recursive(param_name, element)

    def _explore_recursive(self, parent_name, element):
        if isinstance(element, dict):
            for k, v in element.items():
                if isinstance(v, dict) or isinstance(v, list):
                    self._explore_recursive(f'{parent_name}.{k}', v)
                else:
                    self.client.log_param(self.run_id, f'{parent_name}.{k}', v)
        elif isinstance(element, list):
            for i, v in enumerate(element):
                self.client.log_param(self.run_id, f'{parent_name}.{i}', v)
        else:
            self.client.log_param(self.run_id, f'{parent_name}', element)

    def log_torch_model(self, model):
        with mlflow.start_run(self.run_id):
            pytorch.log_model(model, 'models')

    def log_param(self, key, value):
        self.client.log_param(self.run_id, key, value)

    def log_metric(self, key, value, step=None):
        self.client.log_metric(self.run_id, key, value, step)

    def log_artifact(self, local_path):
        self.client.log_artifact(self.run_id, local_path)

    def set_terminated(self):
        self.client.set_terminated(self.run_id)
       