import  os, sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from torch.autograd import Variable

from torch.utils.tensorboard import SummaryWriter

from sklearn.metrics import accuracy_score

from mlflow_writer import MlflowWriter

from datetime import datetime as dt
import numpy as np

from plotly.offline import plot
import plotly.graph_objects as go

from base_process import BaseProcess
from model_modules import *

from mlflow.utils.mlflow_tags import MLFLOW_RUN_NAME,MLFLOW_USER,MLFLOW_SOURCE_NAME

class LearningEvaluator(BaseProcess):
    def __init__(self, _id, mlflow_tags):
        super().__init__(_id)
        self.id = _id
        self.model = None
        self.device = "cpu"
        self.train_losses = []
        self.val_losses = []
        self.build_mlflow_run(**mlflow_tags)

        
    def build_mlflow_run(self, run_name=None, user=None, source=None):
        run_name = dt.now().strftime("%Y%m%d%H%M%s") if run_name is None else run_name
        user = 'ANONYMOUS' if user is None else user
        source = 'PYTHON' if source  is None else source
        self.mlflow_tags = {
                MLFLOW_RUN_NAME:run_name,
                MLFLOW_USER:user,
                MLFLOW_SOURCE_NAME:source,
            }
        
    def create_mlflow_run(self,tracking_uri=None):
        #todo set outside
        tracking_uri= "./tracking/mlruns"
        client_kwargs = {
            "tracking_uri":tracking_uri
        }
        self.mlwriter = MlflowWriter(self.id, self._logger, self.mlflow_tags, client_kwargs)
        
    def close_mlflow_run(self):
        self.mlwriter.set_terminated()
                
    def get_device(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self._logger.info('[DONE] Get device. Device={0}'.format(self.device ))
 
    def get_loss_fn(self, fn_name, fn_params):
        fns = {
            "bcelogitloss": nn.BCEWithLogitsLoss
        }   
        self.loss_fn=fns.get(fn_name.lower())(**fn_params)
        self.loss_fn_name = fn_name
        self._logger.info('[DONE] Get loss fn. Function={0}'.format(self.loss_fn ))
        
    def get_optimizer(self, optimizer_name, optim_params):
        optimizers = {
            "adam": optim.Adam
        }
        self.optimizer = optimizers.get(optimizer_name.lower())(**optim_params)
        self.optimizer_name = optimizer_name
        self._logger.info('[DONE] Get optimizer. Optimiser={0}'.format(self.optimizer ))
        
    def get_model_instance(self, model_name, model_params):
        models = {
            "sdnn": SimpleDnn
        }
        self.model  =  models.get(model_name.lower())(**model_params)
        self.model_name = model_name
        self._logger.info('[DONE]Load Model Instance.')
        
    def load_model_hparameters(self, model_name):
        hparams = {
            "sdnn": parameterParser.sdnn
        }
        self.model_name = model_name
        self.hparams =  hparams.get(model_name.lower())(self.model_config)
        self._logger.info('[DONE]Load hyper params.')
        

    def train_step(self, x, y):
        self.model.train()
        yhat = self.model(x)
        loss = self.loss_fn(y, yhat)
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def train(self, train_loader, val_loader, batch_size=64, n_epochs=50, n_features=1):        
        self._logger.info("[Start] Training. ID={0}".format(self.id))
        _ = self.get_model_save_path()  #todo : designate path
        
        # mlflow
        dict_config = {
            "batch_size":batch_size,
            "n_epochs":n_epochs,
            "optimizer":self.optimizer_name,
            "loss_fn":self.loss_fn_name
        }
        

        self.create_mlflow_run()
        self.mlwriter.log_params_from_omegaconf_dict(dict_config)
        
        # start
        for epoch in range(1, n_epochs + 1):
            batch_losses = []
            for x_batch, y_batch in train_loader:
                self.optimizer.zero_grad()
                # x_batch = x_batch.view([batch_size, -1, n_features]).to(device)
                x_batch = x_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                loss = self.train_step(x_batch, y_batch)
                batch_losses.append(loss)
            training_loss = np.mean(batch_losses)
            self.train_losses.append(training_loss)
            self.mlwriter.log_metric('train_loss',  training_loss, epoch)

            with torch.no_grad():
                batch_val_losses = []
                for x_val, y_val in val_loader:
                    # x_val = x_val.view([batch_size, -1, n_features]).to(device)
                    x_batch = x_batch.to(self.device)
                    y_val = y_val.to(self.device)
                    self.model.eval()
                    yhat = self.model(x_val)
                    val_loss = self.loss_fn(y_val, yhat).item()
                    batch_val_losses.append(val_loss)
                validation_loss = np.mean(batch_val_losses)
                self.val_losses.append(validation_loss)
                self.mlwriter.log_metric('valid_loss', validation_loss, epoch)

            if (epoch <= 10) | (epoch % 50 == 0):
                self._logger.info(
                    f"[{epoch}/{n_epochs}] Training loss: {training_loss:.4f}\t Validation loss: {validation_loss:.4f}"
                )
        self.save_model()
        self._logger.info("[DONE] Training. ID={0}".format(self.id))        
    
    def evaluate(self, test_loader, batch_size=1, n_features=1):
        with torch.no_grad():
            predictions = np.array([])
            truths  = np.array([])
            for x_test, y_test in test_loader:
                # x_test = x_test.view([batch_size, -1, n_features]).to(device)
                x_test = x_test.to(self.device)
                y_test = y_test.to(self.device)
                self.model.eval()
                yhat = self.model(x_test)
                predictions = np.append(predictions, yhat.to(self.device).detach().numpy())
                truths = np.append(truths, y_test.to(self.device).detach().numpy())
                
        # record acc 
        self.prediction = predictions
        self.truths = truths

        acc = accuracy_score(truths, predictions)
        scores={
            "accuracy":acc
        }
        self.mlwriter.log_metric('accuracy',  acc) 
        
        return predictions, truths, scores
    
    def prediction(x):
        with torch.no_grad():
            x = x.to(self.device)
            self.model.eval()
            yhat = self.model(x_test)
            pred = yhat.to(self.device).detach().numpy()
        return pred
    
    def get_model_save_path(self, _dir=None, _id=None):
        _id = self.id if _id is None else _id
        _dir = self.save_dir if _dir is None else _dir
        self.save_path = os.path.join(_dir, "torch_{0}_model.path".format(_id) )
        return self.save_path
        
    def save_model(self,save_path=None):
        save_path = self.save_path if save_path is None else save_path
        torch.save(self.model.to(self.device).state_dict(), save_path)
        self._logger.info("[DONE] Save Model. Path={0}".format(save_path))
        
    def load_model_weight(self, load_path=None):
        load_path = self.save_path if load_path is None else load_path
        self.model.load_state_dict(torch.load(load_path))#.to(self.device)
        self._logger.info("[DONE] Load Model. Path={0}".format(load_path))

    def plot_losses(self):
        """
        The method plots the calculated loss values for training and validation
        """
        train_traj = go.Scatter(
            x=list(range(len(self.train_losses))),
            y=self.train_losses,
            name="train"
        )
        val_traj = go.Scatter(
            x=list(range(len(self.train_losses))),
            y=self.val_losses,
            name="val"
        )
        data = [train_traj, val_traj]

        local_path = plot(data, filename = 'basic-line')
        self.mlwriter.log_artifact(local_path)
        
    def terminated(self):
        #todo: more sophisticate
        self.close_mlflow_run()