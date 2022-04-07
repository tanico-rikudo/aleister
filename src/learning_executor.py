import os, sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from torch.autograd import Variable

from torch.utils.tensorboard import SummaryWriter

from sklearn.metrics import accuracy_score

from mlflow_writer import *

from model.parameter_parser import parameterParser as pp
from model import dnn, lstm, cgm

from datetime import datetime as dt
import numpy as np
import pandas as pd
import tempfile

from plotly.offline import plot
import plotly.graph_objects as go

import mlflow

sys.path.append(os.environ["COMMON_DIR"])
from base_process import BaseProcess


class LearningEvaluator(BaseProcess):
    def __init__(self, _id, model_name):
        super().__init__(_id)
        self.id = _id
        self.model_name = model_name
        self.model = None
        self.device = "cpu"

        self.train_losses = []
        self.val_losses = []

        self.mlwriter = None

    def build_mlflow(self, mlwriter, mlflow_tags):
        self.mlwriter = mlwriter
        self.mlflow_tags = mlflow_tags

    def get_device(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self._logger.info('[DONE] Get device. Device={0}'.format(self.device))

    def get_loss_fn(self, fn_name, fn_params):
        fns = {
            "bcelogitloss": nn.BCEWithLogitsLoss
        }
        self.loss_fn = fns.get(fn_name.lower())(**fn_params)
        self.loss_fn_name = fn_name
        self._logger.info('[DONE] Get loss fn. Function={0}'.format(self.loss_fn))

    def get_optimizer(self, optimizer_name, optim_params):
        optimizers = {
            "adam": optim.Adam
        }
        self.optimizer = optimizers.get(optimizer_name.lower())(**optim_params)
        self.optimizer_name = optimizer_name
        self._logger.info('[DONE] Get optimizer. Optimiser={0}'.format(self.optimizer))

    def get_model_instance(self, model_name, model_params):
        models = {
            "sdnn": dnn.SimpleDnn,
            "slstm": lstm.SimpleLSTM,
            "cgm": cgm.CGM
        }
        self._logger.info('Load Model Instance.  Stucture params={0}'.format(model_params.keys()))
        self.model = models.get(model_name.lower())(**model_params)
        self._logger.info('[DONE] Load Model Instance.  Stucture params={0}'.format(model_params))

    def load_model_hparameters(self, model_name, source='ini'):
        hparams = {
            "sdnn": pp.sdnn,
            "slstm": pp.slstm,
            "cgm": pp.cgm
        }
        self.hparams = hparams.get(model_name.lower())(self.model_config, source)
        self._logger.info(f'[DONE]Load hyper params. Name={self.model_name}, Source={source}')

    def convert_model_value(self, y):
        y_pred_idxs = map(lambda yy: yy.argmax(), y)
        y_pred = list(map(lambda yy: self.out_class[yy], y_pred_idxs))
        return y_pred

    def train_step(self, xs, ys):
        xs = [x.to(self.device) for x in xs]
        ys = ys.to(self.device)
        self.model.train()
        yhat = self.model(*xs)
        loss = self.loss_fn(ys, yhat)
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def eval_step(self, xs, ys=None):
        xs = [x.to(self.device) for x in xs]
        self.model.eval()
        yhat = self.model(*xs)
        prediction = yhat.to(self.device).detach().numpy()
        if ys is not None:
            ys = ys.to(self.device)
            loss = self.loss_fn(ys, yhat).item()
            truth = ys.to(self.device).detach().numpy()
        else:
            truth, loss = None, None

        return prediction, truth, loss


    def train(self, train_loader, val_loader, batch_size=64, n_epochs=10, n_features=1):
        """[summary]

        Args:
            train_loader ([type]): [description]
            val_loader ([type]): [description]
            batch_size (int, optional): [description]. Defaults to 64.
            n_epochs (int, optional): [description]. Defaults to 50.
            n_features (int, optional): [description]. Defaults to 1.

        Returns:
            str: mlflow runid
        """
        self._logger.info("[Start] Training. ID={0}".format(self.id))
        # _ = self.get_model_save_path()  #todo : designate path

        # setup
        self.prediction = {}
        self.predictions_out = {}
        self.truths = {}
        self.truths_out = {}
        self.out_class = eval(self.model_config.get("OUT_CLASS"))

        # mlflow
        dict_config = {
            "batch_size": batch_size,
            "n_epochs": n_epochs,
            "optimizer": self.optimizer_name,
            "loss_fn": self.loss_fn_name
        }

        self.mlwriter.create_experiment(self.id, self.mlflow_tags)
        self.mlwriter.log_params_from_omegaconf_dict(dict_config)

        # start
        for epoch in range(1, n_epochs + 1):
            batch_losses = []
            self.train_loader = train_loader
            for x_batchs, y_batchs in train_loader:
                self.optimizer.zero_grad()
                loss = self.train_step(x_batchs, y_batchs)
                batch_losses.append(loss)
            training_loss = np.mean(batch_losses)
            self.train_losses.append(training_loss)
            self.mlwriter.log_metric('train_loss', training_loss, epoch)

            with torch.no_grad():
                batch_val_losses = []
                predictions = []
                truths = []
                for x_vals, y_val in val_loader:
                    prediction, truth, val_loss = self.eval_step(x_vals, y_val)
                    predictions.append(prediction)
                    truths.append(truth)
                    batch_val_losses.append(val_loss)

                validation_loss = np.mean(batch_val_losses)
                self.val_losses.append(validation_loss)

                #  stacked numpy data  to nu
                predictions = np.concatenate(predictions)
                truths = np.concatenate(truths)

                # record
                self.prediction["val"], self.truths["val"] = predictions, truths
                predictions_out, truths_out = self.convert_model_value(predictions), self.convert_model_value(truths)
                self.predictions_out["val"], self.truths_out["val"] = predictions_out, truths_out
                acc = accuracy_score(truths_out, predictions_out)

                self.mlwriter.log_metric('valid_acc', acc, epoch)
                self.mlwriter.log_metric('valid_loss', validation_loss, epoch)
                # print(self.prediction["val"])
                # print(self.predictions_out["val"])

            if (epoch <= 10) | (epoch % 50 == 0):
                self._logger.info(
                    f"[{epoch}/{n_epochs}] Training loss: {training_loss:.4f}\t Validation loss: {validation_loss:.4f}"
                )

        # record last 

        self.save_numpy_obj(predictions, "val_preds.csv")
        self.save_numpy_obj(truths, "val_truth.csv")
        self.save_numpy_obj(predictions_out, "val_preds_out.csv", type='int')
        self.save_numpy_obj(truths_out, "val_truth_out.csv", type='int')

        # self.save_model()
        self.save_mlflow_model()
        self._logger.info("[DONE] Training. ID={0}".format(self.id))

        return self.mlwriter.run_id

    def evaluate(self, test_loader):
        self._logger.info("[Start] Evaluate. ID={0}".format(self.id))
        with torch.no_grad():
            predictions = []
            truths = []
            self.test_loader = test_loader
            for x_tests, y_test in test_loader:
                prediction, truth, val_loss = self.eval_step(x_tests,y_test)
                predictions.append(prediction)
                truths.append(truth)

        predictions = np.concatenate(predictions)
        truths = np.concatenate(truths)

        # record 
        self.prediction["eval"], self.truths["eval"] = predictions, truths
        predictions_out, truths_out = self.convert_model_value(predictions), self.convert_model_value(truths)
        self.predictions_out["eval"], self.truths_out["eval"] = predictions_out, truths_out

        acc = accuracy_score(truths_out, predictions_out)
        self._logger.info(f"Evaluate: acc score={acc}")

        if self.mlwriter is not None:
            self.mlwriter.log_metric('test_acc', acc)
            self.save_numpy_obj(predictions, "eval_preds.csv")
            self.save_numpy_obj(truths, "eval_truth.csv")
            self.save_numpy_obj(predictions_out, "eval_preds_out.csv", type='int')
            self.save_numpy_obj(truths_out, "eval_truth_out.csv", type='int')

        self._logger.info("[END] Evaluate. ID={0}".format(self.id))
        return predictions, truths

    def save_numpy_obj(self, obj, filename, type='float'):
        if type in 'int':
            fmt = '%d'
        if type in 'float':
            fmt = '%f'
        with tempfile.TemporaryDirectory() as tdname:
            path = os.path.join(tdname, filename)
            np.savetxt(path, obj, delimiter=',', fmt=fmt)
            # self._logger.info(f"[DONE] Save obj to tmp: {path}")
            save_path = self.mlwriter.log_artifact(path)
            self._logger.info("[DONE] Save obj. Filename={0}".format(filename))

    def predict(self, data_loader):

        self._logger.info("[Start] Predict. ID={0}".format(self.id))
        self.prediction = {}
        self.out_class = eval(self.model_config.get("OUT_CLASS"))
        self.predictions_out = {}
        with torch.no_grad():
            predictions = []
            self.test_loader = data_loader
            for x_tests, y_test in data_loader:
                prediction, _, val_loss = self.eval_step(x_tests)
                predictions.append(prediction)

        # print(predictions)
        predictions = np.concatenate(predictions)

        # record
        self.prediction["eval"] = predictions
        predictions_out = self.convert_model_value(predictions)
        self.predictions_out["eval"] = predictions_out

        self._logger.info("[END] Predict. ID={0}".format(self.id))
        return predictions

    # def get_model_save_path(self, _dir=None, _id=None):
    #     _id = self.id if _id is None else _id
    #     _dir = self.save_dir if _dir is None else _dir
    #     self.save_path = os.path.join(_dir, "torch_{0}_model.path".format(_id) )
    #     return self.save_path

    # def save_model(self,save_path=None, module='pytorch'):
    #     save_path = self.save_path if save_path is None else save_path
    #     torch.save(self.model.to(self.device).state_dict(), save_path)
    #     self._logger.info("[DONE] Save Model. Path={0}".format(save_path))

    def save_mlflow_model(self):
        with tempfile.TemporaryDirectory() as tdname:
            pytorch_model_path = os.path.join(tdname, "model")
            self._logger.info(f"[DONE] Save Model to tmp: {pytorch_model_path}")
            mlflow.pytorch.save_model(self.model, pytorch_model_path)
            save_path = self.mlwriter.log_artifact(pytorch_model_path)
            self._logger.info("[DONE] Save Model. Path={0}".format(save_path))

    def load_mlflow_model(self, model_uri):
        # model_uri = os.path.join(self.mlwriter.experiment.artifact_location,"model")
        # name="tttt"
        self._logger.info("[DONE] Load Model. Path={0}".format(model_uri))
        self.model = mlflow.pytorch.load_model(model_uri)
        self._logger.info("[DONE] Load Model. Path={0}".format(model_uri))

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

        with tempfile.TemporaryDirectory() as tdname:
            path = os.path.join(tdname, "epoch_trajectory.html")
            local_path = plot(data, filename=path, auto_open=False)
            self._logger.info(f"[DONE] Save plot to tmp: {local_path}")
            save_path = self.mlwriter.log_artifact(local_path)
            self._logger.info("[DONE] Save plot. Path={0}".format(save_path))

    def terminated(self):
        # todo: more sophisticate
        self.mlwriter.set_terminated()
