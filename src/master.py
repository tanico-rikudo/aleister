import os, sys
import logging
import logging.config
import argparse
from sklearn.model_selection import ParameterGrid
from mlflow_writer import MlflowWriter
from mlflow.utils.mlflow_tags import MLFLOW_RUN_NAME, MLFLOW_USER, MLFLOW_SOURCE_NAME
from datetime import datetime as dt

sys.path.append(os.path.join(os.environ['KULOKO_DIR'], "items"))
sys.path.append(os.path.join(os.path.dirname('__file__'), '..'))
os.environ['INIDIR'] = os.environ['ALEISTER_INI']
INIDIR = os.environ['INIDIR']
LOGDIR = os.environ['ALEISTER_LOGDIR']

from gen_data import DataGen
from feature_preprocess import featurePreprocess
from learning_executor import LearningEvaluator
from model.parameter_parser import parameterParser as pp

sys.path.append(os.environ['COMMON_DIR'])
from util.config import ConfigManager

cm = ConfigManager(os.environ['ALEISTER_INI'])
logger = cm.load_log_config(os.path.join(LOGDIR, 'logging.log'), log_name="ALEISTER")

global fp
global le


# todo: CLass.  it waste to connect   every  time....
class OperateMaster:
    def __init__(self):
        self.dg = None
        self.fp = None
        self.le = None
        self._id = None
        self.general_config_mode = None
        self.private_api_mode = None

    def load_meta(self, _id, model_name, sym, general_config_mode, private_api_mode):
        self.id = _id
        self.sym = sym
        self.model_name = model_name
        self.general_config_mode = general_config_mode
        self.private_api_mode = private_api_mode

    def init_mlflow(self):
        self.mlwriter = MlflowWriter(self.le._logger, self.mlflow_client_kwargs)
        self.mlflow_tags = self.mlwriter.build_mlflow_tags(self.mlflow_tags)
        self.le.build_mlflow(self.mlwriter, self.mlflow_tags)

    def init_prepro(self):
        self.fp = featurePreprocess(self.id)
        self.fp.load_general_config(source="ini", path=None, mode=self.general_config_mode)
        self.fp.load_model_config(source="ini", path=None, model_name=self.model_name)

    def init_learning(self):
        self.le = LearningEvaluator(self.id, self.model_name)
        self.init_mlflow()
        self.le.get_device()
        self.le.load_general_config(source="ini", path=None, mode=self.general_config_mode)

    def init_dataGen(self, remote=False):
        self.dg = DataGen(self.sym, self.general_config_mode, self.private_api_mode, self.fp._logger)
        if remote:
            self.dg.init_mqclient()

    def set_mlflow_settings(self, mlflow_client_kwargs, mlflow_tags):
        self.mlflow_tags = mlflow_tags
        self.mlflow_client_kwargs = mlflow_client_kwargs

    def update_mlflow_tags(self, key, val):
        self.mlflow_tags[key] = val

    def realtime_preprocessing(self, general_config, logger, scaler):

        
        # Fetch real data vim mq
        datas = self.dg.fetch_realdata()
        trades = datas["trade"]
        orderbooks = datas["orderbook"]
        self.fp._logger.info("[DONE] Get prepro raw data")

        # prepro
        X = self.dg.get_Xy(trades, orderbooks)
        X, _ = self.fp.feature_label_split(df=X, target_col=ans_col)
        X, _ = self.fp.scalingX(X, scaself.ler)

        return X

    def realtime_predict(self):
        scaler = self.fp.load_numpy_datas(["x_scaler"])
        
        # Fetch all hist
        # TODO
        # Fetch real data vim mq
        # TODO
        X = self.realtime_preprocessing(self.fp.general_config, self.fp._logger, scaler)

        # todo: mode conider
        uri = self.load_prod_model()
        self.le.load_mlflow_model(uri)
        prediction = self.le.prediction(X)
        # todo: save somwwhre
        return prediction

    def preprocessing(self, sym, train_start, train_end, valid_start, valid_end, test_start, test_end):
        # gen data 
        self.fp._logger.info(f"{train_start}, {train_end}, {valid_start}, {valid_end}, {test_start}, {test_end}")
        fetch_start = min([_date for _date in [train_start, train_end, valid_start, valid_end, test_start, test_end] if
                           _date is not None])
        fetch_end = max([_date for _date in [train_start, train_end, valid_start, valid_end, test_start, test_end] if
                         _date is not None])
        trades = self.dg.get_hist_data(ch="trades", sym=sym, sd=fetch_start, ed=fetch_end)
        orderbooks = self.dg.get_hist_data(ch="trades", sym=sym, sd=fetch_start, ed=fetch_end)
        Xy = self.dg.get_Xy(trades, orderbooks)
        self.fp._logger.info("[DONE] Get prepro raw data. {0}~{1}".format(fetch_start, fetch_end))

        # prepro
        ans_col = self.fp.model_config.get("ANS_COL")
        X, y = self.fp.feature_label_split(df=Xy, target_col=ans_col)
        y = self.fp.onehot_y(y)

        # split period
        # Note: If other kind data , such as attribute data linked to timeseries, will be added,
        #  train_val_test_period_split  can process it. Then, Input [X, data1, data2, y].
        train_datas, val_datas, test_datas = \
            self.fp.train_val_test_period_split([X, y], train_start, train_end, valid_start, valid_end, test_start,
                                                test_end)

        # assign and scaling
        # Note: If other data except  for X is using. Fix fetching train_datas.
        X_trains = train_datas[0].values
        X_trains, x_scaler = self.fp.scalingX(X_trains, scaler_name="minmax")
        y_train = train_datas[-1].values

        if val_datas is not None:
            X_vals = val_datas[0].values
            X_vals, _ = self.fp.scalingX(X_vals, x_scaler)
            y_val = val_datas[-1].values
        else:
            X_val, y_val = None, None

        if test_datas is not None:
            X_tests = test_datas[0].value
            X_tests, _ = self.fp.scalingX(X_tests, x_scaler)
            y_test = test_datas[-1].values
        else:
            X_tests, y_test = None, None

        #  Note: If other data except  for X is using. Fix X_train": X_trains, "X_val":X_vals, "X_test": X_tests
        self.fp.save_numpy_datas(**{
            "X_train": X_trains, "X_val": X_vals, "X_test": X_tests,
            "y_train": y_train, "y_val": y_val, "y_test": y_test,
            "x_scaler": x_scaler
        })

    def train_worker(self, X_trains, X_vals, y_train, y_val):

        # train set up
        input_dim = X_trains[0].shape[1]
        model_params = {_k: self.le.hparams[_k] for _k in self.le.hparams["structure_params"]}
        model_params['input_dim'] = input_dim

        self.le.get_model_instance(self.le.model_name, model_params)

        self.fp.get_dataset_fn(self.le.hparams["dataset"])
        dataset_params = {_k: self.le.hparams[_k] for _k in self.le.hparams["dataset_params"]}
        train_loader, val_loader, _, _ = self.fp.get_dataloader(self.le.hparams["dataset"], X_trains, y_train, X_vals,
                                                                y_val, None, None, **dataset_params)

        lossfn_params = {}
        self.le.get_loss_fn(self.le.hparams["optimizer"], {})

        optim_params = {
            'params': self.le.model.parameters(),
            'weight_decay': self.le.hparams["weight_decay"],
            'lr': self.le.hparams["lr"]
        }

        self.le.get_optimizer(self.le.hparams["loss_fn"], optim_params)

        # train
        self.le.train(train_loader, val_loader,
                      batch_size=self.le.hparams["batch_size"], n_epochs=self.le.hparams["n_epoch"],
                      n_features=1)
        self.le.plot_losses()

        # decline end
        self.le.terminated()

        # validation if data exist
        self.test_out_of_data()

    def train(self):
        obj_keys = ["X_train", "X_val", "y_train", "y_val"]
        X_train, X_val, y_train, y_val = self.fp.load_numpy_datas(obj_keys)
        X_trains = [X_train]
        X_vals = [X_val]

        self.le.load_model_config(source="ini", path=None, model_name=self.le.model_name)
        self.le.load_model_hparameters(self.le.model_name)

        self.train_worker(X_trains, X_vals, y_train, y_val)

    def gtrain(self):
        obj_keys = ["X_train", "X_val", "y_train", "y_val"]
        X_train, X_val, y_train, y_val = self.fp.load_numpy_datas(obj_keys)
        X_trains = [X_train]
        X_vals = [X_val]

        model_config = self.le.load_model_config(source="yaml", path=None, model_name=self.le.model_name, replace=False)
        param_grid = ParameterGrid(model_config)

        original_run_name = self.mlflow_tags[MLFLOW_RUN_NAME]
        for i, g in enumerate(param_grid):
            g = {self.le.model_name: g}
            self.le.load_model_config(source="dict", dict_obj=g, model_name=self.le.model_name, replace=True)
            self.le.load_model_hparameters(self.le.model_name)
            # new experiment
            self.update_mlflow_tags(MLFLOW_RUN_NAME, f"{original_run_name}_{i}")
            self.set_mlflow_settings(self.mlflow_client_kwargs, self.mlflow_tags)
            self.train_worker(X_trains, X_vals, y_train, y_val)

    def test_out_of_data(self):
        obj_keys = ["X_test", "y_test"]
        X_test, y_test = self.fp.load_numpy_datas(obj_keys)
        if X_test is None:
            self.fp._logger.info("No test data.")
            return
        X_tests = [X_test]

        input_dim = X_test.shape[1]
        batch_size = self.le.hparams["batch_size"]

        self.fp.get_dataset_fn(self.le.hparams["dataset"])
        dataset_params = {_k: self.le.hparams[_k] for _k in self.le.hparams["dataset_params"]}
        _, _, test_loader, test_loader_one = self.fp.get_dataloader(self.le.hparams["dataset"], X_trains=None,
                                                                    y_train=None, X_vals=None, y_val=None,
                                                                    X_tests=X_tests, y_test=y_test, **dataset_params)
        # print(test_loader_one)
        lossfn_params = {}
        self.le.get_loss_fn(self.le.hparams["optimizer"], {})
        predictions, values = self.le.evaluate(test_loader_one)

        # decline end
        self.le.terminated()

    def deploy_best_model(self):

        # pick best model 
        model_name = self.id
        experiment_id = self.mlwriter.get_experiment_id(self.id)
        best_runs = self.mlwriter.client.search_runs(
            experiment_ids=[experiment_id],
            max_results=1,
            order_by=["metrics.accuracy DESC"]
        )
        if len(best_runs) == 0:
            self.le._logger.info(f"[END] No runs. Experiment Name={self.id}, Id={experiment_id}")
            return
        else:
            best_run = best_runs[0]

        best_run_id = best_run.info.run_id

        # register model 
        self.mlwriter.register_model(model_name)

        # register(if no momdel)
        mv = self.mlwriter.create_model_version(model_name, experiment_id, best_run_id)

        # move on to prod
        self.mlwriter.deploy_model_to_production(model_name, mv.version)

    def load_prod_model(self):
        model_name = self.id
        prod_mvs = []
        mvs = self.mlwriter.search_staged_models(model_name)
        for mv in mvs:
            if mv.current_stage == 'Production':
                prod_mvs.append(mv)
        if len(prod_mvs):
            raise Exception("No Model on Prod Stage.")
        run_id = prod_mvs[0].run_id  # lastest
        run_info = self.mlwriter.client.get_run(run_id)
        uri = os.path.join(run_info.info.artifact_uri, "model")
        return uri

        # if model_tribe == 'torch':
        #     return mlflow.pytorch.load_model(uri)
        # else:
        #     raise Exception("No artificial model.")


def make_parser():
    parser = argparse.ArgumentParser(
        prog="Aleister interface",
        description="TBD",
        epilog="TBBD"
    )

    parser.add_argument(
        '-u', '--user',
        type=str,
        required=True,
        help='exec user')

    parser.add_argument(
        '-s', '--source',
        type=str,
        required=True,
        help='source ')

    parser.add_argument(
        '-sym', '--symbol',
        type=str,
        required=True,
        help='target symbol. ')

    parser.add_argument(
        '-mode', '--execute_mode',
        type=str,
        required=True,
        choices=['prepro', 'train', 'gtrain', 'rpredict', 'deploy_model'],
        help='Execution mode. ')

    # TOOO: enable to  get from api request
    parser.add_argument(
        '-cs', '--config_source',
        type=str,
        default='ini',
        choices=['db', 'ini'],
        help='Where is ini file.')

    parser.add_argument(
        "-gcm",
        "--general_config_mode",
        type=str,
        required=True,
        help="Select general config mode",
    )
    parser.add_argument(
        "-pam",
        "--private_api_mode",
        type=str,
        required=True,
        help="Select private api mode",
    )

    parser.add_argument(
        '-id', '--model_id',
        type=str,
        required=True,
        help='Model ID.')

    parser.add_argument(
        '-mn', '--model_name',
        type=str,
        required=True,
        help='Model Name.')

    parser.add_argument(
        '-train_sd', '--train_start_date',
        type=int,
        help='train start date ')

    parser.add_argument(
        '-valid_sd', '--valid_start_date',
        type=int,
        help='train start date ')

    parser.add_argument(
        '-test_sd', '--test_start_date',
        type=int,
        help='train start date ')

    parser.add_argument(
        '-train_ed', '--train_end_date',
        type=int,
        help='train end date ')

    parser.add_argument(
        '-valid_ed', '--valid_end_date',
        type=int,
        help='train end date ')

    parser.add_argument(
        '-test_ed', '--test_end_date',
        type=int,
        help='train end date ')
    return parser


def main(args=None):
    parser = make_parser()
    arg_dict = vars(parser.parse_args(args))

    # meta
    _id = arg_dict["model_id"]
    general_config_mode = arg_dict["general_config_mode"].upper()
    private_api_mode = arg_dict["private_api_mode"].upper()
    sym = arg_dict["symbol"]

    model_name = arg_dict["model_name"].upper()

    om = OperateMaster()
    om.load_meta(_id, model_name, sym, general_config_mode, private_api_mode)
    om.init_prepro()

    # mlflow
    mlflow_tags = {
        MLFLOW_USER: arg_dict["user"],
        MLFLOW_SOURCE_NAME: arg_dict["source"],
        MLFLOW_RUN_NAME: f"TRAIN_{dt.now().strftime('%y%m%d%H%M%s')}"
    }
    mlflow_client_kwargs = {
        "tracking_uri": os.environ['MLFLOW_TRACKING_URI']
    }
    om.set_mlflow_settings(mlflow_client_kwargs, mlflow_tags)

    # load conofigs into each module
    om.init_learning()

    if arg_dict["execute_mode"] == "prepro":
        om.init_dataGen()
        # period
        train_start = arg_dict["train_start_date"] if "train_start_date" in arg_dict.keys() else None
        train_end = arg_dict["train_end_date"] if "train_end_date" in arg_dict.keys() else None
        valid_start = arg_dict["valid_start_date"] if "valid_start_date" in arg_dict.keys() else None
        valid_end = arg_dict["valid_end_date"] if "valid_end_date" in arg_dict.keys() else None
        test_start = arg_dict["test_start_date"] if "test_start_date" in arg_dict.keys() else None
        test_end = arg_dict["test_end_date"] if "test_end_date" in arg_dict.keys() else None
        om.preprocessing(sym, train_start, train_end, valid_start, valid_end, test_start, test_end)
    else:
        if arg_dict["execute_mode"] == "train":
            om.train()
        elif arg_dict["execute_mode"] == "gtrain":
            om.gtrain()
        elif arg_dict["execute_mode"] == "deploy_model":
            om.deploy_best_model()
        elif arg_dict["execute_mode"] == "rpredict":
            om.init_dataGen(remote=True)
            om.realtime_predict()


if __name__ == "__main__":
    main()
