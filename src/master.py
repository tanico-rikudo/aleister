import os,sys
import logging
import logging.config
import argparse
from datetime import datetime as dt
os.environ['BASE_DIR'] =  '/Users/macico/Dropbox/btc'
os.environ['KULOKO_DIR'] = os.path.join(os.environ['BASE_DIR'], "kuloko")
os.environ['ALEISTER_DIR'] = os.path.join(os.environ['BASE_DIR'], "aleister")
os.environ['COMMON_DIR'] = os.path.join(os.environ['BASE_DIR'], "geco_commons")
os.environ['KULOKO_INI'] = os.path.join(os.environ['COMMON_DIR'], "ini")
os.environ['ALEISTER_INI'] = os.path.join(os.environ['COMMON_DIR'], "ini")
os.environ['MONGO_DIR'] = os.path.join(os.environ['COMMON_DIR'] ,"mongodb")
os.environ['LOGDIR'] = os.path.join(os.environ['KULOKO_DIR'], "log")
sys.path.append(os.path.join(os.environ['KULOKO_DIR'],"items" ))

sys.path.append(os.path.join(os.path.dirname('__file__'),'..'))
os.environ['INIDIR'] = '/Users/macico/Dropbox/btc/kuloko/ini'
INIDIR=os.environ['INIDIR'] 
LOGDIR=os.environ['LOGDIR']

from  gen_data import DataGen
from  feature_preprocess import featurePreprocess
from learning_executor import LearningEvaluator
from model_modules import parameterParser as pp

sys.path.append(os.environ['COMMON_DIR'] )
from util.config import ConfigManager

cm = ConfigManager(os.environ['KULOKO_INI'])
logger = cm.load_log_config(os.path.join(LOGDIR,'logging.log'),log_name="ALEISTER")
        
def preprocessing(fp, sym, train_start, train_end, valid_start, valid_end, test_start, test_end):
    # gen data 
    dg =  DataGen()
    dg.get_load_data_proxy()
    fetch_start = min([ _date for _date in [train_start, train_end, valid_start, valid_end, test_start, test_end] if _date is not None])
    fetch_end =  max([ _date for _date in [train_start, train_end, valid_start, valid_end, test_start, test_end] if _date is not None])
    trades = dg.get_trades(sym=sym, sd=fetch_start, ed=fetch_end)
    fp._logger.info("[DONE] Get prepro raw data. {0}~{1}".format(fetch_start,fetch_end))
    Xy = dg.get_Xy(trades)

    # prepro
    ans_col = fp.model_config.get("ANS_COL")
    X, y = fp.feature_label_split(df=Xy, target_col=ans_col)
    y = fp.onehot_y(y)
    
    # split period
    train_datas, val_datas , test_datas = \
            fp.train_val_test_period_split([X,y], train_start, train_end, valid_start, valid_end, test_start, test_end)
         
    # assign
    X_trains = [ train_data.values for  train_data in train_datas[:-1] ]
    X_vals = [ val_data.values for  val_data in val_datas[:-1] ]
    X_tests = [ test_data.values for  tests_data in test_datas[:-1] ]
    
    y_train =train_datas[-1].value
    y_val =val_datas[-1].values
    y_test =test_datas[-1].values

    scaler  = fp.get_scaler("minmax")
    X_trains[0],x_scaler = fp.scalingX(X_trains[0])
    X_vals[0],_ = fp.scalingX(X_vals[0],x_scaler)
    X_tests[0],_ = fp.scalingX(X_tests[0],x_scaler)
    fp.save_numpy_datas(**{
        "X_train": X_trains[0], "X_val":X_vals[0], "X_test": X_tests[0],
        "y_train":y_train, "y_val":y_val, "y_test":y_test,
        "x_scaler":x_scaler
    })
    
def train(fp,le):  
    obj_keys = ["X_train", "X_val", "y_train", "y_val"]
    X_train, X_val,  y_train, y_val= fp.load_numpy_datas(obj_keys)
    X_trains = [X_train]
    X_vals = [X_val]
    
     # train set up
    input_dim = X_trains[0].shape[1]
    model_params = { _k :le.hparams[_k] for _k  in le.hparams["structure_params"]}
    model_params['input_dim'] = input_dim
    
    le.get_model_instance(le.model_name,model_params)
    
    fp.get_dataset_fn(le.hparams["dataset"])
    dataset_params = { _k: le.hparams[_k] for  _k in le.hparams["dataset_params"]}
    train_loader, val_loader, _, _ = fp.get_dataloader(le.hparams["dataset"], X_trains, y_train, X_vals,y_val, None, None, **dataset_params)

    lossfn_params = {}
    le.get_loss_fn(le.hparams["optimizer"],{})

    optim_params = {
        'params': le.model.parameters(),
        'weight_decay': le.hparams["weight_decay"],
        'lr':le.hparams["lr"]
        }
    
    le.get_optimizer(le.hparams["loss_fn"],optim_params)

    # train
    le.train(train_loader, val_loader, 
             batch_size=le.hparams["batch_size"], n_epochs=le.hparams["n_epoch"],
             n_features=1)
    le.plot_losses()
    
    # decline end
    le.terminated()
    
    # validation if data exist
    test_out_of_data(fp, le)
    
    
def test_out_of_data(fp,le):
    obj_keys = ["X_test", "y_test"]
    X_test, y_test = fp.load_numpy_datas(obj_keys)
    if X_test is None:
        print("No test data.")
        return
    X_tests = [X_test]
    
    input_dim = X_test.shape[1]
    batch_size = le.hparams["batch_size"]
    
    fp.get_dataset_fn(le.hparams["dataset"])
    dataset_params = { _k: le.hparams[_k] for  _k in le.hparams["dataset_params"]}
    _, _ ,test_loader, test_loader_one = fp.get_dataloaders(le.hparams["dataset"], X_train=None, y_train=None, X_val=None,y_val=None,  X_test=X_tests, y_test=y_test, **dataset_params)
    
    lossfn_params = {}
    le.get_loss_fn(le.hparams["optimizer"],{})
    predictions, values = le.evaluate(test_loader_one)

    
    # decline end
    le.terminated()
        
def make_parser():
    parser = argparse.ArgumentParser(
        prog="Aleister interface",
        description="TBD",
        epilog="TBBD"
    )

    parser.add_argument(
        '-u','--user',
        type=str, 
        required=True,
        help='exec user')    
    
    parser.add_argument(
        '-s','--source',
        type=str, 
        required=True,
        help='source ')    

    parser.add_argument(
        '-sym','--symbol',
        type=str, 
        required=True,
        help='target symbol. ')    

    parser.add_argument(
        '-mode','--execute_mode',
        type=str, 
        required=True,
        choices=[ 'prepro','train','valid'],
        help='Execution mode. ')
    
    parser.add_argument(
        '-cs','--config_source',
        type=str, 
        default= 'ini',
        choices=['db', 'ini'],
        help='Where is ini file.')

    parser.add_argument(
        '-cm','--config_mode',
        type=str, 
        default= 'default',
        choices=['default', 'dev'],
        help='Config mode')
    
    parser.add_argument(
        '-id','--model_id',
        type=str, 
        required=True,
        help='Model ID.')

    parser.add_argument(
        '-mn','--model_name',
        type=str, 
        required=True,
        help='Model Name.')
    
    parser.add_argument(
        '-train_sd','--train_start_date',
        type=int, 
        help='train start date ')
    
    parser.add_argument(
        '-valid_sd','--valid_start_date',
        type=int, 
        help='train start date ')
        
    parser.add_argument(
        '-test_sd','--test_start_date',
        type=int, 
        help='train start date ')
    
    parser.add_argument(
        '-train_ed','--train_end_date',
        type=int, 
        help='train end date ')
    
    parser.add_argument(
        '-valid_ed','--valid_end_date',
        type=int, 
        help='train end date ')
        
    parser.add_argument(
        '-test_ed','--test_end_date',
        type=int, 
        help='train end date ')
    return parser
    
def main(args):
    parser = make_parser()
    arg_dict = vars(parser.parse_args(args))

    #meta
    _id = arg_dict["model_id"]
    config_mode = arg_dict["config_mode"].upper()
    model_name = arg_dict["model_name"].upper()

    # period
    train_start = arg_dict["train_start_date"] if "train_start_date" in arg_dict.keys() else None
    train_end = arg_dict["train_end_date"] if "train_end_date" in arg_dict.keys() else None
    valid_start = arg_dict["valid_start_date"] if "valid_start_date" in arg_dict.keys() else None
    valid_end = arg_dict["valid_end_date"]if "valid_end_date" in arg_dict.keys() else None
    test_start = arg_dict["test_start_date"] if "test_start_date" in arg_dict.keys() else None
    test_end = arg_dict["test_end_date"] if "test_end_date" in arg_dict.keys() else None

    # sym
    sym = arg_dict["symbol"] 

    # load all parent modules
    fp = featurePreprocess(_id)
    fp.load_general_config(source="ini", path=None,mode=config_mode)
    fp.load_model_config(source="ini", path=None,model_name=model_name)

        
    if arg_dict["execute_mode"] == "prepro":
        preprocessing(fp, sym, train_start, train_end, valid_start, valid_end, test_start, test_end)
    elif arg_dict["execute_mode"] == "train":
        mlflow_tags = {
            "user":arg_dict["user"],
            "source":arg_dict["source"],
            "run_name": f"TRAIN_{dt.now().strftime('%y%m%d%H%M%s')}"
        }
        le = LearningEvaluator(_id, mlflow_tags)
        le.get_device()
        le.load_general_config(source="ini", path=None,mode=config_mode)
        le.load_model_config(source="ini", path=None, model_name=model_name)
        le.load_model_hparameters(model_name)
        train(fp, le)
    # elif arg_dict["execute_mode"] == "valid":
    #     mlflow_tags = {
    #         "user":arg_dict["user"],
    #         "source":arg_dict["source"],
    #         "run_name": f"VALID_{dt.now().strftime('%y%m%d%H%M%s')}"
    #     }
    #     le = LearningEvaluator(_id, mlflow_tags)
    #     le.get_device()
    #     le.load_general_config(source="ini", path=None,mode=config_mode)
    #     le.load_model_config(source="ini", path=None, model_name=model_name)
    #     le.load_model_hparameters(model_name)
    #     valid(fp, le)
    else:
        pass
if __name__ == "__main__":
    main(args)

        





    
