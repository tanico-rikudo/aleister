import os,sys
import logging
import logging.config
import argparse
os.environ['BASE_DIR'] =  '/Users/macico/Dropbox/btc'
os.environ['KULOKO_DIR'] = os.path.join(os.environ['BASE_DIR'], "kuloko")
os.environ['COMMON_DIR'] = os.path.join(os.environ['BASE_DIR'], "geco_commons")
os.environ['MONGO_DIR'] = os.path.join(os.environ['COMMON_DIR'] ,"mongodb")
os.environ['LOGDIR'] = os.path.join(os.environ['KULOKO_DIR'], "log")
sys.path.append(os.path.join(os.environ['KULOKO_DIR'],"items" ))

sys.path.append(os.path.join(os.path.dirname('__file__'),'..'))
os.environ['INIDIR'] = '/Users/macico/Dropbox/btc/kuloko/ini'
INIDIR=os.environ['INIDIR'] 
LOGDIR=os.environ['LOGDIR']

logging.config.fileConfig(os.path.join(INIDIR,'logconfig.ini'),defaults={'logfilename': os.path.join(LOGDIR,'logging.log')})
logger = logging.getLogger("ALEISTER")

from  gen_data import DataGen
from  feature_preprocess import featurePreprocess
from learning_executor import LearningEvaluator
from model_modules import parameterParser as pp

def preprocessing():
    # gen data 
    dg =  DataGen()
    dg.get_load_data_proxy()
    trades = dg.get_trades(sym=sym, sd=sd, ed=ed)
    Xy = dg.get_Xy(trades)

    # prepro
    fp = featurePreprocess(logger)
    fp.load_general_config()
    X_train, X_val, X_test, y_train, y_val, y_test = fp.convert_dataset(Xy, ans_col, test_ratio=0.4, valid_ratio=0.5)
    scaler  = fp.get_scaler("minmax")
    X_train,x_scaler = fp.scalingX(X_train)
    X_val,_ = fp.scalingX(X_val,x_scaler)
    X_test,_ = fp.scalingX(X_test,x_scaler)

    train_loader, val_loader, test_loader, test_loader_one = fp.get_dataloader(X_train, y_train, X_val,y_val, X_test, y_test,batch_size)
    
def train():
    fp = featurePreprocess(logger)
    fp.load_general_config()
    X_train, X_val, X_test, y_train, y_val, y_test = fp.load_numpy_datas(*["X_train", "X_val", "X_test", "y_train", "y_val", "y_test" ])

    # train 
    input_dim = X_train.shape[1]
    le = LearningEvaluator(name="TEST_MODEL", logger = logger)
    le.get_device()
    le.load_general_config()
    
    model_params = {
        'input_dim': input_dim,
        'hidden_dim' : le.hparams["hidden_dim"],
        'layer_dim' : le.hparams["layer_dim"],
        'output_dim' : le.hparams["out_dim"],
        'l2_drop_rate':le.hparams["l2_drop_rate"]
    }
    le.get_model_instance(model_type,model_params)

    lossfn_params = {}
    le.get_loss_fn(hparams["optimizer"],{})

    optim_params = {
        'params': le.model.parameters(),
        'weight_decay': le.hparams["weight_decay"],
        'lr':le.hparams["lr"]
        }
    
    le.get_optimizer(hparams["loss_fn"],optim_params)

    le.train(train_loader, val_loader, 
             batch_size=e.hparams["batch_size"], n_epochs=le.hparams["n_epoch"],
             n_features=1)
    le.plot_losses()
    if len(y_test) > 0:
        predictions, values = le.evaluate(
            test_loader_one,
            batch_size=1,
            n_features=input_dim
        )
    
def main():
    #todo: define outside
    sd = 20200101
    ed = 20200131
    sym= 'BTC'
    ans_col = "movingBinary"
    model_type= "sdnn"
    
def make_parser():
    parser = argparse.ArgumentParser(
        prog="Aleister interface",
        description="TBD",
        epilog="TBBD"
    )
    
    parser.add_argument(
        '-mode','--execute_mode',
        type=str, 
        required=True,
        choices=['train', 'prepro'],
        help='Execution mode. ')
    
    parser.add_argument(
        '-cs','--config_source',
        type=str, 
        default= 'ini',
        choices=['db', 'ini'],
        help='Where is ini file.')
    
    parser.add_argument(
        '-id','--model_id',
        type=str, 
        required=True,
        help='Model ID.')
    
    parser.add_argument(
        '-train_sd','--train_start_date',
        type=int, 
        help='train start date ')
    
    parser.add_argument(
        'valid_sd','--valid_start_date',
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
        'valid_ed','--valid_end_date',
        type=int, 
        help='train end date ')
        
    parser.add_argument(
        '-test_ed','--test_end_date',
        type=int, 
        help='train end date ')

    
if __name__ == "__main__":
    parser = make_parser()
    arg_dict = vars(parser.parse_args(args))
    if arg_dict["execute_mode"] == "train":
        train()
    elif arg_dict["execute_mode"] == "test":
        preprocessing()
    else:
        pass
        





    
