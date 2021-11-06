        
def main():
    sd = 20200101
    ed = 20200131
    sym= 'BTC'
    ans_col = "movingBinary"
    batch_size=64

    import os,sys
    import logging
    import logging.config
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

    # gen data 
    dg =  DataGen()
    dg.get_load_data_proxy()
    trades = dg.get_trades(sym=sym, sd=sd, ed=ed)
    Xy = dg.get_Xy(trades)

    # prepro
    fp = featurePreprocess(logger)
    X_train, X_val, X_test, y_train, y_val, y_test = fp.convert_dataset(Xy, ans_col, test_ratio=0.4, valid_ratio=0.5)
    scaler  = fp.get_scaler("minmax")
    X_train,x_scaler = fp.scalingX(X_train)
    X_val,_ = fp.scalingX(X_val,x_scaler)
    X_test,_ = fp.scalingX(X_test,x_scaler)

    train_loader, val_loader, test_loader, test_loader_one = fp.get_dataloader(X_train, y_train, X_val,y_val, X_test, y_test,batch_size)

    input_dim = X_train.shape[1]


    le = LearningEvaluator(name="TEST_MODEL", logger = logger)
    le.get_device()

    model_params = {
        'input_dim': input_dim,
        'hidden_dim' : 32,
        'layer_dim' : 64,
        'output_dim' : 1
        
    }
    le.get_model_instance("sdnn",model_params)

    lossfn_params = {}
    le.get_loss_fn("BCELogitLoss",{})

    optim_params = {
        'params': le.model.parameters(),
        'weight_decay': 1e-6,
        'lr' : 1e-3}
    le.get_optimizer("adam",optim_params)

    le.train(train_loader, val_loader, batch_size=batch_size, n_epochs=50, n_features=1)
    le.plot_losses()
    predictions, values = le.evaluate(
        test_loader_one,
        batch_size=1,
        n_features=input_dim
    )