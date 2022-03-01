
class parameterParser:

    @staticmethod
    def sdnn(model_config, source='ini'):
        hparams = {}
        if source == 'ini':
            hparams["structure_params"] = ["hidden_dim", "layer_dim", "output_dim", "l2_drop_rate"]
            hparams["dataset_params"] = ["batch_size"]
            hparams["dataset"] = model_config.get("DATASET")
            hparams["n_epoch"] = model_config.getint("N_EPOCH")
            hparams["batch_size"] = model_config.getint("BATCHSIZE")
            hparams["hidden_dim"] = model_config.getint("HIDDEN_DIM")
            hparams["layer_dim"] = model_config.getint("LAYER_DIM")
            hparams["output_dim"] = model_config.getint("OUTPUT_DIM")
            hparams["l2_drop_rate"] = model_config.getfloat("L2_DROP_RATE")
            hparams["weight_decay"] = model_config.getfloat("WEIGHT_DECAY")
            hparams["lr"] = model_config.getfloat("LR")
            hparams["optimizer"] = model_config.get("OPTIMIZER")
            hparams["loss_fn"] = model_config.get("LOSSFN")
        # elif source == 'dict':
        #     model_config
        #     hparams["structure_params"] = ["hidden_dim","layer_dim","output_dim","l2_drop_rate"]
        #     hparams["dataset_params"] = ["batch_size"]
        #     hparams["dataset"] = model_config["DATASET"]
        #     hparams["n_epoch"] = model_config["N_EPOCH"]
        #     hparams["batch_size"] = model_config["BATCHSIZE"]
        #     hparams["hidden_dim"] = model_config["HIDDEN_DIM"]
        #     hparams["layer_dim"] = model_config["LAYER_DIM"]
        #     hparams["output_dim"] = model_config["OUTPUT_DIM"]
        #     hparams["l2_drop_rate"] = model_config["L2_DROP_RATE"]
        #     hparams["weight_decay"] = model_config["WEIGHT_DECAY"]
        #     hparams["lr"] = model_config["LR"]
        #     hparams["optimizer"] = model_config["OPTIMIZER"]
        #     hparams["loss_fn"] = model_config["LOSSFN"]
        else:
            raise Exception("Fail to recognize config source={0}".format(source))
        return hparams

    @staticmethod
    def slstm(model_config, source='ini'):
        hparams = {}
        if source == 'ini':
            hparams["structure_params"] = ["hidden_dim", "output_dim", "num_layers", "window_size"]
            hparams["dataset_params"] = ["batch_size", "window_size"]
            hparams["dataset"] = model_config.get("DATASET")
            hparams["n_epoch"] = model_config.getint("N_EPOCH")
            hparams["batch_size"] = model_config.getint("BATCHSIZE")
            hparams["hidden_dim"] = model_config.getint("HIDDEN_DIM")
            hparams["num_layers"] = model_config.getint("NUM_LAYERS")
            hparams["output_dim"] = model_config.getint("OUTPUT_DIM")
            hparams["window_size"] = model_config.getint("WINDOW_SIZE")
            hparams["l2_drop_rate"] = model_config.getfloat("L2_DROP_RATE")
            hparams["weight_decay"] = model_config.getfloat("WEIGHT_DECAY")
            hparams["lr"] = model_config.getfloat("LR")
            hparams["optimizer"] = model_config.get("OPTIMIZER")
            hparams["loss_fn"] = model_config.get("LOSSFN")
        else:
            raise Exception("Fail to recognize config source={0}".format(source))
        return hparams
        return hparams
    

    @staticmethod
    def cgm(model_config, source='ini'):
        hparams = {}
        if source == 'ini':
            hparams["structure_params"] = ["hidden_dim", "vol_input_size","price_input_size","last_dropout_rate",
                                           "seq_dropout_rate","gbl_dropout_rate","relation_num"
                                           "output_dim", "num_layers", "window_size"]
            hparams["dataset_params"] = ["batch_size", "window_size"]
            
            hparams["dataset"] = model_config.get("DATASET")
            hparams["n_epoch"] = model_config.getint("N_EPOCH")
            hparams["batch_size"] = model_config.getint("BATCHSIZE")
            hparams["window_size"] = model_config.getint("WINDOW_SIZE")
            
            hparams["hidden_dim"] = model_config.getint("HIDDEN_DIM")
            hparams["vol_input_size"] = model_config.getint("VOL_INPUT_SIZE")
            hparams["price_input_size"] = model_config.getint("PRICE_INPUT_SIZE")
            hparams["last_dropout_rate"] = model_config.getint("LAST_DROP_RATE")
            hparams["seq_dropout_rate"] = model_config.getint("SEQ_DROP_RATE")
            hparams["gbl_dropout_rate"] = model_config.getint("GBL_DROP_RATE")
            hparams["relation_num"] = model_config.getint("RELATION_NUM")
            
            hparams["num_layers"] = model_config.getint("NUM_LAYERS")
            hparams["l2_drop_rate"] = model_config.getfloat("L2_DROP_RATE")
            hparams["weight_decay"] = model_config.getfloat("WEIGHT_DECAY")
            
            hparams["output_dim"] = model_config.getint("OUTPUT_DIM")
            hparams["lr"] = model_config.getfloat("LR")
            hparams["optimizer"] = model_config.get("OPTIMIZER")
            hparams["loss_fn"] = model_config.get("LOSSFN")
        else:
            raise Exception("Fail to recognize config source={0}".format(source))
        return hparams
        return hparams