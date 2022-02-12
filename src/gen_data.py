import hist_data
import json 
import pandas as pd
import numpy as np
import sys, os

sys.path.append(os.environ['COMMON_DIR'])
from mq.mq_handler import *

from multiprocessing import Process

VOLATILITY_VAR_DAYS = 30
VOLATILITY_DAYS = 1
VOID_ALLOWANCE_RATIO = 1


class DataGen:
    def __init__(self, symbol, general_config_mode, private_api_mode, logger):
        self.hd = hist_data.histData(symbol, general_config_mode, private_api_mode)
        self.logger = logger
        self.mq_settings = load_mq_settings(self.hd.general_config)
        self.init_mqclient()


    def init_mqclient(self):
        self.remote = False if self.mq_settings["mqserver_host"] is None else True
        self.mq_rpc_client = {}
        if self.remote:
            for name in ["historical","realtime"]:
                self.mq_rpc_client[name] = \
                    RpcClient(self.mq_settings["mqserver_host"],
                            self.mq_settings["mqname"][name], 
                            self.mq_settings["routing_key"][name],
                            self.logger)

                self.logger.info(f"[DONE] Set MQ client. Name={name}")
        else:
            self.logger.info(f"[Skip] Set MQ client.") 
            
    def get_hist_data(self, ch, sym, sd, ed, remote=None):
        remote = self.remote if remote is None else remote
        if remote:
            command = {"ch":ch, "sym":sym, "sd":sd, "ed":ed }
            try:
                hist_data = self.mq_rpc_client["historical"].call(command)
                self.logger.debug(f"[DONE] Fetch hist Feed from server. command={command}")
                hist_data = json.loads(hist_data.decode())
                hist_data = pd.json_normalize(hist_data)
            except Exception as e:
                hist_data = None
                self.logger.warning(f"[Failure] Cannot fetch hist Feed from server. command={command}, e={e}",exc_info=True)
        else:
            try:
                hist_data = self.hd.get_data(**command)
            except Exception as e:
                hist_data = None
                self.logger.warning(f"[Failure] Cannot fetch Feed.:{e}",exc_info=True)

        return hist_data

        #  separate datas

    def fetch_realdata(self):
        try:
            realtime_data = self.mq_rpc_client["realtime"].call()
        except Exception as e:
            realtime_data = None
            self.logger.warning(f"[Failure] Cannot fetch Feed from server.:{e}")

        #  separate datas

    #### Convert data ###
    def get_ohlcv(self, trades):
        ohlcv = trades.price.resample('T', label='left', closed='left').ohlc()
        return ohlcv

    def get_Xy(self, trades, orderbooks, mode='train'):
        """[summary]

        Args:
            trades ([type]): [description]

        Returns:
            [type]: [description]
        """
        # trade based data
        buy_size = trades.loc[trades.loc[:, "side"] == "BUY", ["size"]].resample('T', label='left',
                                                                                 closed='left').sum().fillna(0).rename(
            columns={"size": "buy_size"})
        sell_size = trades.loc[trades.loc[:, "side"] == "SELL", ["size"]].resample('T', label='left',
                                                                                   closed='left').sum().fillna(
            0).rename(columns={"size": "sell_size"})
        ohlcv = self.get_ohlcv(trades)
        mtrades = pd.concat([ohlcv, buy_size, sell_size], axis=1)
        mtrades.loc[:, ["buy_size", "sell_size"]] = mtrades.loc[:, ["buy_size", "sell_size"]].fillna(0)
        mtrades.loc[:, "close"] = mtrades.loc[:, "close"].fillna(method="ffill")
        mtrades = mtrades.fillna(axis=1, method='bfill')
        mtrades['size'] = mtrades.buy_size + mtrades.sell_size

        mtrades['buy_size_ratio'] = mtrades.buy_size / mtrades.size
        mtrades['sell_size_ratio'] = mtrades.sell_size / mtrades.size

        # rerative value to lastest value
        ls_last_relative_target_cols = ["open", "high", "low", "close", "size"]
        ls_last_relative_cols = ["rel_" + _col for _col in ls_last_relative_target_cols]
        convert_dict = {_before: _after for _before, _after in zip(ls_last_relative_target_cols, ls_last_relative_cols)}
        relative_values = np.log10(mtrades.loc[:, ls_last_relative_target_cols] / mtrades.last("T").loc[:,
                                                                                  ls_last_relative_target_cols].values).rename(
            columns=convert_dict)

        relative_values.replace([np.inf, -np.inf], np.nan, inplace=True)
        relative_values.fillna(0, inplace=True)

        mtrades = pd.concat([mtrades, relative_values], axis=1)

        #  volatility
        mtrades.loc[:, "volatility"] = mtrades.at_time("00:00")["close"].pct_change().rolling(
            VOLATILITY_VAR_DAYS).std() * np.sqrt(VOLATILITY_DAYS)
        mtrades.loc[:, "volatility"].fillna(method="ffill", inplace=True)
        mtrades.loc[:, "price_chg_allowamce"] = mtrades.open * mtrades.volatility * 0.01 * VOID_ALLOWANCE_RATIO

        return_cols = ls_last_relative_cols + ["buy_size_ratio", "sell_size_ratio"]

        if mode == 'train':
            # Create answer
            mtrades['open30Mafter'] = mtrades['open'].shift(-30)
            mtrades.loc[(mtrades['open30Mafter'] - mtrades['open']) > 0, 'movingBinary'] = 1
            mtrades.loc[(mtrades['open30Mafter'] - mtrades['open']) == 0, 'movingBinary'] = 0
            mtrades.loc[(mtrades['open30Mafter'] - mtrades['open']) < 0, 'movingBinary'] = -1
            # adjust neutral
            mtrades.loc[np.abs(mtrades['open30Mafter'] - mtrades['open']) <= mtrades.loc[:,
                                                                             "price_chg_allowamce"], "movingBinary"] = 0.0
            mtrades.dropna(subset=["movingBinary"], inplace=True)
            return_cols = return_cols + ["movingBinary"]

        # select cols
        Xy = mtrades[return_cols]

        return Xy
