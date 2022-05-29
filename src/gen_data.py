import json
import pandas as pd
import numpy as np
import sys, os
from sklearn.preprocessing import LabelBinarizer

sys.path.append(os.environ['COMMON_DIR'])
sys.path.append(os.environ['KULOKO_DIR'])
from mq import mq_handler as mq_handler
from items import hist_data
from base_process import BaseProcess

from util import daylib
from util import utils

dl = daylib.daylib()

VOLATILITY_VAR_DAYS = 30
VOLATILITY_DAYS = 1
VOID_ALLOWANCE_RATIO = 1


class DataGen(BaseProcess):
    def __init__(self, _id, symbol, general_config_mode, private_api_mode):
        super().__init__(_id)
        self.load_general_config(source="ini", path=None, mode=general_config_mode)
        # hist data Loger use KULOKO logger as default
        self.hd = hist_data.histData(symbol, general_config_mode, private_api_mode, logger=self._logger)

        # MQ
        self.mq_settings = mq_handler.load_mq_settings(self.general_config)
        self.remote = False if self.mq_settings["mqserver_host"] is None else True
        self.mq_rpc_client = {}

        self.output_data_structures = {
            "flatten_v1": self.create_flatten_simple,
            "ts_v1": self.create_ts_simple
        }
        self.input_data_structures = {
            "flatten_v1": self.dataset_flatten_simple,
            "ts_v1": self.dataset_ts_simple
        }

        self.open_mqclient("historical")
        self.open_mqclient("realtime")

    # Data fetch functions
    def open_mqclient(self, name):
        if self.remote:
            self.mq_rpc_client[name] = \
                mq_handler.init_mqclient(self.mq_settings["mqserver_host"],
                                         self.mq_settings["mqname"][name],
                                         self.mq_settings["routing_key"][name],
                                         self._logger)

            self._logger.info(f"[DONE] Connect MQ client. Name={name}")
        else:
            self._logger.info(f"[Skip] Set MQ client.")

    def close_mqclient(self, name):
        self.mq_rpc_client[name].close_mq()

    def get_hist_data(self, ch, sym, sd, ed, remote=None):
        remote = self.remote if remote is None else remote
        command = {"ch": ch, "sym": sym, "sd": sd, "ed": ed}
        if remote:
            try:
                hist_data = self.mq_rpc_client["historical"].call(command)
                self._logger.debug(f"[DONE] Fetch hist Feed from server. command={command}")
                hist_data = pd.read_json(hist_data.decode()).set_index("datetime")
            except Exception as e:
                hist_data = None
                self._logger.warning(f"[Failure] Cannot fetch hist Feed from server. command={command}, e={e}",
                                     exc_info=True)
            finally:
                self.close_mqclient("historical")
                self.open_mqclient("historical")

        else:
            try:
                hist_data = self.hd.get_data(**command)
            except Exception as e:
                hist_data = None
                self._logger.warning(f"[Failure] Cannot fetch Feed.:{e}", exc_info=True)

        return hist_data

        #  separate datas

    def fetch_realdata(self):
        try:
            command = ""

            if not self.mq_rpc_client["realtime"].is_connect():
                self.open_mqclient("realtime")

            realtime_data = self.mq_rpc_client["realtime"].call(command)
            self._logger.info(f"[DONE] Fetch Feed from server.")
        except Exception as e:
            realtime_data = None
            self._logger.warning(f"[Failure] Cannot fetch Feed from server.:{e}", exc_info=True)
        finally:
            self.close_mqclient("realtime")
            self.open_mqclient("realtime")

        #  separate datas
        json_rst = json.loads(realtime_data)
        dfs = {}
        for _key in json_rst.keys():
            try:
                if type(json_rst[_key]) == dict:
                    df = pd.DataFrame([json_rst[_key]])
                else:
                    df = pd.DataFrame(json_rst[_key])
                df["datetime"] = df["time"].apply(lambda x: dl.strYMDHMSF_to_dt(x))
                del df["time"]
                df.set_index("datetime", inplace=True)
                dfs[_key] = df
                self._logger.info(f"[DONE] Load data. Key={_key}")
            except Exception as e:
                self._logger.warning(f"[Failure] Cannot load data. Key={_key}:{e}", exc_info=True)
        return dfs

    #### Mains ###
    def get_Xy(self, mode, input_data_structure, output_data_structure, **kwargs):
        """
        Main Function
        :param mode:
        :param input_data_structure:
        :param output_data_structure:
        :param kwargs:
        :return:
        """
        assert mode in ["train", "test", "realtime"], f"Not allow mode:{mode}"
        self._logger.info(f"{kwargs}")
        datas = self.input_data_structures.get(input_data_structure.lower())(**kwargs)
        self._logger.info(datas)
        Xy = self.output_data_structures.get(output_data_structure.lower())(mode=mode, **datas)
        return Xy

    ### Auxiliary methods ###
    def get_ohlcv(self, data):
        # TODO: Ass volume
        ohlc = data.price.resample('T', label='left', closed='left').ohlc()
        ohlc.loc[:, ['open', 'high', 'low', 'close']] = ohlc.loc[:, ['open', 'high', 'low', 'close']].fillna(
            method='ffill')
        # df_v = data.volume.resample('T', label='left', closed='left').ohlc()
        # df_ohlcv = pd.concat([df_ohlc, df_v],axis=1).reset_index()
        return ohlc

    def onehot_y(self, y):
        lb = LabelBinarizer()
        y_onehot = lb.fit_transform(y)
        idxs = y.index
        y = pd.DataFrame(y_onehot, columns=lb.classes_, index=idxs)
        return y

    ### Data fetch logics ###
    def dataset_flatten_simple(self, sym=None, sd=None, ed=None, trades=None, orderbooks=None):
        trades = self.get_hist_data(ch="trade", sym=sym, sd=sd, ed=ed) if trades is None else trades
        orderbooks = self.get_hist_data(ch="orderbook", sym=sym, sd=sd, ed=ed) if orderbooks is None else orderbooks
        return {"trades": trades, "orderbooks": orderbooks}

    def dataset_ts_simple(self, sd=None, ed=None, related_syms=None, trades=None, orderbooks=None ,**kwargs):
        if related_syms is None:
            related_syms = ["BTC", "ETH"]
        self._logger.info(f"Fetching related syms:{related_syms}")
        fetch_syms = list(set(related_syms))
        datas = {_type: {_sym: None for _sym in fetch_syms} for _type in ["trades", "orderbooks"]}
        self._logger.info(trades)

        for _sym in fetch_syms:
            try:
                if trades is  None:
                    datas["trades"][_sym] = self.get_hist_data( ch="trade", sym=_sym, sd=sd,ed=ed)
                else:
                    datas["trades"][_sym] = self.get_hist_data(
                        ch="trade", sym=_sym, sd=sd,ed=ed) if _sym not in trades.keys() else trades[_sym]
                self._logger.info(f"[DONE] Fetching trade. sym:{_sym}, shape={datas['trades'][_sym].shape}")

            except Exception as e:
                self._logger.warning(f"[Failure] Fetching related sym:{_sym}:{e}")
            try:
                if trades is  None:
                    datas["orderbooks"][_sym] = self.get_hist_data(ch="orderbook", sym=_sym, sd=sd,ed=ed)
                else:
                    datas["orderbooks"][_sym] = self.get_hist_data(
                        ch="orderbook", sym=_sym, sd=sd,ed=ed)\
                        if _sym not in orderbooks.keys() else orderbooks[_sym]
                self._logger.info(f"[DONE] Fetching Orderbook. sym:{_sym}, shape={datas['orderbooks'][_sym].shape}")

            except Exception as e:
                self._logger.warning(f"[Failure] Fetching related sym:{_sym}:{e}")


        return datas

    ### Structure Logics ###
    # Note:Separate X and y ==> some data mightn't have y data.
    def create_flatten_simple(self, trades, orderbooks, mode):
        # trade based data
        buy_size = trades.loc[trades.loc[:, "side"] == "BUY", ["size"]] \
            .resample('T', label='left', closed='left') \
            .sum() \
            .fillna(0) \
            .rename(columns={"size": "buy_size"})
        sell_size = trades.loc[trades.loc[:, "side"] == "SELL", ["size"]] \
            .resample('T', label='left', closed='left') \
            .sum() \
            .fillna(0) \
            .rename(columns={"size": "sell_size"})

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
        relative_values = \
            np.log10(mtrades.loc[:, ls_last_relative_target_cols] / mtrades.last("T").loc[:,
                                                                    ls_last_relative_target_cols].values) \
                .rename(columns=convert_dict)

        relative_values.replace([np.inf, -np.inf], np.nan, inplace=True)
        relative_values.fillna(0, inplace=True)

        mtrades = pd.concat([mtrades, relative_values], axis=1)

        #  volatility
        mtrades.loc[:, "volatility"] = mtrades.at_time("00:00")["close"].pct_change().rolling(
            VOLATILITY_VAR_DAYS).std() * np.sqrt(VOLATILITY_DAYS)
        mtrades.loc[:, "volatility"].fillna(method="ffill", inplace=True)
        mtrades.loc[:, "price_chg_allowamce"] = mtrades.open * mtrades.volatility * 0.01 * VOID_ALLOWANCE_RATIO

        # Define return cols
        return_cols = ls_last_relative_cols + ["buy_size_ratio", "sell_size_ratio"]

        if mode == 'train':
            # Create answer
            mtrades['open30Mafter'] = mtrades['open'].shift(-30)  # TODO: go outside
            mtrades.loc[(mtrades['open30Mafter'] - mtrades['open']) > 0, 'movingBinary'] = 1
            mtrades.loc[(mtrades['open30Mafter'] - mtrades['open']) == 0, 'movingBinary'] = 0
            mtrades.loc[(mtrades['open30Mafter'] - mtrades['open']) < 0, 'movingBinary'] = -1
            # adjust neutral
            mtrades.loc[np.abs(mtrades['open30Mafter'] - mtrades['open']) <= mtrades.loc[:,
                                                                             "price_chg_allowamce"], "movingBinary"] = 0.0
            mtrades.dropna(subset=["movingBinary"], inplace=True)
            return_cols = return_cols
            y = self.onehot_y(mtrades.loc[:, ["movingBinary"]])
        else:
            y = None

        X = mtrades.loc[:, return_cols]

        return {"X": X, 'y': y}

    def create_ts_simple(self, trades, orderbooks, mode):
        # trade based data
        syms = trades.keys()
        self._logger.info(f"Building dataset... syms:{syms}")
        Xys = {}
        for _sym in syms:
            trade = trades[_sym]
            buy_size = trade.loc[trade.loc[:, "side"] == "BUY", ["size"]] \
                .resample('T', label='left', closed='left') \
                .sum() \
                .fillna(0) \
                .rename(columns={"size": "buy_size"})
            sell_size = trade.loc[trade.loc[:, "side"] == "SELL", ["size"]] \
                .resample('T', label='left', closed='left') \
                .sum() \
                .fillna(0) \
                .rename(columns={"size": "sell_size"})

            ohlcv = self.get_ohlcv(trade)
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
            convert_dict = {_before: _after for _before, _after in
                            zip(ls_last_relative_target_cols, ls_last_relative_cols)}
            relative_values = \
                np.log10(mtrades.loc[:, ls_last_relative_target_cols] / mtrades.last("T").loc[:,
                                                                        ls_last_relative_target_cols].values) \
                    .rename(columns=convert_dict)

            relative_values.replace([np.inf, -np.inf], np.nan, inplace=True)
            relative_values.fillna(0, inplace=True)

            mtrades = pd.concat([mtrades, relative_values], axis=1)

            #  volatility
            mtrades.loc[:, "volatility"] = mtrades.at_time("00:00")["close"].pct_change().rolling(
                VOLATILITY_VAR_DAYS).std() * np.sqrt(VOLATILITY_DAYS)
            mtrades.loc[:, "volatility"].fillna(method="ffill", inplace=True)
            mtrades.loc[:, "price_chg_allowamce"] = mtrades.open * mtrades.volatility * 0.01 * VOID_ALLOWANCE_RATIO

            # Define return cols
            return_cols = ls_last_relative_cols + ["buy_size_ratio", "sell_size_ratio"]

            if mode == 'train':
                # Create answer
                mtrades['open30Mafter'] = mtrades['open'].shift(-30)  # TODO: go outside
                mtrades.loc[(mtrades['open30Mafter'] - mtrades['open']) > 0, 'movingBinary'] = 1
                mtrades.loc[(mtrades['open30Mafter'] - mtrades['open']) == 0, 'movingBinary'] = 0
                mtrades.loc[(mtrades['open30Mafter'] - mtrades['open']) < 0, 'movingBinary'] = -1
                # adjust neutral
                mtrades.loc[np.abs(mtrades['open30Mafter'] - mtrades['open']) <= mtrades.loc[:,
                                                                                 "price_chg_allowamce"], "movingBinary"] = 0.0
                mtrades.dropna(subset=["movingBinary"], inplace=True)

                y = self.onehot_y(mtrades.loc[:, ["movingBinary"]])
            else:
                y = None
            X = mtrades.loc[:, return_cols]
            Xys[_sym] = {"X": X, 'y': y}

        return Xys
