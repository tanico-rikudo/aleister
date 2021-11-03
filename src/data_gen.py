import hist_data
VOLATILITY_VAR_DAYS = 30
VOLATILITY_DAYS = 1 
VOID_ALLOWANCE_RATIO = 1
class DataGen:
    def __init__():
        self.hd = None
        
    def get_load_data_proxy():
        hd = hist_data.histData()
        self.hd = hd
        
    def get_trades(sym, sd, ed):
        trades=self.hd.load(sym,'trades', sd ,ed)
        trades.timestamp =  pd.to_datetime(trades.timestamp)
        trades.set_index("timestamp",inplace=True)
        return trade

    def get_ohlcv(trades):
        ohlcv =  trades.price.resample('T', label='left', closed='left').ohlc()
        return ohlcv
    
    def get_Xy(trades):
        """[summary]

        Args:
            trades ([type]): [description]

        Returns:
            [type]: [description]
        """
        
        buy_size = trades.loc[trades.loc[:,"side"]=="BUY",["size"]].resample('T', label='left', closed='left').sum().fillna(0).rename(columns={"size":"buy_size"})
        sell_size = trades.loc[trades.loc[:,"side"]=="SELL",["size"]].resample('T', label='left', closed='left').sum().fillna(0).rename(columns={"size":"sell_size"})
        ohlcv = get_ohlcv(trades)
        mtrades = pd.concat([ohlcv,buy_size, sell_size],axis=1)
        mtrades.loc[:,["buy_size","sell_size"]] = mtrades.loc[:,["buy_size","sell_size"]].fillna(0)
        mtrades.loc[:,"close"] = mtrades.loc[:,"close"].fillna(method="ffill")
        mtrades  =  mtrades.fillna(axis=1, method='bfill')
        mtrades['size'] = mtrades.buy_size  + mtrades.sell_size

        mtrades['buy_size_ratio'] =  mtrades.buy_size /mtrades.size
        mtrades['sell_size_ratio'] =  mtrades.sell_size /mtrades.size

        # rerative value to lastest value
        ls_last_relative_target_cols =["open","high","low","close","size"]
        ls_last_relative_cols = [ "rel_"+_col for _col in  ls_last_relative_target_cols]
        convert_dict = { _before: _after for _before, _after in zip (ls_last_relative_target_cols, ls_last_relative_cols )}
        relative_values = np.log10(mtrades.loc[:,ls_last_relative_target_cols] /mtrades.last("T").loc[:,ls_last_relative_target_cols].values).rename(columns=convert_dict)

        relative_values.replace([np.inf, -np.inf],np.nan, inplace=True)
        relative_values.fillna(0,inplace=True)

        mtrades = pd.concat([mtrades, relative_values],axis=1)

        # Create answer
        mtrades['open30Mafter']  = mtrades['open'].shift(-30)
        mtrades.loc[(mtrades['open30Mafter']  - mtrades['open']) > 0, 'movingBinary']  =  1 
        mtrades.loc[(mtrades['open30Mafter']  - mtrades['open']) == 0, 'movingBinary']  =  0
        mtrades.loc[(mtrades['open30Mafter']  - mtrades['open']) < 0, 'movingBinary']  =  -1

        mtrades.loc[:,"volatility"] = mtrades.at_time("00:00")["close"].pct_change().rolling(VOLATILITY_VAR_DAYS).std()* np.sqrt(N_VOLATILITY_DAYS)
        mtrades.loc[:,"volatility"].fillna(method="ffill",inplace=True)
        mtrades.loc[:, "price_chg_allowamce"] = mtrades.open * mtrades.volatility * 0.01 * VOID_ALLOWANCE_RATIO

        mtrades.loc[np.abs(mtrades['open30Mafter']  - mtrades['open']) <= mtrades.loc[:, "price_chg_allowamce"], "movingBinary"] = 0.0
        
        Xy = mtrades[ls_last_relative_cols+["buy_size_ratio","sell_size_ratio"]+["movingBinary"]]
        return Xy
