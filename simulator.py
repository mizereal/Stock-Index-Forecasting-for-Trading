import numpy as np
import math, datetime, mpld3
import matplotlib.pyplot as plt
from utils import *

class Simulator:
    def __init__(
            self,
            csv_file_path:str,
            model_type:str='tdnn',
            balance:int=1000000,
            buy_limit_from_balance_portion:float=1/3,
            trade_base_fee:float=0,
            trade_fee:float=.15, # e.g. commission fee in percentage
            vat:float=7, # in percentage
            cut_loss:float=2, #in percentage
            unit_multiplier:int=100,
            trade_sensitivity:float=.02,
            ema_alpha:float=.25,
            adaptive_ema_alpha:bool=False,
            trade_freq:int=1,
            annualized_rf:float=.43): # in percentage
        self.model_type = model_type
        self.balance = balance
        self.buy_limit_from_balance_portion = buy_limit_from_balance_portion
        self.annualized_rf = annualized_rf
        self.trade_base_fee = trade_base_fee
        self.trade_fee = trade_fee/100
        self.vat = vat/100
        self.cut_loss = cut_loss/100
        self.unit_multiplier = unit_multiplier
        self.trade_sensitivity = trade_sensitivity # how delicate to graph changes in order to decide to buy/sell/hold
        self.trade_freq = trade_freq # do trade every trade_freq days
        self.adaptive_ema_alpha = adaptive_ema_alpha
        self.ema_alpha = ema_alpha
        if self.trade_sensitivity > 1 or self.trade_sensitivity < 0:
            raise Exception('trade_sensitivity must be in range of 0 to 1')
        # if self.max_buy_unit%unit_multiplier != 0 or self.max_sell_unit%unit_multiplier != 0:
            # raise Exception('max_buy_unit or max_sell_unit must be able to devide by unit_multiplier evenly.')
        self.predictor = Predictor(self.model_type, 32)
        self.volumes = get_volumes(csv_file_path)
        self.x, self.y, self.getmax, self.getmin = self.predictor.load_transform(csv_file_path)
        train_test_split_factor = .80
        self.test_x, self.test_y = self.x[math.floor(len(self.x)*train_test_split_factor):],               self.y[math.floor(len(self.y)*train_test_split_factor):] 
        self.y = self.test_y.reshape(-1, 1)[:, 0]
        for i in range(0, len(self.y)):
            self.y[i] = (self.y[i]*(self.getmax- self.getmin))+ self.getmin
        self.real_hist_price = self.y.reshape(-1, 1)[:, 0]
        self.pred = self.predictor.predict(self.test_x)
        self.pred = self.pred.reshape(-1, 1)[:, 0]
        for i in range(0, len(self.pred)):
            self.pred[i] = (self.pred[i]*(self.getmax- self.getmin))+ self.getmin
        self.adaptive_window_size = math.ceil(self.pred.std()*.3)
        if self.adaptive_ema_alpha == True:
            self.pred_ema = Smoother(self.pred.reshape(-1, 1)[:, 0], self.adaptive_window_size).transform('exponential')
        else:
            self.pred_ema = Smoother(self.pred.reshape(-1, 1)[:, 0], alpha=self.ema_alpha).transform('exponential')
        self.brought_units = {} # {"close_price":"n_units"}
        self.trade_record = [] # ('b/s/h', x, y, n_units, value) for record buy/sell/hold use for plot the graph

    def __cal_sharpe_ratio(self, return_port:list, risk_free_rate:float):
        sharpe_ratio = []
        if len(return_port) >= 252:
            for i in range(len(return_port)//252):
                curr_port = return_port[252*i:252*i+252]
                pct_change = np.diff(curr_port)/curr_port[:-1]*100
                ann_expected_return = pct_change.mean()*len(curr_port)
                s = (ann_expected_return-risk_free_rate)/(pct_change.std()*np.sqrt(len(curr_port)))
                sharpe_ratio.append(s)
            else:
                if i*252 != len(return_port):
                    curr_port = return_port[252*i+252:]
                    pct_change = np.diff(curr_port)/curr_port[:-1]*100
                    ann_expected_return = pct_change.mean()*len(curr_port)
                    s = (ann_expected_return-risk_free_rate)/(pct_change.std()*np.sqrt(len(curr_port)))
                    sharpe_ratio.append(s)
        else:
            curr_port = return_port
            pct_change = np.diff(curr_port)/curr_port[:-1]*100
            ann_expected_return = pct_change.mean()*len(curr_port)
            s = (ann_expected_return-risk_free_rate)/(pct_change.std()*np.sqrt(len(curr_port)))
            sharpe_ratio.append(s)
        return sharpe_ratio

    def __calculate_trade_value(self, trade_type:str, n_units:int, price:float):
        if trade_type not in ['s', 'b']:
            raise Exception('trade_type must be s or b')
        value = n_units*price
        commission_fee = value*self.trade_fee
        vat_fee = commission_fee*self.vat
        if trade_type == 'b':
            value += commission_fee+vat_fee+self.trade_base_fee
        elif trade_type == 's':
            value -= commission_fee+vat_fee+self.trade_base_fee
        return value

    def __buy(self, curr_price:float, trade_portion:float, day:int):
        if trade_portion > 3:
            trade_portion = 3
        max_budget = self.balance*self.buy_limit_from_balance_portion
        tot_trade_budget = max_budget*trade_portion
        tot_trade_units = self.unit_multiplier*((tot_trade_budget//curr_price)//self.unit_multiplier)
        if tot_trade_units > 0:
            if curr_price not in list(self.brought_units):
                self.brought_units[curr_price] = tot_trade_units
            else:
                self.brought_units[curr_price] += tot_trade_units
            paid_off = self.__calculate_trade_value('b', tot_trade_units, curr_price)
            self.balance -= paid_off
            self.trade_record.append(('b', day, self.real_hist_price[day], tot_trade_units, -paid_off))
        return self.balance

    def __hold(self, day:int):
        self.trade_record.append(('h', day, 0, 0, 0))
        return self.balance

    def __sell(self, prev_price:float, curr_price:float, trade_portion:float, day:int):
        if trade_portion > 3:
            trade_portion = 3
        if curr_price <= prev_price-prev_price*self.cut_loss:
            # cut loss
            return_value = self.__calculate_trade_value('s', self.brought_units[prev_price], curr_price)
            self.balance += return_value
            self.trade_record.append(('s', day, self.real_hist_price[day], self.brought_units[prev_price], return_value))
            del(self.brought_units[prev_price])
            return self.balance
        else:
            tot_trade_units = self.unit_multiplier * (self.brought_units[prev_price]*trade_portion//self.unit_multiplier)
            if self.brought_units[prev_price]-tot_trade_units == 100:
                # trade all
                tot_trade_units += 100

            if tot_trade_units > 0:
                return_value = self.__calculate_trade_value('s', tot_trade_units, curr_price)
                past_paid_off = self.__calculate_trade_value('b', tot_trade_units, prev_price)
                if return_value-past_paid_off >= 0:
                    # sell if profit or at par
                    self.balance += return_value
                    self.trade_record.append(('s', day, self.real_hist_price[day], tot_trade_units, return_value))
                else:
                    # if curr_price <= prev_price-prev_price*self.cut_loss:
                        # self.balance += return_value
                        # self.trade_record.append(('s', tot_trade_units, return_value))
                    # else:
                    return self.__hold(day)
                self.brought_units[prev_price] -= tot_trade_units
                if self.brought_units[prev_price] == 0:
                    del(self.brought_units[prev_price])
            return self.balance

    def __autotrade(self, curr_price:float, next_price:float, day:int, date_duration:int=1):
        # return [0 or 1, n_of_unit_to_buy/sell], n_of_unit_to_buy must be able to devide by unit_multiplier
        slope_direction = (next_price-curr_price)/abs(next_price-curr_price) # + is mean up, - mean down
        pct_change = abs(((next_price/curr_price)-1)*100)

        if pct_change >= self.trade_sensitivity*100:
            if slope_direction == 1:
                self.__buy(next_price, 1+(pct_change/5), day)
            else:
                for price_hist in list(self.brought_units):
                    self.__sell(price_hist, next_price, 1+(pct_change/5), day)
        else:
            # if curr_price <= prev_price-prev_price*self.cut_loss:
                # self.__sell(prev_price, curr_price, degree/90)
            # else:
            self.__hold(day)
        return self.balance

    def plot(self, plot_pred_price:bool=True, plot_buy_sell_point:bool=True, in_range:list=[0,None], save:bool=False):
        if len(in_range) != 2:
            raise Exception("in_range must be a list size of 2.")
        in_range[1] = len(self.real_hist_price) if in_range[1] == None else in_range[1]
        if in_range[0] == None or in_range[1] > len(self.real_hist_price) or in_range[0] < 0 or in_range[0] > in_range[1]:
            raise Exception("invalid values in in_range parameter.")
        b_x, b_points, b_units = [], [], []
        s_x, s_points, s_units = [], [], []
        reduce_units_b, reduce_units_s = {}, {}
        for rec in self.trade_record:
            if rec[0] == 'b':
                if rec[1] >= in_range[0] and rec[1] < in_range[1]:
                    b_x.append(rec[1])
                    b_points.append(rec[2])
                    b_units.append(rec[3])
            elif rec[0] == 's':
                if rec[1] >= in_range[0] and rec[1] < in_range[1]:
                    s_x.append(rec[1])
                    s_points.append(rec[2])
                    s_units.append(rec[3])
        for idx, z in enumerate([zip(b_x, b_points, b_units), zip(s_x, s_points, s_units)]):
            for val in z:
                if idx == 0:
                    if val[0] in list(reduce_units_b):
                        reduce_units_b[val[0]][1] += val[2]
                    else:
                        reduce_units_b[val[0]] = [val[1], val[2]]
                else:
                    if val[0] in list(reduce_units_s):
                        reduce_units_s[val[0]][1] += val[2]
                    else:
                        reduce_units_s[val[0]] = [val[1], val[2]]
        pluged = False
        mpld3.enable_notebook()
        real_days = np.arange(in_range[0], len(self.real_hist_price[in_range[0]:in_range[1]])+in_range[0])
        pred_days = np.arange(in_range[0], len(self.pred_ema[in_range[0]:in_range[1]])+in_range[0])
        fig, ax = plt.subplots()
        ax.set_title(self.model_type)
        ax.plot(real_days, self.real_hist_price[in_range[0]:in_range[1]], 'black', label='Actual')
        if plot_pred_price:
            ax.plot(real_days, (self.pred.reshape(-1, 1)[:, 0])[in_range[0]:in_range[1]], c='orange', ls='--', label='Predict')
        ax.plot(pred_days, self.pred_ema[in_range[0]:in_range[1]], 'blue', label='Predict-EMA')
        ax.set_xlabel('Days')
        ax.set_ylabel('Close Price')
        ax.legend()
        if plot_buy_sell_point:
            if len(reduce_units_b) != 0:
                b_scat = ax.scatter(list(reduce_units_b), np.array(list(reduce_units_b.values()))[:,0].astype('float'), c='g', linewidths=1)
                b_tooltip = mpld3.plugins.PointLabelTooltip(b_scat, labels=np.array(list(reduce_units_b.values()))[:,1].astype('float'))
                mpld3.plugins.connect(fig, b_tooltip)
                pluged = True
            if len(reduce_units_s) != 0:
                s_scat = ax.scatter(list(reduce_units_s), np.array(list(reduce_units_s.values()))[:,0].astype('float'), c='r', linewidths=1)
                s_tooltip = mpld3.plugins.PointLabelTooltip(s_scat, labels=np.array(list(reduce_units_s.values()))[:,1].astype('float'))
                mpld3.plugins.connect(fig, s_tooltip)
                pluged = True
            # if show_buy_sell_units:
                # pluged = True
                # for redu in [reduce_units_b, reduce_units_s]:
                    # for idx in range(len(redu)):
                        # if len(redu) != 0:
                            # ax.annotate(np.array(list(redu.values()))[idx,1].astype('float'), (list(redu)[idx], np.array(list(redu.values()))[idx,0].astype('float')))
        if save:
            if pluged:
                mpld3.save_html(fig, f'plot_{self.model_type}_{datetime.datetime.now().strftime("%Y%m%d%H%M%S")}.html')
            else:
                plt.savefig(f'plot_{self.model_type}_{datetime.datetime.now().strftime("%Y%m%d%H%M%S")}.png')

    def run(self):
        trim_days = 6 #self.adaptive_window_size if self.adaptive_ema_alpha == True else math.ceil(2/self.ema_alpha)
        for day in range(trim_days, len(self.real_hist_price)-trim_days):
            if day%self.trade_freq == 0:
                # if it start new trade period
                self.__autotrade(
                        curr_price=self.real_hist_price[day],
                        next_price=self.pred_ema[day+trim_days],
#                         next_price=self.real_hist_price[day+trim_days],
                        day=day,
                        date_duration=self.trade_freq)
        # sell all units
        for price in list(self.brought_units):
            self.balance += price*self.brought_units[price]
            del(self.brought_units[price])
        sharpe_ratio_real = self.__cal_sharpe_ratio(self.real_hist_price, self.annualized_rf)
        sharpe_ratio_pred = self.__cal_sharpe_ratio(self.pred.reshape(-1, 1)[:,0], self.annualized_rf)
        return self.balance, sharpe_ratio_real, sharpe_ratio_pred
        # return self.balance,self.real_hist_price, self.y, self.pred, self.pred_ema, self.obv, self.obv_ema, self.trade_record, self.brought_units
