import numpy as np

'''
Same as the backtest class in backtest.py, only difference being that it takes the signal directly as input and also adjusts the percentage of capital to be invested based on signal strength.
Can be merged into one file by adding a parameter. 
'''

class Backtest:
    def __init__(self, dataset, signal, max_cap=100000, rrr=1, leverage=1):
        self.close = dataset

        self.signal = signal

        self.curr_holding = 0
        self.curr_balance = max_cap
        self.portfolio_val = max_cap
        self.curr_position = 0
        self.portfolio_val_arr = [max_cap]
        self.curr_bal_arr = [max_cap]

        self.trade_num = 0
        self.percnt_cap_per_trade = 0.8
        self.leverage = leverage

        self.stop_loss_percent = 0.05
        self.RRR = rrr
        # self.take_profit_percent = self.stop_loss_percent * self.RRR
        self.take_profit_percent = 0.05
        self.stop_val = 0
        self.take_val = 0
        self.avg_entry_price = 0
        
        self.sim_trade()
    
    def sim_trade(self):
        for day in range(len(self.signal)):
            # if(self.check_exit_condition(day)):
            #     self.close_position(day)
            self.percnt_cap_per_trade = abs(self.signal[day])
            if day == len(self.signal) - 1:  # If last day, close all positions
                self.close_position(day)
            elif self.signal[day] > 0.2:
                self.buy_asset(day)
            elif self.signal[day] < -0.2:
                self.sell_asset(day)
            self.update_portfolio(day)
            self.portfolio_val_arr.append(self.portfolio_val)
            self.curr_bal_arr.append(self.curr_balance)
            self.update_risk_percents()
            self.update_risk_params()
            
    def buy_asset(self, day):
        lot_size = ((self.curr_balance * self.percnt_cap_per_trade * self.leverage) // self.close[day])
        self.update_avg_entry_price(day, lot_size, 1)
        self.curr_holding += lot_size
        # Deduct only the actual balance used (not leveraged balance)
        self.curr_balance -= lot_size * self.close[day] / self.leverage
        self.update_position()
        
    def sell_asset(self, day):
        lot_size = (max(0, (self.portfolio_val * self.percnt_cap_per_trade * self.leverage) // self.close[day]))*5
        self.update_avg_entry_price(day, lot_size, -1)
        self.curr_holding -= lot_size
        # Add only the actual balance gained (not leveraged balance)
        self.curr_balance += lot_size * self.close[day] / self.leverage
        self.update_position()

    def update_position(self):
        if self.curr_holding == 0:
            self.curr_position = 0
        else:
            self.curr_position = self.curr_holding / abs(self.curr_holding)
        self.trade_num += 1

    def close_position(self, day):
        if self.curr_position == -1:
            lot_size = min((self.curr_balance*self.leverage) // self.close[day], abs(self.curr_holding))
        else:
            lot_size = self.curr_holding
        self.update_avg_entry_price(day, lot_size, -self.curr_position)
        self.curr_balance += lot_size * self.curr_position * self.close[day] / self.leverage
        self.curr_holding -= lot_size * self.curr_position
        self.update_position()
    
    def update_portfolio(self, day):
        # Portfolio value includes leveraged holdings
        self.portfolio_val = self.curr_balance + self.curr_holding * self.close[day]
    
    def check_exit_condition(self, day):
        if self.curr_position == 1 and (self.close[day] <= self.stop_val or self.close[day] >= self.take_val):
            return True
        if self.curr_position == -1 and (self.close[day] >= self.stop_val or self.close[day] <= self.take_val):
            return True
        return False
    
    def update_avg_entry_price(self, day, lot_size, act_signal):
        if (self.curr_holding + lot_size * act_signal) == 0:
            self.avg_entry_price = 0
        elif lot_size > self.curr_holding and act_signal * self.curr_position < 0:
            self.avg_entry_price = self.close[day]
        else:
            self.avg_entry_price = abs((self.avg_entry_price * self.curr_holding + self.close[day] * lot_size * act_signal) / (self.curr_holding + lot_size * act_signal))
        
    def update_risk_params(self):
        self.stop_val = (1 - self.curr_position * self.stop_loss_percent) * self.avg_entry_price
        self.take_val = (1 + self.curr_position * self.stop_loss_percent) * self.avg_entry_price

    def update_risk_percents(self):
        pass

    def trade_no(self):
        return (self.trade_num // 2)
    
    def sharpe_ratio(self):
        capital = self.portfolio_val_arr
        daily_ret = []
        for i in range(1,len(capital)):
            daily_ret.append((capital[i]-capital[i-1])/capital[i-1])
        annualized_return = np.prod((np.array(daily_ret)+1))**(252/len(daily_ret))-1
        daily_std = np.std(np.array(daily_ret))
        risk_free_rate = 0.05
        sharpe = 0
        if(daily_std>0):
            sharpe = (annualized_return-risk_free_rate)/daily_std
        fitness = sharpe
        if fitness>0:
            return capital,fitness
        else:
            return capital,0
        
    def sortino_ratio(self):
        sortino = 0
        capital = self.portfolio_val_arr
        daily_ret = []
        for i in range(1,len(capital)):
            daily_ret.append((capital[i]-capital[i-1])/capital[i-1])
        annualized_return = np.prod((np.array(daily_ret)+1))**(252/len(daily_ret))-1
        daily_ret = np.array(daily_ret)
        downside_ret = daily_ret[daily_ret<0.01]
        downside_dev = np.sqrt(np.sum(downside_ret**2)/len(capital))
        risk_free_rate = 0.05
        sortino = 0
        if(downside_dev>0):
            sortino = (annualized_return-risk_free_rate)/downside_dev
        if sortino>0:
            return capital,sortino
        else:
            return capital,0
        
    def net_return(self):
        capital = self.portfolio_val_arr
        profit=0
        loss=0
        for i in range(len(capital)):
            if capital[i]>0:
                profit=profit+capital[i]
            else:
                loss=loss+capital[i]
        return capital, profit, loss
    
    def pnl_factor(self):
        capital,profit,loss=self.net_return()
        pnl_factor=profit/loss
        return capital,pnl_factor
    
    def regularized_sharpe(self, interval_size=20):
        alpha = 1
        capital, sr = self.sharpe_ratio()
        signal = self.signal

        interval_variances = [
            np.std(signal[i:i + interval_size])
            for i in range(0, len(signal), interval_size)
        ]
        avg_flip_var = np.mean(interval_variances)

        # tr = self.trade_no()

        fitness = sr*20 - alpha * avg_flip_var # + (tr/len(signal))*5
        if fitness>0:   
            return capital, fitness
        else:
            return capital, 0

    def fitness_function(self):
        return self.sharpe_ratio()