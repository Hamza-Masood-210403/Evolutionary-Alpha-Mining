import numpy as np
from signal_gen import tree_signal

class Backtest:
    def __init__(self, buy_tree, sell_tree, dataset, base_buy_signals, base_sell_signals, max_cap=100000, rrr=1, leverage=1):
        self.close = dataset

        self.signal = [buy - sell for buy, sell in zip(tree_signal(base_buy_signals, buy_tree), tree_signal(base_sell_signals, sell_tree))]

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

            # Checks the exit conditions based on stop loss and take profit - exclude to remove risk management
            if(self.check_exit_condition(day)):
                self.close_position(day)
            
            # If last day, close all positions
            if day == len(self.signal) - 1:
                self.close_position(day)

            # If buy signal, buy asset
            elif self.signal[day] == 1:
                self.buy_asset(day)

            # If sell signal, sell asset
            elif self.signal[day] == -1:
                self.sell_asset(day)

            # Update the portfolio value after the current trading decisions
            self.update_portfolio(day)
            self.portfolio_val_arr.append(self.portfolio_val)
            self.curr_bal_arr.append(self.curr_balance)

            # Update risk parameters
            self.update_risk_percents()
            self.update_risk_params()
    
    '''
    Executes buy decision for a given day.
    lot sized based on available capital
    current balance, holdings and avg entry price of the holdings updated accordingly
    '''
    def buy_asset(self, day):
        lot_size = ((self.curr_balance * self.percnt_cap_per_trade * self.leverage) // self.close[day])
        self.update_avg_entry_price(day, lot_size, 1)
        self.curr_holding += lot_size
        # Deduct only the actual balance used (not leveraged balance)
        self.curr_balance -= lot_size * self.close[day] / self.leverage
        self.update_position()
    
    '''
    Executes sell decision for a given day.
    lot sized based on current portfolio value (we should be able to buy back the lots we are selling at this instance)
    current balance, holdings and avg entry price of the holdings updated accordingly
    '''
    def sell_asset(self, day):
        lot_size = (max(0, (self.portfolio_val * self.percnt_cap_per_trade * self.leverage) // self.close[day]))*5
        self.update_avg_entry_price(day, lot_size, -1)
        self.curr_holding -= lot_size
        # Add only the actual balance gained (not leveraged balance)
        self.curr_balance += lot_size * self.close[day] / self.leverage
        self.update_position()

    '''
    Updates the position - 0 for no position
                           1 for long
                          -1 for short 
    '''
    def update_position(self):
        if self.curr_holding == 0:
            self.curr_position = 0
        else:
            self.curr_position = self.curr_holding / abs(self.curr_holding)
        self.trade_num += 1

    ''''
    Close the current position - sell all the bought lots or buy back the sold lots
    (Can only buy back the lost that we have capital for, others still part of short postion)
    '''
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
    
    '''
    Checks the exit conditions based on stop loss and take profit
    '''
    def check_exit_condition(self, day):
        if self.curr_position == 1 and (self.close[day] <= self.stop_val or self.close[day] >= self.take_val):
            return True
        if self.curr_position == -1 and (self.close[day] >= self.stop_val or self.close[day] <= self.take_val):
            return True
        return False
    
    '''
    Updates the average entry price of our holdings
    '''
    def update_avg_entry_price(self, day, lot_size, act_signal):
        if (self.curr_holding + lot_size * act_signal) == 0:
            self.avg_entry_price = 0
        elif lot_size > self.curr_holding and act_signal * self.curr_position < 0:
            self.avg_entry_price = self.close[day]
        else:
            self.avg_entry_price = abs((self.avg_entry_price * self.curr_holding + self.close[day] * lot_size * act_signal) / (self.curr_holding + lot_size * act_signal))
    

    '''
    Updates the risk parameters
    '''
    def update_risk_params(self):
        self.stop_val = (1 - self.curr_position * self.stop_loss_percent) * self.avg_entry_price
        self.take_val = (1 + self.curr_position * self.stop_loss_percent) * self.avg_entry_price

    '''
    Not implemented yet - use ATR to dynamically adjust stop loss
    '''
    def update_risk_percents(self):
        pass

    '''
    Number of trades - on an average it is the (number of buy actions + number of sell actions)/2 (Can also take it to be the min of two but average seems more reasonable)
    '''
    def trade_no(self):
        return (self.trade_num // 2)
    
    '''
    Computes the sharpe ratio
    '''
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
    
    '''
    Computes the sortino ratio
    '''
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
        if(downside_dev > 0):
            sortino = (annualized_return-risk_free_rate)/downside_dev
        if sortino > 0:
            return capital, sortino
        else:
            return capital, 0
    

    '''
    Computes the sterling ratio (Caution: This might be buggy)
    '''
    def sterling_ratio(self):
        capital = self.portfolio_val_arr
        daily_ret = []
        for i in range(1,len(capital)):
            daily_ret.append((capital[i]-capital[i-1])/capital[i-1])
        annualized_return = np.prod((np.array(daily_ret)+1))**(252/len(daily_ret))-1
        cum_ret = np.cumprod(np.array(daily_ret)+1)
        run_max_ret=np.maximum.accumulate(cum_ret)
        drawdown_arr = (run_max_ret-cum_ret)/run_max_ret
        max_drawdown = np.max(drawdown_arr)
        if(max_drawdown):
            sterling = (annualized_return-0.05)/max_drawdown
            return capital,sterling
        return capital,annualized_return
        
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
    
    '''
    A custom fitness function - tries to penalize the variance in the signal promoting the capture of consistent trends and stabilizing the signals
    '''
    def regularized_sharpe(self, interval_size=20):
        alpha = 2
        capital, sr = self.sharpe_ratio()
        signal = self.signal

        interval_variances = [
            np.std(signal[i:i + interval_size])
            for i in range(0, len(signal), interval_size)
        ]
        avg_flip_var = np.mean(interval_variances)

        fitness = sr - alpha * avg_flip_var
        if fitness>0:   
            return capital, fitness
        else:
            return capital, 0
    
    '''
    The fitness function being used for the training
    '''
    def fitness_function(self):
        return self.sharpe_ratio()