import torch
import pandas as pd
import torch.nn as nn
import numpy as np
from datetime import datetime
from dateutil.relativedelta import relativedelta as rd
import time
import math
import torch.optim as optim

from Min_Var import min_var
from Mean_Var import mean_var
from PIT import PIT_admm
from PIT_Unrolled import PITNet1

from SSPO import SSPO_fun
import matplotlib.pyplot as plt

# stocks data csv read
df = pd.read_csv('data.csv')
df = df.set_index('Date')

# s&p data csv read
df_sp = pd.read_csv('sp500.csv')
df_sp = df_sp.set_index('Date')

# stocks data csv read for partial replication
df_reduce = pd.read_csv('data.csv')
df_reduce = df_reduce.set_index('Date')


def data_process(df):
    '''
    this function gets the dataframe as input, processes it, and ouputs the cumulative change of the stocks
    that is used as input for training the model.
    '''
    df = df.pct_change()
    df = df.tail(-1)
    df = df + 1
    df = df.cumprod()
    df = df - 1
    df = df.iloc[-1, :]
    # df = df.iloc[:, :]  # Keep all rows and columns
    df = df.to_numpy()
    df = torch.from_numpy(df).type(torch.Tensor)
    return df


def data_process_m(df, m):
    '''
    This function gets the dataframe and the number of past days (m) as input,
    processes the dataframe, and outputs the cumulative change of the stocks
    for the past m days that is used as input for training the model.
    '''
    df = df.pct_change()
    df = df.tail(-1)
    df = df.tail(m)  # Consider only the last m days
    df = df + 1
    df = df.cumprod()
    df = df - 1
    df = df.iloc[:, :]  # Keep all rows and columns
    df = df.to_numpy()
    df = torch.from_numpy(df).type(torch.Tensor)
    return df


def date_slicer(df, start, duration, rebalancing_period=0):
    '''
    this function is used to slice out specific section of the data
    '''
    start = str(datetime.strptime(start, '%Y-%m-%d').date() + rd(months=rebalancing_period))
    end = str(datetime.strptime(start, '%Y-%m-%d').date() + rd(months=duration) - rd(days=1))
    return df.loc[start:end]


def mday_cw(df, m):
    '''
    This function gets the dataframe and the number of past days (m) as input,
    processes the dataframe, and outputs the cumulative change of the stocks
    for the past m days that is used as input for training the model.
    '''
    df = df.pct_change()
    df = df.tail(-m)      # Consider only the last m days
    df = df + 1
    df = df.cumprod()
    df = df - 1
    df = df.iloc[-1, :]
    df = df.to_numpy()
    df = torch.from_numpy(df).type(torch.Tensor)
    return df


def weight_mday_cw(df, w, m):
    '''
    This function gets the dataframe, the number of past days (m),
    and portfolio weights (w) as input, processes the dataframe,
    and outputs the cumulative change of the portfolio for the past m days.
    '''
    # Step 1: Calculate daily returns
    df = df.pct_change()

    # Step 2: Consider only the last m days
    df = df.tail(m)

    # Step 3: Convert to numpy for matrix operations
    df = df.to_numpy()

    # Step 4: Calculate weighted returns
    weighted_returns = torch.tensor(df, dtype=torch.float32) * w

    # Step 5: Calculate cumulative returns
    cumulative_returns = torch.cumprod(1 + weighted_returns, dim=0) - 1

    # Step 6: Get the last value of cumulative returns
    cumulative_returns = cumulative_returns[-1]
    cumulative_returns_tensor = cumulative_returns.clone().detach().float()
    return cumulative_returns_tensor


def daily_change(df):
    '''
    this function calculate the daily change of stocks included in the dataframe.
    '''
    df = df.pct_change()
    df = df.tail(-1)
    return df


def daily_return(df):
    '''
    this function calculate the daily return of stocks included in the dataframe, note that
    daily return is equal to daily change + 1
    '''
    df = df.pct_change()
    df = df.tail(-1)
    df = df + 1
    return df


def index_finder(df):
    '''
    this function is just being used for extracting the stocks symbols
    '''
    df = df.pct_change()
    df = df.tail(-1)
    df = df + 1
    df = df.cumprod()
    df = df - 1
    df = df.iloc[-1, :]
    return df


# storing stocks symbols
stocks_index = index_finder(df).index
# rebalancing period = one or three months
rbp = 1

# epochs
num_epochs = 100
sparsity = 50
'''
loss function is set to MSE and Adam optimizer is used in this model.
'''
# %%
lr = 1e-5  # learning rate
# UITNet tune
PITNet = PITNet1()
PITNet_loss_fun = torch.nn.MSELoss(reduction='mean')
PITNet_optimizer = torch.optim.Adam(PITNet.parameters(), lr=lr)


# RMSE
def RMSE(x, y, weights):
    '''
    this function calculates the root mean squere error of constructed portfollio and benchmark index
    that is used for evaluating trained models.
    '''
    temp = 0
    for i in range(len(x)):
        weighted_sum = np.dot(x.iloc[i].values, weights)
        temp += (weighted_sum - y.iloc[i]) ** 2
        #temp += (sum(x.iloc[i] * weights) - y.iloc[i]) ** 2
    return math.sqrt(temp / len(x))


# %%
def portfolio_return(df, x_test,  q_t, model, i, temp):
    x_return = date_slicer(df, '2014-08-01', 1, i)
    x_return = x_return.pct_change()
    x_return = x_return.tail(-1)
    x_return = x_return + 1
    x_return = x_return.cumprod()

    weights = np.array(model(x_test, q_t).detach())
    for j in range(len(x_return)):
        temp.append(sum(x_return.iloc[j] * weights))
    temp = np.array(temp)
    return temp


def PIT_portfolio_return(df, x_test, y_test, q_t, model, i, temp):
    x_return = date_slicer(df, '2014-08-01', 1, i)
    x_return = x_return.pct_change()
    x_return = x_return.tail(-1)
    x_return = x_return + 1
    x_return = x_return.cumprod()

    weights = np.array(model(x_test, y_test, q_t).detach())
    for j in range(len(x_return)):
        temp.append(sum(x_return.iloc[j] * weights))
    temp = np.array(temp)
    return temp


# %%
def index_return(df_sp, i, temp):
    '''
    this function outputs the cumulative return of the benchmark index test dataset of the given dataframe
    '''
    y_return = date_slicer(df_sp, '2014-08-01', 1, i)
    y_return = y_return.pct_change()
    y_return = y_return.tail(-1)
    y_return = y_return + 1
    y_return = y_return.cumprod()

    for i in range(len(y_return)):
        temp.append(sum(y_return.iloc[i]))
    temp = np.array(temp)
    return temp


# %%
def test_fun(x_test_m, q_t, i, model):
    x_change = daily_change(date_slicer(df_reduce, '2016-07-01', 6, i))
    y_change = daily_change(date_slicer(df_sp, '2016-07-01', 6, i))
    weights = np.array(model(x_test_m, q_t).detach())
    test_rmse = RMSE(x_change, y_change, weights)

    print(f'\nPITNet test Results for model {(i / rbp) + 1} (Partial replication):')
    print(f'Test RMSE: {test_rmse}')
    return test_rmse


def PIT_test_fun(x_test, y_test, q_t, k, i, model):
    '''
    this function gets test dataset, model and rebalaning period as input, then outputs the RMSE, Mean and volatility
    of the given dataset.
    '''
    x_change = daily_change(date_slicer(df_reduce, '2016-07-01', 6, i))
    y_change = daily_change(date_slicer(df_sp, '2016-07-01', 6, i))
    x_return = daily_return(date_slicer(df_reduce, '2016-07-01', 6, i))
    y_return = daily_return(date_slicer(df_sp, '2016-07-01', 6, i))
    outputs = model(x_test, y_test, q_t, k)
    weights = np.array(outputs.detach())
    #weights = np.array(model(x_test, y_test, q_t).detach())
    test_rmse = RMSE(x_change, y_change, weights)
    print(f'PIT_admm Test RMSE: {test_rmse}')
    return test_rmse


# %%
# PITNet training function
def train_PITNet(x_train, y_train, q_t, i):
    start_time_PITNet = time.time()

    print(f'\nPITNet Training & Results for model {(i / rbp) + 1} (Partial replication):')

    for epoch in range(num_epochs):
        output = PITNet(x_train, q_t)
        cumulative_change = sum(output * x_train)
        loss_PITNet = PITNet_loss_fun(cumulative_change, y_train)

        if epoch == 0 or epoch == num_epochs - 1:
            weights = output.detach()
            print(f'Epoch {epoch + 1} of {num_epochs} | MSE: {loss_PITNet.item()}')
        PITNet_optimizer.zero_grad()
        loss_PITNet.backward()
        PITNet_optimizer.step()

    training_time = format(time.time() - start_time_PITNet, '0.2f')
    print(f'Training time: {training_time}')

    return weights


def main():
    PIT_algo_test_results = []
    PIT_model_test_results = []
    PIT_model_test_plot = []
    PIT_result_plot = []
    index_test_plot = []  # storing the index test data return for plotting later on

    for i in range(int(24 / rbp)):
        x_train = data_process(date_slicer(df, '2014-08-01', 36, i * rbp))
        x_train_m = data_process_m(date_slicer(df, '2014-08-01', 36, i * rbp), 600)
        y_train = data_process(date_slicer(df_sp, '2014-08-01', 36, i * rbp))
        y_train_m = data_process_m(date_slicer(df_sp, '2014-08-01', 36, i * rbp), 600)

        x_test_m = data_process_m(date_slicer(df, '2016-07-01', 1, i * rbp), 600)
        y_test_m = data_process_m(date_slicer(df_sp, '2016-07-01', 1, i * rbp), 600)

        sspo_weight = SSPO_fun(x_train)
        minvar_weight = min_var(x_train_m)
        meanvar_weigth = mean_var(x_train_m)
        equal_weight = torch.ones_like(x_train) / 471
        '''
        构造特征矩阵,特征包括5类：
        1.前m天累积财富
        2.最小方差方法得到的权重对应的前m天积累财富
        3.sspo_L1方法得到的权重对应的前m天积累财富
        4.等权重方法对应的前m天积累财富
        5. 均值方差方法得到的权重对应的前m天积累财富
        '''
        five_day_cw = mday_cw(date_slicer(df, '2014-08-01', 36, i * rbp), 5)
        ten_day_cw = mday_cw(date_slicer(df, '2014-08-01', 36, i * rbp), 10)
        fifteen_day_cw = mday_cw(date_slicer(df, '2014-08-01', 36, i * rbp), 15)
        twenty_day_cw = mday_cw(date_slicer(df, '2014-08-01', 36, i * rbp), 20)
        twentyfive_day_cw = mday_cw(date_slicer(df, '2014-08-01', 36, i * rbp), 25)
        thirty_day_cw = mday_cw(date_slicer(df, '2014-08-01', 36, i * rbp), 30)

        five_day_sspo = weight_mday_cw(date_slicer(df, '2014-08-01', 36, i * rbp), sspo_weight, 5)
        ten_day_sspo = weight_mday_cw(date_slicer(df, '2014-08-01', 36, i * rbp), sspo_weight, 10)
        fifteen_day_sspo = weight_mday_cw(date_slicer(df, '2014-08-01', 36, i * rbp), sspo_weight, 15)
        twenty_day_sspo = weight_mday_cw(date_slicer(df, '2014-08-01', 36, i * rbp), sspo_weight, 20)
        twentyfive_day_sspo = weight_mday_cw(date_slicer(df, '2014-08-01', 36, i * rbp), sspo_weight, 25)
        thirty_day_sspo = weight_mday_cw(date_slicer(df, '2014-08-01', 36, i * rbp), sspo_weight, 30)

        five_day_minvar = weight_mday_cw(date_slicer(df, '2014-08-01', 36, i * rbp), minvar_weight, 5)
        ten_day_minvar = weight_mday_cw(date_slicer(df, '2014-08-01', 36, i * rbp), minvar_weight, 10)
        fifteen_day_minvar = weight_mday_cw(date_slicer(df, '2014-08-01', 36, i * rbp), minvar_weight, 15)
        twenty_day_minvar = weight_mday_cw(date_slicer(df, '2014-08-01', 36, i * rbp), minvar_weight, 20)
        twentyfive_day_minvar = weight_mday_cw(date_slicer(df, '2014-08-01', 36, i * rbp), minvar_weight, 25)
        thirty_day_minvar = weight_mday_cw(date_slicer(df, '2014-08-01', 36, i * rbp), minvar_weight, 30)

        five_day_equal = weight_mday_cw(date_slicer(df, '2014-08-01', 36, i * rbp), equal_weight, 5)
        ten_day_equal = weight_mday_cw(date_slicer(df, '2014-08-01', 36, i * rbp), equal_weight, 10)
        fifteen_day_equal = weight_mday_cw(date_slicer(df, '2014-08-01', 36, i * rbp), equal_weight, 15)
        twenty_day_equal = weight_mday_cw(date_slicer(df, '2014-08-01', 36, i * rbp), equal_weight, 20)
        twentyfive_day_equal = weight_mday_cw(date_slicer(df, '2014-08-01', 36, i * rbp), equal_weight, 25)
        thirty_day_equal = weight_mday_cw(date_slicer(df, '2014-08-01', 36, i * rbp), equal_weight, 30)

        five_day_meanvar = weight_mday_cw(date_slicer(df, '2014-08-01', 36, i * rbp), meanvar_weigth, 5)
        ten_day_meanvar = weight_mday_cw(date_slicer(df, '2014-08-01', 36, i * rbp), meanvar_weigth, 10)
        fifteen_day_meanvar = weight_mday_cw(date_slicer(df, '2014-08-01', 36, i * rbp), meanvar_weigth, 15)
        twenty_day_meanvar = weight_mday_cw(date_slicer(df, '2014-08-01', 36, i * rbp), meanvar_weigth, 20)
        twentyfive_meanvar = weight_mday_cw(date_slicer(df, '2014-08-01', 36, i * rbp), meanvar_weigth, 25)
        thirty_day_meanvar = weight_mday_cw(date_slicer(df, '2014-08-01', 36, i * rbp), meanvar_weigth, 30)

        # Combine the asset characteristics
        q_t = torch.stack([
            five_day_cw, ten_day_cw, fifteen_day_cw, twenty_day_cw, twentyfive_day_cw, thirty_day_cw,
            five_day_sspo, ten_day_sspo, fifteen_day_sspo, twenty_day_sspo, twentyfive_day_sspo, thirty_day_sspo,
            five_day_minvar, ten_day_minvar, fifteen_day_minvar, twenty_day_minvar, twentyfive_day_minvar, thirty_day_minvar,
            five_day_equal, ten_day_equal, fifteen_day_equal, twenty_day_equal, twentyfive_day_equal, thirty_day_equal,
            five_day_meanvar, ten_day_meanvar, fifteen_day_meanvar, twenty_day_meanvar, twentyfive_meanvar, thirty_day_meanvar
        ])
        weights = train_PITNet(x_train_m, y_train, q_t, i * rbp)

        PIT_model_test_results.append(test_fun(x_test_m, q_t, i * rbp, PITNet))
        PIT_algo_test_results.append(PIT_test_fun(x_test_m, y_test_m, q_t, 50, i * rbp, PIT_admm))

        portfolio_return(df, x_test_m, q_t, PITNet, i, PIT_model_test_plot)
        index_return(df_sp, i, index_test_plot)
        PIT_portfolio_return(df, x_test_m, y_test_m, q_t, PIT_admm, i, PIT_result_plot)
        PITNet.reset_parameters()

    # 记录所有窗口期的测试误差
    Average_test_rmse = sum(PIT_model_test_results) / 24
    PIT_admm_avgrmse = sum(PIT_algo_test_results) / 24

    print(f'Selected Model Test Results are:')
    print(f'PITNet best RMSE =', min(PIT_model_test_results))
    print('PITNet mean RMSE =:', Average_test_rmse)
    print(f'PIT_admm best RMSE =', min(PIT_algo_test_results))
    print('PIT_admm mean RMSE =:', PIT_admm_avgrmse)

    PIT_model_test_plot = np.array(PIT_model_test_plot).reshape(-1, 1)
    index_test_plot = np.array(index_test_plot).reshape(-1, 1)
    PIT_result_plot = np.array(PIT_result_plot).reshape(-1, 1)

    plot_test = pd.concat(
        [pd.DataFrame(PIT_model_test_plot), pd.DataFrame(PIT_result_plot), pd.DataFrame(index_test_plot)], axis=1, join='inner')

    plot_test.columns = ['PITNet', 'PIT-ADMM', 'S&P 500']

    # 绘制折线图
    plt.figure(figsize=(10, 6))
    plot_test['PITNet'].plot(label='PITNet', linewidth=1)
    plot_test['PIT-ADMM'].plot(label='PIT-ADMM', linewidth=1)
    plot_test['S&P 500'].plot(label='S&P 500', linewidth=1)
    plt.xlabel('Trading Days')
    plt.ylabel('Cumulative Return')

    # 显示图例
    plt.legend()

    # 显示图形
    plt.show()


if __name__ == "__main__":
    main()

