from scipy.interpolate import interp1d
import pandas as pd
import os
import numpy as np

R = 0.99
AltDeg = 4

# read the income data and the consumption data
base_path = os.path.dirname(__file__)
income_fp = os.path.join(base_path, 'data', 'labor_income_process.xls')
mortal_fp = os.path.join(base_path, 'data', 'MORTALITY-QUANCHENG.xlsx')
consmp_fp = os.path.join(base_path, 'results', 'consumption.xlsx')

income = pd.read_excel(income_fp, sheet_name='Income profiles')
income = income.loc[income['AltDeg'] == AltDeg]
income.reset_index(inplace=True, drop=True)
c_df = pd.read_excel(consmp_fp)

# conditional survival probabilities
surviv_prob = pd.read_excel(mortal_fp)
surviv_prob.set_index('AGE', inplace=True)
cond_prob = surviv_prob.loc[21:99, 'Pt_Average_P0_22']
prob = cond_prob.cumprod()

# discount factor
delta = np.ones((79, 1)) * 0.99
delta[0] = 1
delta = np.cumprod(delta)
delta = delta[None].T

# policy functions C_t(W_t)
def c_func(c_df, w, age):
    spline = interp1d(c_df['100'], c_df[str(age)], kind='cubic')
    c = spline(w)
    return c

# calculating the wealth at each age
w = np.zeros((79, 1))
w[0] = 1.1  # income.loc[0, 'e(Labor~)']  # assume the initial wealth equals to income at age 22
for t in range(78):   # t: 0 to 77
    w[t+1] = R * (w[t] - c_func(c_df, w[t], t+22)) + income.loc[t+1, 'TotYwUnempwSS']
w_adj = prob.values[None].T * delta * w


print('Total Wealth: ', sum(w_adj))

c = np.zeros((79, 1))
for t in range(79):
    c[t] = c_func(c_df, w[t], t+22)
c_adj = prob.values[None].T * delta * c

print('\n')
print('Average of consumption: ', np.mean(c_adj))

print('\n')
wc = np.concatenate((w_adj, c_adj), axis=1)

print('adjusted w and c')
print(wc)
print('wealth')
print(w)
print('consumption')
print(c)
print('prob')
print(prob.values[None].T)
print('income')
print(income['TotYwUnempwSS'])
print('delta')
print(delta)


# labor income
#                 average of c     sum of wealth
# no high school  27547.6864112    1375105.73210829
# high school     35067.4213978    1651169.71330568
# college         57223.8679602    2683333.35554625

# total income
#                 average of c     sum of wealth
# no high school  22094.2930475    1103415.7694
# high school     30079.5150547    1416768.00147
# college         53030.2787226    2487258.37053

# 100
#                 average of c     sum of wealth
# no high school  20049.0271382    1935226.66268
# high school     26184.96834      2391129.44889
# college         41982.2302443    4117647.08556

# discounted
#                 average of c     sum of wealth
# no high school   7595.1142992     808599.42599
# high school     10353.5293365    1018738.76052
# college         18193.286268     1798940.20439