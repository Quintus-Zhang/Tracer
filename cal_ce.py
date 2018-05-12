from scipy.interpolate import CubicSpline, interp1d
import pandas as pd
import os
import numpy as np
from functions import utility
from constants import *
from scipy.stats import bernoulli
import matplotlib.pyplot as plt
import sys

# policy functions: C_t(W_t)
def c_func(c_df, w, age):
    """ Given the consumption functions and wealth at certain age, return the corresponding consumption """
    w = np.where(w < UPPER_BOUND_W, w, UPPER_BOUND_W)
    w = np.where(w > 1, w, 1)

    # using cubic spline
    spline = CubicSpline(c_df[str(END_AGE)], c_df[str(age)], bc_type='natural')
    c = spline(w)
    # set coeffs of 2nd and 3rd order term to 0, 1st order term to the slope between the first two points
    if any(c < 0):
        spline.c[:2, 0] = 0
        spline.c[2, 0] = (c_df.loc[1, str(age)] - c_df.loc[0, str(age)]) / (c_df.loc[1, str(END_AGE)] - c_df.loc[0, str(END_AGE)])
    c = spline(w)

    # # using linear interpolation
    # linear_interp = interp1d(c_df[str(END_AGE)], c_df[str(age)], kind='linear')
    # c = linear_interp(w)

    return c


def generate_consumption_process(inc, c_func_df):
    """ Calculating the certainty equivalent annual consumption and life time wealth"""

    YEARS = END_AGE - START_AGE + 1

    ###########################################################################
    #               COH_t+1 = (1 + R)*(COH_t - C_t) + Y_t+1                   #
    #                wealth = (1 + R)*(COH_t - C_t)                           #
    ###########################################################################
    cash_on_hand = np.zeros((N_SIM, YEARS))
    c = np.zeros((N_SIM, YEARS))

    cash_on_hand[:, 0] = INIT_WEALTH + inc[:, 0]   # cash on hand at age 22

    # 0-77, calculate consumption from 22 to 99, cash on hand from 23 to 100
    for t in range(YEARS - 1):
        c[:, t] = c_func(c_func_df, cash_on_hand[:, t], t + START_AGE)
        cash_on_hand[:, t+1] = (1 + R) * (cash_on_hand[:, t] - c[:, t]) + inc[:, t+1]  # 1-78
    c[:, -1] = c_func(c_func_df, cash_on_hand[:, -1], END_AGE)   # consumption at age 100

    # # GRAPH - Average Cash-on-hand & consumption over lifetime
    # plt.plot(cash_on_hand.mean(axis=0), label='cash-on-hand')
    # plt.plot(c.mean(axis=0), label='consumption')
    # plt.title(f'Average Cash-on-hand and Consumption over the life cycle\n UPPER_BOUND_W = {UPPER_BOUND_W}')
    # plt.xlabel('Age')
    # plt.ylabel('Dollar')
    # plt.legend()
    # plt.grid()
    # plt.show()

    return c, inc


def cal_certainty_equi(prob, c):

    # discount factor
    YEARS = END_AGE - START_AGE + 1
    delta = np.ones((YEARS, 1)) * DELTA
    delta[0] = 1
    delta = np.cumprod(delta)

    util_c = np.apply_along_axis(utility, 1, c, GAMMA)
    simu_util = np.sum(np.multiply(util_c[:, :44], (delta * prob)[:44]), axis=1)

    if GAMMA == 1:
        c_ce = np.exp(np.mean(simu_util) / np.sum((delta * prob)[:44]))
    else:
        c_ce = ((1 - GAMMA) * np.mean(simu_util) / np.sum((delta * prob)[:44]))**(1 / (1-GAMMA))
    total_w_ce = prob[:44].sum() * c_ce   # 42.7

    return c_ce, total_w_ce
