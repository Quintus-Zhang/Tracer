from scipy.interpolate import CubicSpline
import pandas as pd
import os
import numpy as np
from functions import utility
from constants import *
from scipy.stats import bernoulli
import matplotlib.pyplot as plt
import sys


def c_func(c_df, COH, age):
    """ Given the consumption functions and cash-on-hand at certain age, calculate the corresponding consumption using
    the cubic spline interpolation

    :param c_df: DataFrame, consumption functions from age 100 to 22
    :param COH: array, cash-on-hand in every possible scenario
    :param age: scalar
    :return: array with same size as COH, consumption
    """
    COH = np.where(COH < UPPER_BOUND_COH, COH, UPPER_BOUND_COH)  # prevent extrapolation
    COH = np.where(COH > 1, COH, 1)                              # prevent extrapolation
    spline = CubicSpline(c_df[str(END_AGE)], c_df[str(age)], bc_type='natural')
    C = spline(COH)
    if any(C < 0):
        # set coeffs of 2nd and 3rd order term to 0, 1st order term to the slope between the first two points
        spline.c[:2, 0] = 0
        spline.c[2, 0] = (c_df.loc[1, str(age)] - c_df.loc[0, str(age)]) / (c_df.loc[1, str(END_AGE)] - c_df.loc[0, str(END_AGE)])
        C = spline(COH)  # redo the interpolation
    return C


def generate_consumption_process(income_bf_ret, sigma_perm_shock, sigma_tran_shock, c_func_df, *, flag='orig'):
    """ Simulate the consumption process and income process

    :param income_bf_ret   : array with size of (44,), deterministic component of income over the working period
    :param sigma_perm_shock: float64, standard deviation of the permanent shock in the log term
    :param sigma_tran_shock: float64, standard deviation of the transitory shock in the log term
    :param c_func_df       : DataFrame, consumption functions from age 100 to 22
    :param flag            : string, flag var to choose adding details of ISA or Loan or nothing(just the original model)
    :return:
           c               : array with size of N_SIM-by-79, consumption process
           inc             : array with size of N_SIM-by-79, income process
    """

    YEARS = END_AGE - START_AGE + 1

    ###########################################################################
    #                         simulate income process                         #
    #           include both income risks and unemployment risks              #
    ###########################################################################
    # add income risks - generate the random walk and normal r.v.
    # - before retirement
    rn_perm = np.random.normal(MU, sigma_perm_shock, (N_SIM, RETIRE_AGE - START_AGE + 1))
    rand_walk = np.cumsum(rn_perm, axis=1)
    rn_tran = np.random.normal(MU, sigma_tran_shock, (N_SIM, RETIRE_AGE - START_AGE + 1))
    inc_with_inc_risk = np.multiply(np.exp(rand_walk) * np.exp(rn_tran), income_bf_ret)

    # - retirement TODO: wrong here but not affect the CE, get rid of the transitory shock
    ret_income_vec = ret_frac[AltDeg] * np.tile(inc_with_inc_risk[:, -1], (END_AGE - RETIRE_AGE, 1)).T
    inc_with_inc_risk = np.append(inc_with_inc_risk, ret_income_vec, axis=1)

    # add unemployment risks - generate bernoulli random variables
    p = 1 - unempl_rate[AltDeg]
    r = bernoulli.rvs(p, size=(RETIRE_AGE - START_AGE + 1, N_SIM)).astype(float)
    r[r == 0] = unemp_frac[AltDeg]

    ones = np.ones((END_AGE - RETIRE_AGE, N_SIM))
    bern = np.append(r, ones, axis=0)

    inc = np.multiply(inc_with_inc_risk, bern.T)

    # ISA, Loan or orig
    if flag == 'rho':
        inc[:, :TERM] *= rho
    elif flag == 'ppt':
        inc[:, :TERM] -= ppt
    else:
        pass

    ################################################################################
    #                      COH_t+1 = (1 + R)*(COH_t - C_t) + Y_t+1                 #
    #                       wealth = (1 + R)*(COH_t - C_t)                         #
    ################################################################################
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
    # plt.title(f'Average Cash-on-hand and Consumption over the life cycle\n UPPER_BOUND_COH = {UPPER_BOUND_COH}')
    # plt.xlabel('Age')
    # plt.ylabel('Dollar')
    # plt.legend()
    # plt.grid()
    # plt.show()

    return c, inc


def cal_certainty_equi(prob, c):
    """ Calculate consumption CE and wealth CE

    :param prob: array with size of (78,), conditional survival probability
    :param c   : array with size of , consumption process
    :return:
           c_ce: float64, consumption CE
           total_w_ce: float64, total wealth CE
    """

    # discount factor
    YEARS = END_AGE - START_AGE + 1
    delta = np.ones((YEARS, 1)) * DELTA
    delta[0] = 1
    delta = np.cumprod(delta)

    util_c = np.apply_along_axis(utility, 1, c, GAMMA)
    simu_util = np.sum(np.multiply(util_c[:, :44], (delta * prob)[:44]), axis=1)

    if GAMMA == 1:
        c_ce = np.exp(np.mean(simu_util) / np.sum((delta * prob)[:44]))  # inverse function of CARA
    else:
        c_ce = ((1 - GAMMA) * np.mean(simu_util) / np.sum((delta * prob)[:44]))**(1 / (1-GAMMA))  # inverse function of CARA
    total_w_ce = prob[:44].sum() * c_ce   # 42.7

    return c_ce, total_w_ce
