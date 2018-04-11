import os
import numpy as np
import pandas as pd
from numpy.polynomial.hermite import hermgauss
from functions import utility, exp_val, exp_val_r
from constants import *


def dp_solver(income, income_ret, sigma_perm_shock, sigma_tran_shock, prob, flag):
    ###########################################################################
    #                                Setup                                    #
    ###########################################################################
    # Gauss-Hermite Quadrature
    [sample_points, weights] = hermgauss(3)
    sample_points = sample_points[None].T
    weights = weights[None].T

    # shocks
    inc_shk_perm = lambda t: np.sqrt(2) * np.sqrt(t) * sample_points * sigma_perm_shock
    inc_shk_tran = np.sqrt(2) * sample_points * sigma_tran_shock
    income_with_tran = np.exp(inc_shk_tran) * income

    # construct grids
    even_grid = np.linspace(0, 1, N_W)
    grid_w = LOWER_BOUND_W + (UPPER_BOUND_W - LOWER_BOUND_W) * even_grid**EXPAND_FAC
    even_grid = np.linspace(0, 1, N_D)
    grid_d = LOWER_BOUND_D + (UPPER_BOUND_D - LOWER_BOUND_D) * even_grid**EXPAND_FAC

    # initialize arrays for value function, consumption and payment
    v = np.zeros((N_D, N_W))
    c = np.zeros((N_D, N_W))
    v_proxy = np.zeros((N_D, N_W))
    c_proxy = np.zeros((N_D, N_W))
    repayment = np.zeros((N_D, N_W))

    # terminal period: consume all the wealth, repay all the debt
    ut = utility(grid_w, GAMMA)
    v[:] = ut
    c[:] = grid_w
    repayment[:] = grid_d[None].T

    # # collect results
    # col_names = [str(age + START_AGE) for age in range(END_AGE-START_AGE, -1, -1)]     # 100 to 22
    # c_collection = pd.DataFrame(index=pd.Int64Index(range(N_W)), columns=col_names)
    # v_collection = pd.DataFrame(index=pd.Int64Index(range(N_W)), columns=col_names)
    # c_collection[str(END_AGE)] = c[0, :]
    # v_collection[str(END_AGE)] = v[0, :]

    ###########################################################################
    #                         Dynamic Programming                             #
    ###########################################################################
    for t in range(END_AGE-START_AGE-1, -1, -1):       # t: 77 to 0 / t+22: 99 to 22
        print('############ Age: ', t+START_AGE, '#############')

        for i in range(N_W):
            for j in range(N_D):
                repymt = np.linspace(0, min(grid_d[j], grid_w[i]), N_P)  # TODO:
                consmp = np.zeros((N_P, N_C))
                for k in range(len(repymt)):
                    consmp[k, :] = np.linspace(0, grid_w[i] - repymt[k], N_C)

                u_r = np.apply_along_axis(utility, 1, consmp, GAMMA)
                u_r = u_r.flatten()

                savings = consmp[:, -1][None].T - consmp  #
                savings_incr = savings * (1 + R)
                savings_incr = savings_incr.flatten()

                # debt evolution equation
                debt = grid_d[j] * (1 + I) - repymt
                debt[debt > grid_d[-1]] = grid_d[-1]
                debt[debt < grid_d[0]] = grid_d[0]
                debt = np.repeat(debt, N_C)

                if t + START_AGE >= RETIRE_AGE:
                    expected_value = exp_val_r(income_ret, np.exp(inc_shk_perm(RETIRE_AGE-START_AGE+1)),
                                               savings_incr, debt, grid_w, grid_d, v, weights)
                else:
                    expected_value = exp_val(income_with_tran[:, t+1], np.exp(inc_shk_perm(t+1)),
                                             savings_incr, debt, grid_w, grid_d, v, weights, t+START_AGE, flag)  # using Y_t+1 !

                v_array = u_r + DELTA * prob[t] * expected_value    # v_array has size (1, N_P * N_C)
                v_array = v_array.reshape(N_P, N_C)
                v_proxy[j, i] = np.max(v_array)
                pos = np.unravel_index(np.argmax(v_array, axis=None), v_array.shape)
                c_proxy[j, i] = consmp[pos[1]]
                repayment[j, i] = repymt[pos[0]]

        # # dump consumption array and value function array
        # c_collection[str(t + START_AGE)] = c[1, :]
        # v_collection[str(t + START_AGE)] = v[1, :]

        # change v & c for calculation next stage
        v = v_proxy
        c = c_proxy  # useless here

    # return c_collection, v_collection

