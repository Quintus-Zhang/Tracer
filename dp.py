import os
import numpy as np
import pandas as pd
from numpy.polynomial.hermite import hermgauss
from functions import utility, exp_val, exp_val_r
from constants import *


def dp_solver(income, income_ret, sigma_perm_shock, sigma_tran_shock, prob, TERM, rho, GAMMA):
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

    # power expanding grid
    even_grid = np.linspace(0, 1, N_W)
    grid_w = LOWER_BOUND_W + (UPPER_BOUND_W - LOWER_BOUND_W) * even_grid**EXPAND_FAC

    # # exponential expanding grid
    # even_grid = np.linspace(0, np.log(UPPER_BOUND_W - LOWER_BOUND_W + 1), N_W)
    # grid_w = LOWER_BOUND_W + (np.exp(even_grid) - 1)


    # initialize arrays for value function and consumption
    v = np.zeros((2, N_W))
    c = np.zeros((2, N_W))

    # terminal period: consume all the wealth
    ut = utility(grid_w, GAMMA)
    v[0, :] = ut
    c[0, :] = grid_w

    # collect results
    col_names = [str(age + START_AGE) for age in range(END_AGE-START_AGE, -1, -1)]     # 100 to 22
    c_collection = pd.DataFrame(index=pd.Int64Index(range(N_W)), columns=col_names)
    v_collection = pd.DataFrame(index=pd.Int64Index(range(N_W)), columns=col_names)
    c_collection[str(END_AGE)] = c[0, :]
    v_collection[str(END_AGE)] = v[0, :]

    ###########################################################################
    #                         Dynamic Programming                             #
    ###########################################################################
    for t in range(END_AGE-START_AGE-1, -1, -1):       # t: 77 to 0 / t+22: 99 to 22
        print('############ Age: ', t+START_AGE, '#############')
        for i in range(N_W):

            # Grid Search: for each W in the grid_w, we search for the C which maximizes the V
            consmp = np.linspace(LOWER_BOUND_C, grid_w[i], N_C)
            # even_grid = np.linspace(0, 1, N_C)
            # consmp = LOWER_BOUND_C + (grid_w[i] - LOWER_BOUND_C) * even_grid ** EXPAND_FAC

            u_r = utility(consmp, GAMMA)
            u_r = u_r[None].T

            savings = grid_w[i] - consmp
            savings_incr = savings * (1 + R)
            savings_incr = savings_incr[None].T

            if t + START_AGE >= RETIRE_AGE:
                expected_value = exp_val_r(income_ret, np.exp(inc_shk_perm(RETIRE_AGE-START_AGE+1)), savings_incr,
                                           grid_w, v[0, :], weights)
            else:
                expected_value = exp_val(income_with_tran[:, t+1], np.exp(inc_shk_perm(t+1)),
                                         savings_incr, grid_w, v[0, :], weights, t+START_AGE, TERM, rho)  # using Y_t+1 !

            try:
                v_array = u_r + DELTA * prob[t] * expected_value    # v_array has size N_C-by-1
            except:
                print(v_array)
                print(v_array.shape)
                print(u_r)
                print(prob[t])
                print(expected_value)
                print(expected_value.shape)
            v[1, i] = np.max(v_array)
            pos = np.argmax(v_array)
            c[1, i] = consmp[pos]

        # dump consumption array and value function array
        c_collection[str(t + START_AGE)] = c[1, :]
        v_collection[str(t + START_AGE)] = v[1, :]

        # change v & c for calculation next stage
        v[0, :] = v[1, :]
        c[0, :] = c[1, :]  # useless here

    return c_collection, v_collection

