import time
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
    grid_d = LOWER_BOUND_D + (UPPER_BOUND_D - LOWER_BOUND_D) * even_grid**EXPAND_FAC  # grid of debt

    # initialize arrays for value function, consumption and payment
    v = np.zeros((N_D, N_W))
    c = np.zeros((N_D, N_W))
    v_proxy = np.zeros((N_D, N_W))

    # terminal period: consume all the wealth, repay all the debt
    ut = utility(grid_w, GAMMA)
    v[:] = ut
    c[:] = grid_w

    # collect results
    c_over_age = np.zeros((END_AGE-START_AGE+1, N_D, N_W))
    v_over_age = np.zeros((END_AGE-START_AGE+1, N_D, N_W))

    c_over_age[-1] = c
    v_over_age[-1] = v

    ###########################################################################
    #                         Dynamic Programming                             #
    ###########################################################################
    for t in range(END_AGE-START_AGE-1, -1, -1):       # t: 77 to 0 / t+22: 99 to 22
        print('############ Age: ', t+START_AGE, '#############')
        start_time = time.time()

        for i in range(N_W):
            print('wealth_grid_progress: ', i / N_W * 100)
            consmp = np.linspace(0, grid_w[i], N_C)
            u_r = utility(consmp, GAMMA)
            u_r = u_r[None].T

            savings = grid_w[i] - np.linspace(0, grid_w[i], N_C)
            savings_incr = savings * (1 + R)
            savings_incr = savings_incr[None].T

            debt = grid_d

            if t + START_AGE >= RETIRE_AGE:
                expected_value = exp_val_r(income_ret, np.exp(inc_shk_perm(RETIRE_AGE-START_AGE+1)),
                                           savings_incr, debt, grid_w, grid_d, v, weights)
            else:
                expected_value = exp_val(income_with_tran[:, t+1], np.exp(inc_shk_perm(t+1)),
                                         savings_incr, debt, grid_w, grid_d, v, weights, t+START_AGE, flag)  # using Y_t+1 !

            v_array = u_r + DELTA * prob[t] * expected_value    # v_array has size (1, N_P * N_C)
            v_array = v_array.reshape(N_P, N_C)
            v_proxy[i, j] = np.max(v_array)
            n_row, n_col = np.unravel_index(np.argmax(v_array, axis=None), v_array.shape)
            c[i, j] = consmp[n_row, n_col]

        c_over_age[t] = c
        v_over_age[t] = v_proxy

        print("--- %s seconds ---" % (time.time() - start_time))

        # change v for calculation next stage
        v = v_proxy

    return c_over_age, v_over_age

