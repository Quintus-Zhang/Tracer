# # cal_ce, to test if consumption is negative
# if any(c < 0):
#     print(w[c < 0])
#     print(age)
#     print(c[c < 0])
#
#     w_test = np.linspace(1, 4000, num=1000)
#     plt.plot(c_df.loc[:3, str(END_AGE)], c_df.loc[:3, str(age)], 'o', w_test, spline(w_test), '-')
#     plt.xlabel('Wealth')
#     plt.ylabel('Consumption')
#     # plt.show()
#     plt.savefig('negative_consumption_1.png')
#
#     spline.c[:2, 0] = 0
#     spline.c[2, 0] = (c_df.loc[1, str(age)] - c_df.loc[0, str(age)]) / (
#             c_df.loc[1, str(END_AGE)] - c_df.loc[0, str(END_AGE)])
#     plt.figure()
#     plt.plot(c_df.loc[:3, str(END_AGE)], c_df.loc[:3, str(age)], 'o', w_test, spline(w_test), '-')
#     plt.xlabel('Wealth')
#     plt.ylabel('Consumption')
#     # plt.show()
#     plt.savefig('negative_consumption_2.png')
#     sys.exit('negative consumption!')



# # income - deterministic component
# ret_income_vec = income_ret * np.ones(END_AGE - RETIRE_AGE)  # TODO: wrong here, add perm shock at last working period to retirement income
# income = np.append(income_bf_ret, ret_income_vec)
#
# # income - add income shocks
# rn_perm = np.random.normal(MU, sigma_perm_shock, (N_SIM, RETIRE_AGE - START_AGE + 1))
# rand_walk = np.cumsum(rn_perm, axis=1)
# rn_tran = np.random.normal(MU, sigma_tran_shock, (N_SIM, RETIRE_AGE - START_AGE + 1))
#
# zeros = np.zeros((N_SIM, END_AGE - RETIRE_AGE))
# perm = np.append(rand_walk, zeros, axis=1)
# tran = np.append(rn_tran, zeros, axis=1)
#
# inc_with_inc_risk = np.multiply(np.exp(perm) * np.exp(tran), income)  # inc.shape: (simu_N x 79)
