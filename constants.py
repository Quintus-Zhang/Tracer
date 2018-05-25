###########################################################################
#                            Set Constants                                #
###########################################################################
START_AGE = 22              # age starts to work
END_AGE = 100               # age assumed to die
RETIRE_AGE = 65             # retirement age
N_COH = 501                 # the number of points in the grid of cash-on-hand
LOWER_BOUND_COH = 1         # lower bound of cash-on-hand
UPPER_BOUND_COH = 15000000  # upper bound of cash-on-hand
EXPAND_FAC = 3              # expanding factor, use to generate expanding spacing grid
N_C = 1501                  # the number of points in the grid of consumption
LOWER_BOUND_C = 0           # lower bound of consumption
GAMMA = 2                   # risk preference parameter
R = 0.02                    # risk-free rate
DELTA = 0.99                # discount factor
MU = 0                      # expectation of income shocks
N_SIM = 100000              # number of draws
INIT_WEALTH = 0             # initial wealth that individual assumed to have when starts to work

AltDeg = 4                  # education level identifier
flag = 'orig'               # flag var, 'orig' or 'rho' or 'ppt'
run_dp = True               # Bool, run the dp_solver or not

#
education_level = {
    1: 'No High School',
    2: 'High School Graduates',
    4: 'College Graduates'
}

# replacement rate of retirement earnings (lambda)
ret_frac = {
    1: 0.6005,
    2: 0.5788,
    4: 0.4516,
}

# replacement rate of unemployment earnings (theta)
unemp_frac = {
    1: 0.7891,
    2: 0.7017,
    4: 0.5260 # 1-10**(-5),  # 0.5260,
}

# probability of suffering an unemployed spell (pi)
unempl_rate = {
    1: 0.2024,
    2: 0.1438,
    4: 0.0738 # 10**(-5),  # 0.0738,
}

# rho
rho = 0.900796641891997  # rho = 0.931001221
TERM = 10

# ppt
ppt = 3483.25  # 10031.76 #