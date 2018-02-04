
***************************
Dynamic Programming Problem
***************************


The Model
=========
We are trying to solve a stochastic dynamic programming problem with discrete and
finite horizon, and continuous state space. The Bellman equation for this problem
is given by:

.. math::

    V_t(W_t) = Max_{C_t \ge 0} [U(C_t) + \beta p_tE_t(V_{t+1}(W_{t+1}))],\ \ for\ t\lt T

where :math:`C_t` is the level of consumption at time :math:`t`, :math:`\beta \lt 1` is
the discount factor, :math:`p_t` is the probability that an agent is alive at time
:math:`t+1` conditional on being alive at time :math:`t`

:math:`W_{t+1}` is given by the agent's wealth accumulation equation:

.. math::

    W_{t+1} = R(W_t-C_t) + Y_{t+1}

:math:`Y_t`, agent's labor income at time :math:`t`, is given by:

.. math::

    log(Y_t) = f_t + v_t + \varepsilon_t

:math:`U(C_t)` is the utility function:

.. math::

    U(C_t) = {(C_t)^{1-\gamma} \over 1-\gamma}

The problem is to solve for the policy rules as a function of the state variables,
that is, :math:`C_t(W_t)`.




