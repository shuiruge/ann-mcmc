
* As indicated by `/test/sgd_by_neurolab.py`, and comparing with `/test/test_mcmc_2.ipynb`, we find:
    
    1. gradient decent method often runs into a local minimum which is far from global minimum;

    2. while stochastic gradient decent method works so perfect, and there's no such local minimum problem.

This is a typical example of the statement in P69 (pgf 2) of Mitchell (Chinese version), where one advantage of employing stochastic gradient decent rather than pure gradient decent is that stochastic can further avoid local minimum!
