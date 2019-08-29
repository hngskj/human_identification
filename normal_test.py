"""
normality test for ccv data
ref) https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.normaltest.html
"""
import glob
import numpy as np
from scipy import stats

k2, p = stats.normaltest()