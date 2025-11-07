import math

import numpy as np
import pandas as pd

import exact_probs
import match_models

""" MEN'S WORLD CUP 2022 """
# take fifa ratings from october 2022
input = pd.read_csv("../data/fmwc22_fifa1022.csv", sep=",")
mwc_points = input.iloc[:, 2]
mwc_teams = input.iloc[:, 1]
n = len(mwc_teams)  # total number of teams

# match outcome model and its optional arguments
model = match_models.fifaRankingExtended
optArgs = {}
# we do not include a home field advantage
hfa_teams = []
scalar = 300
scalar_draw = 2
optArgs['scalar_draw'] = scalar_draw
optArgs['scalar'] = scalar

M = match_models.computeMatchProbs(model, mwc_points, hfa_teams, **optArgs)
# here: time-independent match outcome probabilities
M = np.array([M, M, M, M, M])
# men's tournament tree is mixing until the last round
res = exact_probs.runExactProbs(M)

# (formatted) print to console
leave_group = res[0:n, 0] + res[0:n, 1]
res = np.insert(res, [2], leave_group, axis=1)
res = np.round(res * 100, 4)

ind_lastcol = res.shape[1] - 1

# sort according to decreasing tournament win probabilities
r = np.argsort(-res[0:n, ind_lastcol])
res = res[r, 0:(ind_lastcol + 1)]
res = np.around(res, 4)

# for fix size of groups = 4, where first two advance
ko_rounds = int(math.log(n / 4 * 2, 2))

labels = ["team", "winF", "enterF", "enterSF", "enterQF"]
if ko_rounds > 3:
    labels += ["enter" + str(2 ** i) for i in range(4, ko_rounds + 1)]

labels += ["group1st", "group2nd"]
line = str(",").join(labels)
print(line)

for t in range(0, n):
    probs = (np.asarray(res[t, 0:(ind_lastcol + 1)])).tolist()[::-1]
    line = mwc_teams[r[t]] + "," + str(",").join([str(format(p, '.2f')) for p in probs])
    print(line)
