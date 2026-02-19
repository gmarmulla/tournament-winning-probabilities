import math

import numpy as np
import pandas as pd

import match_models
from knockout_phase import runExactProbs
from src.group_phase import leaveGroup

""" WOMEN'S EUROS 2025 """
# take fifa ratings from december 2024
input = pd.read_csv("../data/uwe25_fifa1224.csv", sep=",")
uew_points = input.iloc[:, 2]
uew_teams = input.iloc[:, 1]
n = len(uew_teams)  # total number of teams
groupsize = 4  # size of the groups

# match outcome model and its optional arguments
model = match_models.fifaRankingExtended
optArgs = {}
# we do not include a home field advantage
hfa_teams = []
# reverse-engineered women's scale
scalar = 200
scalar_draw = 2
optArgs['scalar_draw'] = scalar_draw
optArgs['scalar'] = scalar

M = match_models.computeMatchProbs(model, uew_points, hfa_teams, **optArgs)
# here: time-independent match outcome probabilities
M = np.array([M, M, M, M])

# compute the probabilities to leave the group
# single round-robin phase
E = leaveGroup(M[0])
res = runExactProbs(M, E, groupsize)

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
    line = uew_teams[r[t]] + "," + str(",").join([str(format(p, '.2f')) for p in probs])
    print(line)
