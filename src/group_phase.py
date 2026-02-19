import numpy as np
from scipy.stats import rankdata

sout = [0, 1, 2]  # potential values the match outcome can take
points = [0, 1, 3]  # points for result, indexed according to sout
# we construct C only for group size of 4
groupsize = 4


def getGroupOutcomeInd(s: list) -> int:
    '''Computes the index of a group outcome to look up its probability
            :param s: a list with exactly six integers either 0,1,2,
            :returns r: index
            :rtype: int
            '''
    r = 0
    for k in range(0, 6):
        r = r + 3 ** k * s[k]
    return r


def setUpC():
    '''C contains the probability of team 0 in a group of size 4 to end up on rank 1 (or 2) and team 1/2/3 on rank 2 (or 1) given an outcome sequence. '''
    global C
    # C[o*3 + j] / C[3**6 * 3 + o*3 + j] = probability of team 0 ending on rank 1 (or 2) with team j on rank 2 (or 1) for group outcome o (o=s5s4s3s2s1s0)
    C = np.zeros(3 ** 6 * (groupsize - 1) * 2)
    P = np.zeros([groupsize, 6])  # point matrix, P[i,j] = points for team i by match j
    for s0 in sout:
        P[0, 0] = points[s0]
        P[1, 0] = points[2 - s0]
        for s1 in sout:
            P[1, 1] = points[s1]
            P[2, 1] = points[2 - s1]
            for s2 in sout:
                P[2, 2] = points[s2]
                P[3, 2] = points[2 - s2]
                for s3 in sout:
                    P[3, 3] = points[s3]
                    P[0, 3] = points[2 - s3]
                    for s4 in sout:
                        P[0, 4] = points[s4]
                        P[2, 4] = points[2 - s4]
                        for s5 in sout:
                            P[1, 5] = points[s5]
                            P[3, 5] = points[2 - s5]

                            # sum points for each team, gives the table
                            T = P.sum(axis=1)

                            # count how many teams have more or the same number of points
                            above = np.greater(T[1:4], [T[0], T[0], T[0]]).sum()
                            equal = np.equal(T[1:4], [T[0], T[0], T[0]]).sum()

                            # if above is larger than 1 -> focal team won't be on first or second rank,
                            # i.e., we only need to check <=1
                            if above == 0:
                                if equal == 0:  # focal team is the only one on rank 1
                                    groupranks = rankdata(-T, method='min')
                                    # specify team(s) on group rank 2
                                    below = np.concatenate(np.argwhere(groupranks == 2))
                                    for j in below:
                                        # need to deduce 1 from j because of index offset
                                        # (j takes values from 1-3 because the setup is from perspective of team 0)
                                        C[getGroupOutcomeInd([s5, s4, s3, s2, s1, s0]) * 3 + j - 1] = 1.0 / len(below)
                                else:  # focal team 0 shares 1st/2nd rank with other teams
                                    equals = np.concatenate(
                                        np.argwhere(T[1:4] == T[0]))  # teams that have the same number of points
                                    for j in equals:
                                        # focal team on rank 1 by uniform chance: 1/(equal + 1)
                                        # any other team on rank 2 by uniform chance: 1/equal
                                        C[getGroupOutcomeInd([s5, s4, s3, s2, s1, s0]) * 3 + j] = 1.0 / (
                                                equal + 1) / equal
                                        # ..or focal team on rank 2 by uniform chance: 1/(equal+1)
                                        # any other team on rank 1: 1/equal
                                        C[(3 ** 6 + getGroupOutcomeInd([s5, s4, s3, s2, s1, s0])) * 3 + j] = 1.0 / (
                                                equal + 1) / equal

                            if above == 1:
                                j = np.concatenate(np.argwhere(T[1:4] > T[0]))[0]
                                C[(3 ** 6 + getGroupOutcomeInd([s5, s4, s3, s2, s1, s0])) * 3 + j] = 1.0 / (equal + 1)


def leaveGroup(M: np.ndarray) -> np.ndarray:
    '''Computes the probabilities of each team reaching their first or second group rank.
                :param M: matrix of match outcome probabilities
                :returns pLG: probability matrix of group rankings
                :rtype: numpy array
                '''
    setUpC()
    n = M.shape[0]  # get number of teams
    if n % 4 != 0:
        raise Exception("Sorry, but for the moment it is only possible to have groups of size 4!")

    pLG = np.zeros((n, n), dtype=float)

    for h in range(0, int(n / groupsize)):
        i = 4 * h  # first team of a group
        for s0 in sout:
            for s1 in sout:
                for s2 in sout:
                    for s3 in sout:
                        for s4 in sout:
                            for s5 in sout:
                                p = M[i, i + 1, s0] * \
                                    M[i + 1, i + 2, s1] * \
                                    M[i + 2, i + 3, s2] * \
                                    M[i, i + 3, 2 - s3] * \
                                    M[i, i + 2, s4] * \
                                    M[i + 1, i + 3, s5]

                                # gives the indices to look up the probabilities in C from the respective team's perspective
                                indices = [[s5, s4, s3, s2, s1, s0],
                                           [2 - s4, s5, s0, s3, s2, s1],
                                           [2 - s5, 2 - s4, s1, s0, s3, s2],
                                           [s4, 2 - s5, s2, s1, s0, s3]]

                                for o in range(0, 3):
                                    t = i + o
                                    # s = index offset within group block
                                    for s in range(1, 4 - o):
                                        j = t + s
                                        pLG[t, j] = pLG[t, j] + p * C[getGroupOutcomeInd(indices[o]) * 3 + s - 1]
                                        pLG[j, t] = pLG[j, t] + p * C[
                                            (3 ** 6 + getGroupOutcomeInd(indices[o])) * 3 + s - 1]

    return pLG
