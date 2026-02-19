import numpy as np
import math
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
    '''C contains the probability of team 0 in a group to end up on rank 1 (or 2) and team 1/2/3 on rank 2 (or 1) given an outcome sequence. '''
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
    n = M.shape[0]  # get number of teams
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


def advanceProbsMix(P: np.ndarray, M: np.ndarray, groupsize: int, round_el: int) -> np.ndarray:
    '''
    Computes the probabilities of surviving the elimination round round_el.
    :param P: probability matrix from round before, i.e., entering the current elimination round
    :param M: matrix of match outcome probabilities
    :param round_el: what elimination round we are at
    :return:
    '''

    n = M.shape[0]  # number of teams
    ngroups = int(n / groupsize)  # number of groups

    pA = np.zeros((n, n), dtype=float)  # pA[i,j] = probability that i and j progress to the next stage

    nslots = int(ngroups / (2 ** (round_el - 1)))  # number of free slots through which teams can advance,
    # i.e., 8 slots will lead to 4 matches -> entering quarter final

    # number of teams that can end up in a slot, depends on how the elimination round
    nteamsperslot = int((2 ** round_el) * groupsize)

    # "neighboring" groups that need to be looked at, depends on the elimination round
    nt = range(2 ** (round_el - 1), 2 ** round_el)
    # first elimination round -> immediate neighboring group, xor 1
    # second elimination round -> two groups of neighboring subtree, xor 2,3, etc.

    for b in range(0, int(nslots / 2)):  # subtree number
        for h in range(0, nteamsperslot - 1):
            t = nteamsperslot * b + h
            gt = int(t / groupsize)

            for k in range(h + 1, nteamsperslot):
                s = nteamsperslot * b + k  # s is never equal to t
                gs = int(s / groupsize)

                pts = 0  # for the case that t takes the first slot, s on second...
                pst = 0  # ... and the other way around

                # if any true in vals, t and s are coming from the same subtree
                vals = [gt ^ m == gs for m in range(0, 2 ** (round_el - 1))]
                samesub = any(vals)

                # ogt = opponent group of t
                for ogt in nt:
                    for i in range(0, groupsize):
                        ot = (gt ^ ogt) * groupsize + i

                        if ot == s:
                            continue

                        # ogs = opponent group of s
                        for ogs in nt:
                            for j in range(0, groupsize):
                                os = (gs ^ ogs) * groupsize + j

                                if (os == t) or (ot == os):
                                    continue

                                # the outcome that t and s win their matches
                                p = (M[t, ot, 2] + 0.5 * M[t, ot, 1]) * (M[s, os, 2] + 0.5 * M[s, os, 1])

                                # in the very first mix, the slots are twisted for every second group
                                if round_el == 1 and samesub:
                                    # need to get the right order for the first mix
                                    # r = 1 if are t and s from an even group and we take [t,s]
                                    # if not, we need the swapped one [s,t]
                                    r = int((gt + 1) % 2)
                                    pts = pts + P[t * r + s * (1 - r), s * r + t * (1 - r)] * P[
                                        os * r + ot * (1 - r), ot * r + os * (1 - r)] * p
                                    pst = pst + P[s * r + t * (1 - r), t * r + s * (1 - r)] * P[
                                        os * (1 - r) + ot * r, os * r + ot * (1 - r)] * p
                                elif round_el == 1 and not samesub:
                                    # t,s come from different groups, either both via rank 1 or both via rank 2
                                    r = int(gt < gs)  # = 1 if true, then we know that both came via group rank 1
                                    pts = pts + P[t * r + (1 - r) * os, t * (1 - r) + r * os] * P[
                                        r * s + (1 - r) * ot, r * ot + (1 - r) * s] * p
                                    pst = pst + P[t * (1 - r) + r * os, t * r + (1 - r) * os] * P[
                                        (1 - r) * s + r * ot, (1 - r) * ot + r * s] * p

                                else:
                                    # need to select the correct probabilities dependent on which subtree teams are coming from
                                    if samesub:  # if t and s are coming from the same subtree -> parallel shift
                                        pts = pts + P[t, s] * P[ot, os] * p
                                        pst = pst + P[s, t] * P[os, ot] * p
                                    else:  # if t and s are coming from two different subtress
                                        pts = pts + P[t, os] * P[ot, s] * p
                                        pst = pst + P[os, t] * P[s, ot] * p

                pA[t, s] = pts
                pA[s, t] = pst

    return pA


def advanceProbsMerge(P: np.ndarray, M: np.ndarray, round_el: int) -> np.ndarray:
    '''Computes the probabilities of pairs of teams advancing to the next round when subtrees are not mixed but merged.
        :param P: probabilities of pairs of teams making it to the current round
        :param M: array of all match outcome probabilities
        :param round_el: what elimination round we are at, i.e., how often subtrees have been mixed or merged, >0
        :returns pA: probabilities of pairs of teams advancing to the next round
        :rtype: numpy array'''

    n = M.shape[0]  # number of teams

    pA = np.zeros((n, n), dtype=float)  # pA[i,j] = probability that i and j progress to the next stage

    nslots = int(n / (2 ** (round_el + 1)))  # number of free slots through which teams can advance,
    # since first two ranks of each team advance -> n/2 at most, divided by 2 for each elimination round
    # depends only on the level, not on the mixing before - different to how many teams can end up in a slot

    # first merge is never doubling teams per slot, i.e., we need to deduce 1
    # but afterwards merging is also always doubling the number of teams ending up in a slot
    nteamsperslot = int((2 ** (round_el - 1)) * groupsize)

    for b in range(0, int(nslots / 2)):  # number of pairs of subtrees, handle pairs of subtrees that get merged
        for h in range(0, nteamsperslot):
            t = nteamsperslot * (2 * b) + h

            for k in range(0, nteamsperslot):
                s = nteamsperslot * (
                        2 * b + 1) + k  # t and s are never the same -> since new slots come from merging subtrees only one team can advance from one subtree

                # they are always coming from different subtrees
                # the slot they take is fix, no transpose possible
                pCum = 0

                # ot = opponent of team t
                for ot in range(nteamsperslot * (2 * b), nteamsperslot * (2 * b) + nteamsperslot):
                    # t cannot be opponent of itself, t and ot are from same subtree
                    if t == ot:
                        continue

                    # os = opponent of s
                    for os in range(nteamsperslot * (2 * b + 1),
                                    nteamsperslot * (2 * b + 1) + nteamsperslot):

                        # s cannot be opponent of itself
                        if s == os:
                            continue

                        p = (M[t, ot, 2] + 0.5 * M[t, ot, 1]) * (M[s, os, 2] + 0.5 * M[s, os, 1])

                        pCum = pCum + (P[t, ot] + P[ot, t]) * (P[s, os] + P[os, s]) * p

                pA[t, s] = pCum

    return pA


def winFinal(pF: np.ndarray, M: np.ndarray) -> np.ndarray:
    '''Computes the probabilities of a team winning the final
                        :param pF: probabilities of pairs of teams entering the final
                        :param M: array of all match outcome probabilities
                        :returns pF: probabilities of teams winning the final
                        :rtype: numpy array
                        '''
    n = pF.shape[0]
    pW = np.zeros(n, dtype=float)  # different return type, which is why we have a separate function for the final

    for t in range(0, n):
        p = 0
        for ot in range(0, n):
            if t == ot:
                continue

            p = p + (pF[t, ot] + pF[ot, t]) * (M[t, ot, 2] + 0.5 * M[t, ot, 1])

        pW[t] = p

    return pW


def tinyCheckBlocks(l, m):
    '''
    Checks whether probabilities sum up to one.
    :param l: number of possible teams in a block
    :param m: matrix with probabilities of two teams advancing
    :return:
    '''
    problems = []
    for b in range(0, int(np.sqrt(m.size) / l)):
        s = np.sum(m[b * l:(b + 1) * l, b * l:(b + 1) * l])
        if not math.isclose(s, 1, rel_tol=1e-7):
            problems.append(b)
            print(b)

    if problems:
        raise Exception("Probabilities do not sum up to one.")
    else:
        return


def runExactProbs(M: np.array, mixuntil=math.inf) -> np.ndarray:
    '''Computes the exact probabilities given all match outcome probabilities for each stage.
        :param M: array of matrices for match probabilities on single tournament levels
        :param mixuntil: integer that defines until which elimination round (inclusive) subtrees are mixed
                (default value infinity, such that default format is a tournament tree which is mixing at every elimination round)
        :return res: returns the table with exact probabilities
                    res[i,r] = probability that team i makes it into a stage r
                    (columns start with group 2nd, group 1st, first elimination round, etc.)'''

    setUpC()
    n = M.shape[1]
    ko_rounds = int(math.log(n / groupsize * 2, 2))

    if mixuntil < 1:
        raise Exception("Format assumes that the first elimination round is always a mix of the groups.")

    P = np.zeros((ko_rounds, n, n))
    P[0, :, :] = leaveGroup(M[0])
    tinyCheckBlocks(groupsize, P[0, :, :])

    for round_el in range(1, ko_rounds):
        if round_el > mixuntil:
            res = advanceProbsMerge(P[round_el - 1, :, :], M[round_el], round_el)
        else:
            res = advanceProbsMix(P[round_el - 1, :, :], M[round_el], round_el)

        tinyCheckBlocks(2 ** round_el * groupsize, res)
        P[round_el, :, :] = res.copy()

    # last ko-round is the final
    pW = winFinal(P[ko_rounds - 1, :, :], M[ko_rounds])
    s = np.sum(pW)
    if not math.isclose(s, 1, rel_tol=1e-7):
        raise Exception("Probabilities do not sum up to one.")

    res = [[sum(P[0, 0:n, t]) for t in range(0, n)],
           [sum(P[0, t, 0:n]) for t in range(0, n)]]
    res += [[sum(P[i, t, 0:n]) + sum(P[i, 0:n, t]) for t in range(0, n)] for i in range(1, ko_rounds)]
    res.append([pW[t] for t in range(0, n)])
    res = np.transpose(np.matrix(res))

    return res


def runExactProbsCL(M: np.array, groupsize: int) -> np.ndarray:
    '''Computes the exact probabilities given all match outcome probabilities for each stage.
        :param M: array of matrices for match probabilities on single tournament levels
        :param mixuntil: integer that defines until which elimination round (inclusive) subtrees are mixed
                (default value infinity, such that default format is a tournament tree which is mixing at every elimination round)
        :return res: returns the table with exact probabilities
                    res[i,r] = probability that team i makes it into a stage r
                    (columns start with group 2nd, group 1st, first elimination round, etc.)'''

    n = M.shape[1]
    ko_rounds = int(math.log(n / groupsize * 2, 2))

    P = np.zeros((ko_rounds, n, n))
    # assign random draw to each group of two -> leaving group on rank 1 or 2 is coin flip
    for d in range(0,16):
        P[0, d*2, d*2+1] =  P[0, d * 2+1, d * 2] = 0.5
    tinyCheckBlocks(groupsize, P[0, :, :])

    for round_el in range(1, ko_rounds):
        res = advanceProbsMix(P[round_el - 1, :, :], M[round_el], groupsize,round_el)
        tinyCheckBlocks(2 ** round_el * groupsize, res)
        P[round_el, :, :] = res.copy()

    # last ko-round is the final
    pW = winFinal(P[ko_rounds - 1, :, :], M[ko_rounds])
    s = np.sum(pW)
    if not math.isclose(s, 1, rel_tol=1e-7):
        raise Exception("Probabilities do not sum up to one.")

    res = [[sum(P[0, 0:n, t]) for t in range(0, n)],
           [sum(P[0, t, 0:n]) for t in range(0, n)]]
    res += [[sum(P[i, t, 0:n]) + sum(P[i, 0:n, t]) for t in range(0, n)] for i in range(1, ko_rounds)]
    res.append([pW[t] for t in range(0, n)])
    res = np.transpose(np.matrix(res))

    return res