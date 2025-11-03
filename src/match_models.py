import numpy as np


def fifaRankingExtended(rt: float, ru: float, s: int, **kwargs) -> float:
    '''Implements the match outcome model embedded within the fifa ranking procedure, extended to include draws
            :param rt: rating of team t (home team)
            :param ru: rating of team u (away team)
            :param s: match outcome (0,1,2)
            :param kwargs: optional model parameters - scalar (scaling of points), scalar_draw (weight of draws)
            :returns p: probability of match outcome s out of perspective of t (0 = t loses, 1 = draw, 2= t wins)
            :rtype: float
            '''

    scalar = kwargs.get('scalar', 1.0)  # default value if no scale is specified
    scalar_draw = kwargs.get('scalar_draw', 1.0)
    scalar_hfa = kwargs.get('scalar_hfa', 1.0)
    strength_u = 1.0 / (1 + 10 ** ((rt - ru) / scalar))
    strength_t = 1.0 / (1 + 10 ** ((ru - rt) / scalar))
    if s == 0:
        return strength_u / (scalar_hfa * strength_t + scalar_draw * np.math.sqrt(strength_t * strength_u) + strength_u)
    elif s == 1:
        return scalar_draw * np.math.sqrt(strength_t * strength_u) / (
                scalar_hfa * strength_t + scalar_draw * np.math.sqrt(strength_t * strength_u) + strength_u)
    else:
        return scalar_hfa * strength_t / (
                scalar_hfa * strength_t + scalar_draw * np.math.sqrt(strength_t * strength_u) + strength_u)


def computeMatchProbs(transform, ratings: np.array, hfa_teams: list, **kwargs) -> np.ndarray:
    '''Computes all match outcome probabilities for a set of teams based on a model.
            :param transform: a function that transforms two ratings r,t and a match outcome s into a probability
                            (from the perspective of the first team)
            :param ratings: vector of team ratings
            :param hfa_teams: list of teams that have a home advantage (e.g., when tournament takes place in more than one country)
            :param kwargs: optional parameters of the model
            :returns M: M[i,j,s] = probability of i facing j with outcome s (s=0,s=1,s=2)
            :rtype: numpy array
            '''
    n = len(ratings)
    M = np.zeros(n * n * 3)  # upper triangular matrix contains all match outcome probabilities of all paired teams
    M.shape = (n, n, 3)
    scalar_hfa = kwargs.get('scalar_hfa', 1.0)
    for i in range(0, n - 1):
        for j in range(i + 1, n):
            # remove the optional argument and only insert it back if *only* either i or j is a home team
            # if both would have a hfa but play each other -> hfas cancel each other out (think of new zealand and australia as hosts for the women's world cup)
            if 'scalar_hfa' in kwargs:
                kwargs.pop('scalar_hfa')

            if (i in hfa_teams) and (j not in hfa_teams):
                kwargs['scalar_hfa'] = scalar_hfa

            # swap order of i and j for the model since only the first team can have the home advantage
            if (j in hfa_teams) and (i not in hfa_teams):
                kwargs['scalar_hfa'] = scalar_hfa
                M[i, j, 2] = M[j, i, 0] = transform(ratings[j], ratings[i], 0, **kwargs)
                M[i, j, 1] = M[j, i, 1] = transform(ratings[j], ratings[i], 1, **kwargs)
                M[i, j, 0] = M[j, i, 2] = transform(ratings[j], ratings[i], 2, **kwargs)

                continue

            M[i, j, 0] = M[j, i, 2] = transform(ratings[i], ratings[j], 0, **kwargs)
            M[i, j, 1] = M[j, i, 1] = transform(ratings[i], ratings[j], 1, **kwargs)
            M[i, j, 2] = M[j, i, 0] = transform(ratings[i], ratings[j], 2, **kwargs)

    return M
