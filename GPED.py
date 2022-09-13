import random

import numpy as np
from deap.tools import selRandom
from sklearn.decomposition import PCA


def cart2pol(x, y):
    """
    Convert a Cartesian coordinate to a polar coordinate.
    """
    rho = np.sqrt(x ** 2 + y ** 2)
    phi = np.arctan2(y, x)
    return (rho, phi)


def get_semantic_matrix(case_values):
    semantic_matrix = case_values
    try:
        semantic_matrix = PCA(2).fit_transform(semantic_matrix)
    except:
        semantic_matrix = PCA(2, svd_solver='randomized').fit_transform(semantic_matrix)
    rho, phi = cart2pol(semantic_matrix[:, 0], semantic_matrix[:, 1])
    # last one is the real target
    rho -= rho[-1]
    phi -= phi[-1]
    assert rho[-1] == 0
    assert phi[-1] == 0
    rho = rho[:-1]
    phi = phi[:-1]

    bins = np.linspace(rho.min(), rho.max(), 10)
    rho = np.digitize(rho, bins)
    assert np.all(rho <= 10) & np.all(rho > 0)
    bins = np.linspace(phi.min(), phi.max(), 8)
    phi = np.digitize(phi, bins)
    assert np.all(phi <= 8) & np.all(phi > 0)
    return phi, rho


def selGPED(individuals, k, phi=None, rho=None, real_target=None):
    individuals = list(filter(lambda x: x.fitness.wvalues >
                                        10 * np.median([x.fitness.wvalues for x in individuals]), individuals))
    if phi is None and rho is None:
        phi, rho = get_semantic_matrix(np.array([ind.prediction for ind in individuals] + [real_target]))

    def selTournamentPlus(individuals, k, tournsize):
        """
        Select individuals based on case values
        """
        chosen = []
        for i in range(k):
            if len(individuals) > tournsize:
                aspirants = selRandom(individuals, tournsize)
            else:
                aspirants = individuals
            chosen.append(np.argmax(np.sum(aspirants, axis=1)))
        return chosen

    assert k % 2 == 0, 'k must be an even integer'
    inds = []
    case_values = np.array([ind.fitness.wvalues for ind in individuals])

    for _ in range(k // 2):
        index = selTournamentPlus(case_values, 1, 7)
        parent_a = index[0]
        r = random.random()
        parent_b = None
        indexes = np.arange(0, len(phi))
        if r < 0.2:
            # high, opposite direction
            if phi[parent_a] <= 4:
                tmp_indexes = indexes[phi == (phi[parent_a] + 4)]
                if len(tmp_indexes) > 0:
                    parent_b = selTournamentPlus(case_values[tmp_indexes], 1, 7)[0]
                    parent_b = tmp_indexes[parent_b]
            else:
                tmp_indexes = indexes[phi == (phi[parent_a] - 4)]
                if len(tmp_indexes) > 0:
                    parent_b = selTournamentPlus(case_values[tmp_indexes], 1, 7)[0]
                    parent_b = tmp_indexes[parent_b]
        if r < 0.4 and parent_b is None:
            # median
            tmp_indexes = indexes[(phi != phi[parent_a]) & (rho != rho[parent_a])]
            if len(tmp_indexes) > 0:
                parent_b = selTournamentPlus(case_values[tmp_indexes], 1, 7)[0]
                parent_b = tmp_indexes[parent_b]
        if r < 0.8 and parent_b is None:
            # standard
            tmp_indexes = indexes[(phi != phi[parent_a]) | (rho != rho[parent_a])]
            if len(tmp_indexes) > 0:
                parent_b = selTournamentPlus(case_values[tmp_indexes], 1, 7)[0]
                parent_b = tmp_indexes[parent_b]
        if parent_b is None:
            # low
            tmp_indexes = indexes[(phi == phi[parent_a])]
            if len(tmp_indexes) > 0:
                parent_b = selTournamentPlus(case_values[tmp_indexes], 1, 7)[0]
                parent_b = tmp_indexes[parent_b]
        inds.append(individuals[parent_a])
        inds.append(individuals[parent_b])
    return inds
