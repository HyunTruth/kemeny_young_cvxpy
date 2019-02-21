# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.3'
#       jupytext_version: 1.0.0
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

import numpy as np
import cvxpy as cp
from itertools import combinations, permutations


def build_weight_graph(rank_matrix):
    num_voters, num_candidates = rank_matrix.shape

    # initialize matrix w for weight graph
    w = np.zeros((num_candidates, num_candidates))
    for a, b in combinations(range(num_candidates), 2):
        prefer = rank_matrix[:, a] - rank_matrix[:, b]

        # preferring a to b
        h_ab = np.sum(prefer < 0)

        # preferring b to a
        h_ba = np.sum(prefer > 0)

        # if h_ab > h_ba, draw edge with w_ab
        if h_ab > h_ba:
            w[a, b] = h_ab - h_ba
        # if h_ba > h_ab, draw edge with w_ba
        elif h_ba > h_ab:
            w[b, a] = h_ba - h_ab

    return w


def kenemy_young(rank_matrix):
    w = build_weight_graph(rank_matrix)
    num_candidates = w.shape[0]
    # x cannot be negative (no negative rank!)
    x = cp.Variable(w.shape, nonneg=True)

    # objective function of min sum_e(w_e*x_e) in a matrix multiplication form
    obj = cp.Minimize(cp.sum(w.T*x))
    constraints = []

    # pairwise constraints, so that x_ab + x_ba = 1 for all distinct a,b pair in our candidates
    for comb in combinations(range(num_candidates), 2):
        a, b = comb
        constraints.append(x[a, b] + x[b, a] == 1)

    # triangular constraints, so that x_ab + x_bc + x_ca >= 1 for all distinct a,b,c in our candidates
    for perm in permutations(range(num_candidates), 3):
        a, b, c = perm
        constraints.append(x[a, b] + x[b, c] + x[c, a] >= 1)
    
    # state problem
    prob = cp.Problem(obj, constraints)
    
    # Returns the optimal value.
    prob.solve() 
    print("status:", prob.status)
    print("optimal value", prob.value)
    
    # recast into an integer
    x_val = np.round((x.value))
    
    # columnar sum into total rank
    agg_rank = x_val.sum(axis=1)
    
    # move the starting index from 0 -> 1
    agg_rank += 1
    
    return agg_rank


# Use numpy matrix or numpy array, with each row as each voter's rank choice

example_arr = np.array([[0, 1, 2, 3, 4],
                        [0, 1, 3, 2, 4],
                        [4, 1, 2, 0, 3],
                        [4, 1, 0, 2, 3],
                        [4, 1, 3, 2, 0],
                        [4, 1, 3, 2, 0]])

kenemy_young(example_arr)

example_mat = np.matrix(example_arr)

print(example_mat)

kenemy_young(example_mat)
