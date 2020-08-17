from itertools import combinations
from sklearn.metrics import mean_squared_error


def concordance_index(gold_truths, predictions):
    gold_combs, pred_combs = combinations(gold_truths, 2), combinations(predictions, 2)
    nominator, denominator = 0, 0
    for (g1, g2), (p1, p2) in zip(gold_combs, pred_combs):
        if g2 > g1:
            nominator = nominator + 1 * (p2 > p1) + 0.5 * (p2 == p1)
            denominator = denominator + 1

    return nominator / denominator


def mse(gold_truths, predictions):
    return mean_squared_error(gold_truths, predictions)
