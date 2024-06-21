from statsmodels.stats.contingency_tables import mcnemar
from itertools import combinations
import pandas as pd

def compute_mcnemar_contingency(holdout_labels, pred_A, pred_B):
    n00 = n01 = n10 = n11 = 0
    for actual, a, b in zip(holdout_labels, pred_A, pred_B):
        if a == actual and b == actual:
            n00 += 1
        elif a == actual and b != actual:
            n01 += 1
        elif a != actual and b == actual:
            n10 += 1
        else:
            n11 += 1
    return n00, n01, n10, n11

def compare_prediction_sets(pred_set, holdout_labels, names=None):
    if names is None:
        names = [f'phase{i}' for i in range(len(pred_set))]
    
    # Ensure names list is the same length as pred_set
    if len(names) != len(pred_set):
        raise ValueError("The length of names must match the length of pred_set")
    
    # Create a list of tuples containing results
    results_tuples = [
        (
            names[i],
            names[j],
            *compute_mcnemar_contingency(holdout_labels, pred_set[i], pred_set[j]),
            mcnemar([[n00, n01], [n10, n11]], exact=False, correction=True).pvalue
        )
        for i, j in combinations(range(len(pred_set)), 2)
    ]
    
    # Convert list of tuples to DataFrame
    results = pd.DataFrame(results_tuples, columns=['Prediction_Set_A', 'Prediction_Set_B', 'n00', 'n01', 'n10', 'n11', 'P_Value'])
    
    return results

# Example usage
# names = ['pre', 'phase1', 'phase2', 'phase3', 'phase4', 'phase5']

# results = compare_prediction_sets(pred_set, holdout_labels, names)

# display(results)