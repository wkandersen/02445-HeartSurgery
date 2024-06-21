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
    
    # Create an empty DataFrame to store the results
    results = pd.DataFrame(columns=['Prediction_Set_A', 'Prediction_Set_B', 'n00', 'n01', 'n10', 'n11', 'P_Value'])
    
    for (i, pred_A), (j, pred_B) in combinations(enumerate(pred_set), 2):
        print(f"Comparing prediction sets {names[i]} and {names[j]}")
        
        n00, n01, n10, n11 = compute_mcnemar_contingency(holdout_labels, pred_A, pred_B)
        
        # Create confusion matrix
        confusion_matrix = [[n00, n01], [n10, n11]]
        result = mcnemar(confusion_matrix, exact=False, correction=True)
        
        # Append results to the DataFrame
        results = results.append({
            'Prediction_Set_A': names[i],
            'Prediction_Set_B': names[j],
            'n00': n00,
            'n01': n01,
            'n10': n10,
            'n11': n11,
            'P_Value': result.pvalue
        }, ignore_index=True)
    
    return results

# Example usage
# names = ['pre', 'phase1', 'phase2', 'phase3', 'phase4', 'phase5']

# results = compare_prediction_sets(pred_set, holdout_labels, names)

# display(results)