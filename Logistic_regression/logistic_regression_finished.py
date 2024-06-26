import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset, TensorDataset
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split

data1 = pd.read_csv('pseudodata_præoperation.csv')
data2 = pd.read_csv('pseudodata_præoperation.csv')
data3 = pd.read_csv('pseudodata_præoperation.csv')
data4 = pd.read_csv('pseudodata_præoperation.csv')
data5 = pd.read_csv('pseudodata_præoperation.csv')
data6 = pd.read_csv('pseudodata_præoperation.csv')

base = data1
phase1 = pd.concat([data1, data2], axis = 1)
phase2 = pd.concat([data1, data2, data3], axis = 1)
phase3 = pd.concat([data1, data2, data3, data4], axis = 1)
phase4 = pd.concat([data1, data2, data3, data4, data5], axis = 1)
phase5 = pd.concat([data1, data2, data3, data4, data5, data6], axis = 1)

data_list = [base, phase1, phase2, phase3, phase4, phase5]
preds_log = []
models_log = []
holdout_log = []
true_log = []
for i in range(1):
    data = data_list[i]
    # Convert to numpy array and generate synthetic labels for demonstration
    X = data.to_numpy()
    y = np.random.choice([0, 1], size=len(data))

    # Standardize the data
    X = StandardScaler().fit_transform(X)

    # Convert to PyTorch tensors
    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32)

    # Create a dataset
    dataset = TensorDataset(X_tensor, y_tensor)

    # Split data into holdout set and remaining set
    train_indices, holdout_indices = train_test_split(np.arange(len(dataset)), test_size=500, random_state=42, stratify=y_tensor.numpy())

    holdout_set = Subset(dataset, holdout_indices)
    remaining_set = Subset(dataset, train_indices)

    y_1_percent = np.count_nonzero(y_tensor) / len(y_tensor)
    y_0_percent = 1 - y_1_percent
    print(f"Class 1 percentage: {y_1_percent * 100:.2f}%")
    print(f"Class 0 percentage: {y_0_percent * 100:.2f}%")
    # Define class weights for BCEWithLogitsLoss
    class_weights = torch.tensor([1/y_0_percent, 1/y_1_percent])

    # Define the logistic regression model with L2 regularization
    class LogisticRegressionModel(nn.Module):
        def __init__(self, input_dim):
            super(LogisticRegressionModel, self).__init__()
            self.linear = nn.Linear(input_dim, 1)
        
        def forward(self, x):
            out = torch.sigmoid(self.linear(x))
            return out

    input_dim = data.shape[1]

    # Function to get target tensor from Subset
    def get_targets(subset):
        return subset.dataset.tensors[1][subset.indices]

    # Get target tensors for holdout and remaining sets
    holdout_targets = get_targets(holdout_set)
    remaining_targets = get_targets(remaining_set)

    # Print class distributions
    print(f"Class distribution in holdout set: {np.unique(holdout_targets.numpy(), return_counts=True)}")
    print(f"Class distribution in remaining set: {np.unique(remaining_targets.numpy(), return_counts=True)}")

    weight_decay_values = np.logspace(-5, 1, 7)  # for example, try values from 1e-5 to 10

    # Function to train and evaluate the model with L2 regularization
    def train_and_evaluate_model(train_loader, val_loader, input_dim, num_epochs=10, lr=0.01, weight_decay=1e-5):
        model = LogisticRegressionModel(input_dim)
        criterion = nn.BCEWithLogitsLoss(pos_weight=class_weights[1])
        optimizer = optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay)
        
        # Training loop
        for epoch in range(num_epochs):
            model.train()
            for inputs, labels in train_loader:
                # Forward pass
                outputs = model(inputs)
                loss = criterion(outputs.squeeze(), labels)
                
                # Backward pass and optimization
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        
        # Evaluate on validation set
        model.eval()
        correct = 0
        total = 0
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for inputs, labels in val_loader:
                outputs = model(inputs)
                predicted = (outputs > 0.5).float().squeeze()
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                correct += (predicted == labels).sum().item()
                total += labels.size(0)

        f1 = f1_score(all_labels, all_preds)
        accuracy = correct / total
        return f1, model, accuracy, all_preds, all_labels

    outer_kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    # Function to perform grid search over weight_decay values
    def grid_search(train_set, outer_kf, weight_decays, input_dim, num_epochs=10, lr=0.01):
        best_weight_decay = None
        best_outer_f1 = 0
        y_remaining = y_tensor[train_indices].numpy()

        for weight_decay in weight_decays:
            outer_f1_scores = []
            
            for outer_train_index, outer_test_index in outer_kf.split(np.zeros(len(train_set)), y_remaining):
                
                outer_train_subset = Subset(train_set, outer_train_index)
                outer_test_subset = Subset(train_set, outer_test_index)
                
                # Inner 5-Fold Cross-Validation
                inner_kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
                inner_f1_scores = []
                
                for inner_train_index, inner_val_index in inner_kf.split(np.zeros(len(outer_train_index)), y_remaining[outer_train_index]):
                    
                    inner_train_indices = np.array(outer_train_index)[inner_train_index]
                    inner_val_indices = np.array(outer_train_index)[inner_val_index]
                    
                    inner_train_subset = Subset(train_set, inner_train_indices)
                    inner_val_subset = Subset(train_set, inner_val_indices)
                    
                    train_loader = DataLoader(inner_train_subset, batch_size=64, shuffle=True)
                    val_loader = DataLoader(inner_val_subset, batch_size=64, shuffle=False)
                    
                    f1, _, _, _, _ = train_and_evaluate_model(train_loader, val_loader, input_dim, num_epochs, lr, weight_decay)
                    inner_f1_scores.append(f1)
                            
                # Train on outer train subset and evaluate on outer test subset
                train_loader = DataLoader(outer_train_subset, batch_size=64, shuffle=True)
                test_loader = DataLoader(outer_test_subset, batch_size=64, shuffle=False)
                
                outer_f1, _, _, _, _ = train_and_evaluate_model(train_loader, test_loader, input_dim, num_epochs, lr, weight_decay)
                outer_f1_scores.append(outer_f1)
            
            mean_outer_f1 = np.mean(outer_f1_scores)
            print(f'Weight Decay: {weight_decay}, Mean Outer F1 Score: {mean_outer_f1:.4f}')
            
            if mean_outer_f1 > best_outer_f1:
                best_outer_f1 = mean_outer_f1
                best_weight_decay = weight_decay
        
        return best_weight_decay


    # Perform grid search over weight_decay values

    best_weight_decay = grid_search(remaining_set, outer_kf, weight_decay_values, input_dim, num_epochs=10, lr=0.01)

    print(f'Best Weight Decay: {best_weight_decay}')

    # Evaluate the final model on the holdout set
    final_model = LogisticRegressionModel(input_dim)
    train_loader = DataLoader(remaining_set, batch_size=64, shuffle=True)
    holdout_loader = DataLoader(holdout_set, batch_size=64, shuffle=False)

    # Train the final model with the best weight decay found
    holdout_f1, _, holdout_acc, holdout_pred, holdout_all_labels = train_and_evaluate_model(train_loader, holdout_loader, input_dim, num_epochs=10, lr=0.01, weight_decay=best_weight_decay)
    print(f'Final Model F1 Score on Holdout Set: {holdout_f1:.4f}')
    print(f'Holdout Set Accuracy: {holdout_acc * 100:.2f}%')

    models_log.append(final_model)
    holdout_log.append(holdout_loader)
    true_log.append(holdout_all_labels)
    preds_log.append(holdout_pred)

    # Evaluate precision and recall on holdout set
    final_model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for inputs, labels in holdout_loader:
            outputs = final_model(inputs)
            predicted = (outputs > 0.5).float()
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    holdout_precision = precision_score(all_labels, all_preds)
    holdout_recall = recall_score(all_labels, all_preds)

    print(f'Holdout Set Precision: {holdout_precision * 100:.2f}%')
    print(f'Holdout Set Recall: {holdout_recall * 100:.2f}%')

    # Open the file in write mode
    with open('output.txt', 'a') as file:
        print(f"Phase {i + 1}", file=file)
        print(f'Best Weight Decay: {best_weight_decay}', file=file)
        print(f'Final Model F1 Score on Holdout Set: {holdout_f1:.4f}', file=file)
        print(f'Holdout Set Accuracy: {holdout_acc * 100:.2f}%', file=file)
        print(f'Holdout Set Precision: {holdout_precision * 100:.2f}%', file=file)
        print(f'Holdout Set Recall: {holdout_recall * 100:.2f}%', file=file)
        print('\n', file=file)
        print('Output written to output.txt')
