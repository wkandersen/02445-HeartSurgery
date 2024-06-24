#%%
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc
import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset
import torch.nn as nn
import torch.optim as optim


#%%
# Load data function
def load_data(file_path):
    df = pd.read_csv(file_path)
    return df.values

# Directory containing data
data_dir = "data"
files = [f for f in os.listdir(data_dir) if os.path.isfile(os.path.join(data_dir, f))]
patient_ids = set(f.split('_')[0] for f in files)

# Initialize data lists
pre_data = []
op_data = []
labels = []

# Load patient data
for pid in patient_ids:
    pre_file = os.path.join(data_dir, f"{pid}_pre.csv")
    op_files = sorted([os.path.join(data_dir, f) for f in files if f.startswith(f"{pid}_op")], key=lambda x: x)
    
    pre_data.append(load_data(pre_file))
    op_data.append([load_data(op_file) for op_file in op_files])
    labels.append(pid)

pre_data = np.array(pre_data)
op_data = np.array(op_data)
labels = np.array(labels)

# Split patient IDs into training, validation, and hold-out sets
train_ids, holdout_ids, train_labels, holdout_labels = train_test_split(labels, labels, test_size=1000, stratify=labels)
train_ids, val_ids, train_labels, val_labels = train_test_split(train_ids, train_labels, test_size=0.2, stratify=train_labels)

# Function to get data by patient IDs
def get_data_by_ids(ids, all_pre_data, all_op_data, all_labels):
    idx = [np.where(all_labels == pid)[0][0] for pid in ids]
    return all_pre_data[idx], all_op_data[idx], all_labels[idx]

# Get train, validation, and holdout data
train_pre_data, train_op_data, train_labels = get_data_by_ids(train_ids, pre_data, op_data, labels)
val_pre_data, val_op_data, val_labels = get_data_by_ids(val_ids, pre_data, op_data, labels)
holdout_pre_data, holdout_op_data, holdout_labels = get_data_by_ids(holdout_ids, pre_data, op_data, labels)

# Convert to PyTorch tensors
train_pre_data = torch.tensor(train_pre_data, dtype=torch.float32)
train_labels = torch.tensor(train_labels, dtype=torch.float32)
val_pre_data = torch.tensor(val_pre_data, dtype=torch.float32)
val_labels = torch.tensor(val_labels, dtype=torch.float32)
holdout_pre_data = torch.tensor(holdout_pre_data, dtype=torch.float32)
holdout_labels = torch.tensor(holdout_labels, dtype=torch.float32)

train_op_data = torch.tensor(train_op_data, dtype=torch.float32)
val_op_data = torch.tensor(val_op_data, dtype=torch.float32)
holdout_op_data = torch.tensor(holdout_op_data, dtype=torch.float32)

# Create DataLoader
train_dataset = TensorDataset(train_pre_data, train_labels)
val_dataset = TensorDataset(val_pre_data, val_labels)
holdout_dataset = TensorDataset(holdout_pre_data, holdout_labels)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
holdout_loader = DataLoader(holdout_dataset, batch_size=32, shuffle=False)


#%%
class LogisticRegressionModel(nn.Module):
    def __init__(self, input_dim):
        super(LogisticRegressionModel, self).__init__()
        self.linear = nn.Linear(input_dim, 1)

    def forward(self, x):
        out = torch.sigmoid(self.linear(x))
        return out

#%%
def train_model(model, train_loader, val_loader, num_epochs=50, learning_rate=0.001, weight_decay=0.01):
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    for epoch in range(num_epochs):
        model.train()
        for i, (data, labels) in enumerate(train_loader):
            outputs = model(data)
            loss = criterion(outputs.squeeze(), labels.float())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for i, (data, labels) in enumerate(val_loader):
                outputs = model(data)
                loss = criterion(outputs.squeeze(), labels.float())
                val_loss += loss.item()

        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {loss.item()}, Val Loss: {val_loss/len(val_loader)}')

def evaluate_model(model, data_loader):
    model.eval()
    all_labels = []
    all_outputs = []
    with torch.no_grad():
        for data, labels in data_loader:
            outputs = model(data)
            all_labels.extend(labels.numpy())
            all_outputs.extend(outputs.squeeze().numpy())

    roc_auc = roc_auc_score(all_labels, all_outputs)
    precision, recall, _ = precision_recall_curve(all_labels, all_outputs)
    pr_auc = auc(recall, precision)

    return roc_auc, pr_auc
#%%
input_dim = train_pre_data.shape[1]
model = LogisticRegressionModel(input_dim)

train_model(model, train_loader, val_loader)

roc_auc, pr_auc = evaluate_model(model, holdout_loader)
print(f'Preoperative Data - ROC-AUC: {roc_auc}, PR-AUC: {pr_auc}')


#%%
# Function to concatenate preoperative and operative data
def concatenate_data(pre_data, op_data, phases):
    concatenated_data = pre_data
    for i in range(phases):
        concatenated_data = torch.cat((concatenated_data, op_data[:, i, :]), dim=1)
    return concatenated_data

phases = train_op_data.shape[1]  # Number of phases

for phase in range(phases + 1):
    print(f"Training model with {phase} phases of operative data...")

    if phase > 0:
        train_data = concatenate_data(train_pre_data, train_op_data, phase)
        val_data = concatenate_data(val_pre_data, val_op_data, phase)
        holdout_data = concatenate_data(holdout_pre_data, holdout_op_data, phase)
    else:
        train_data = train_pre_data
        val_data = val_pre_data
        holdout_data = holdout_pre_data

    # Update DataLoader
    train_dataset = TensorDataset(train_data, train_labels)
    val_dataset = TensorDataset(val_data, val_labels)
    holdout_dataset = TensorDataset(holdout_data, holdout_labels)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    holdout_loader = DataLoader(holdout_dataset, batch_size=32, shuffle=False)

    # Initialize and train model
    input_dim = train_data.shape[1]
    model = LogisticRegressionModel(input_dim)
    train_model(model, train_loader, val_loader)

    # Evaluate model
    roc_auc, pr_auc = evaluate_model(model, holdout_loader)
    print(f"Phase {phase}: ROC-AUC: {roc_auc}, PR-AUC: {pr_auc}")
