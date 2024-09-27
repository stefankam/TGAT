import random
import matplotlib.pyplot as plt
import pickle

import torch_geometric
from torch import nn
import networkx as nx
import torch.nn.functional as F
from sklearn.svm import OneClassSVM
from sklearn.cluster import DBSCAN
import numpy as np
from multiprocessing import Pool
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
import numpy as np
import pickle
import random
import matplotlib.pyplot as plt
import torch_geometric.utils
import torch
import networkx as nx
import pickle
from torch_geometric.data import Data

# Load graph
graph = pickle.load(open('data/eth_block_18168871_18168890.pickle', 'rb'))

# Create a mapping from node IDs (strings) to integer indices
node_id_map = {node: idx for idx, node in enumerate(graph.nodes())}

# Convert node features
node_features = [
    (
        float(node_data.get('incoming_value_variance', 0)),
        float(node_data.get('outgoing_value_variance', 0)),
        float(node_data.get('activity_rate', 0)),
        float(node_data.get('change_in_activity', 0)),
        float(node_data.get('time_since_last', 0)),
        float(node_data.get('tx_volume', 0)),
        float(node_data.get('frequent_large_transfers', 0)),
        float(node_data.get('gas_price', 0)),
        float(node_data.get('token_swaps', 0)),
        float(node_data.get('smart_contract_interactions', 0))
    )
    for node, node_data in graph.nodes(data=True)
]
node_features = torch.tensor(node_features, dtype=torch.float32)

# Convert the weighted adjacency matrix to a dense matrix
adj_matrix_dense = nx.to_numpy_array(graph, weight='weight')
# Convert the dense matrix to a PyTorch tensor
adj_matrix = torch.tensor(adj_matrix_dense, dtype=torch.float32)
data = Data(x=node_features, edge_index=adj_matrix.nonzero().t())

# Verify the number of node features
num_nodes = len(node_features)
print(f"Number of nodes in features: {num_nodes}")

# Convert edge_index
edge_index = []
edge_attr = []

for u, v, data in graph.edges(data=True):
    if u in node_id_map and v in node_id_map:
        edge_index.append([node_id_map[u], node_id_map[v]])
        edge_attr.append(data.get('timestamp', 0))
    else:
        print(f"Warning: Node ID {u} or {v} not found in node_id_map")

# Convert edge_index and edge_attr to tensors
edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
# Assuming edge_attr should be [num_edges, num_edge_features]
edge_attr = torch.tensor(edge_attr, dtype=torch.float32)
if edge_attr.dim() == 1:
    # If edge_attr is 1D, assume single feature per edge
    edge_attr = edge_attr.view(-1, 1)  # Shape [num_edges, 1]


# Verify edge index dimensions
print("Edge index shape:", edge_index.shape)
print("Edge attributes shape:", edge_attr.shape)
print("First 5 edge indices:", edge_index[:, :5])  # Display first 5 edges for verification

# Create PyTorch Geometric Data object
data = Data(x=node_features, edge_index=edge_index, edge_attr=edge_attr)

# Check for out-of-bounds indices
max_index = edge_index.max().item()
if max_index >= num_nodes:
    print(f"Error: Index {max_index} is out of bounds for node features with size {num_nodes}")

# Print for verification
print("Node features shape:", node_features.shape)

# Split data into training and test sets
num_samples = len(node_features)
num_train_samples = int(.8 * num_samples)  # You can adjust the split ratio
train_indices, test_indices = train_test_split(range(num_samples), train_size=num_train_samples, random_state=42)

# Training data
train_data = Data(x=node_features[train_indices], edge_index=adj_matrix[train_indices][:, train_indices].nonzero().t())

# Test data
test_data = Data(x=node_features[test_indices], edge_index=adj_matrix[test_indices][:, test_indices].nonzero().t())

class TGATModel(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, heads, dropout=0.5):
        super(TGATModel, self).__init__()
        # The number of output features of the first layer should match the number of input features for the second layer
        self.conv1 = TGATConv(in_channels, hidden_channels, heads=heads, dropout=dropout)
        self.conv2 = TGATConv(hidden_channels * heads, out_channels, heads=heads, dropout=dropout)

    def forward(self, x, edge_index, edge_attr):
        x = self.conv1(x, edge_index, edge_attr)
        x = F.relu(x)
        x = self.conv2(x, edge_index, edge_attr)
        return x

# Instantiate the TGAT model
model = TGATModel(in_channels=10, hidden_channels=20, out_channels=10,heads=2)

# Define loss and optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
criterion = nn.MSELoss()


# Training function
def train_model(model, data, epochs=200):
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        out = model(data.x, data.edge_index, data.edge_attr)
        loss = criterion(out, data.x)
        loss.backward()
        optimizer.step()
    return model


# Train the model
trained_model = train_model(model, train_data)

# After training, get embeddings
trained_model.eval()
train_embeddings = trained_model(train_data.x, train_data.edge_index, train_data.edge_attr).cpu().detach().numpy()
test_embeddings = trained_model(test_data.x, test_data.edge_index, test_data.edge_attr).cpu().detach().numpy()


# Compute anomaly scores
def compute_anomaly_scores(embeddings, node_freqs_values):
    scores = []
    for idx, embedding in enumerate(embeddings):
        mean = np.mean(embedding)
        std = np.std(embedding)
        latest_value = embedding[-1]
        z_score = (latest_value - mean) / std
        weighted_z_score = z_score * node_freqs_values[idx]
        scores.append(weighted_z_score)
    return scores


train_anomaly_scores = compute_anomaly_scores(train_embeddings, list(node_freqs.values()))
test_anomaly_scores = compute_anomaly_scores(test_embeddings, list(node_freqs.values()))

# Detection and evaluation
from sklearn.metrics import precision_score, recall_score, f1_score


def evaluate_anomalies(anomaly_scores, true_labels, thresholds):
    precisions, recalls, f1_scores = [], [], []
    for threshold in thresholds:
        detected_anomalies = [idx for idx, score in enumerate(anomaly_scores) if abs(score) > threshold]
        detected_labels = [1 if idx in detected_anomalies else 0 for idx in range(len(true_labels))]

        precision = precision_score(true_labels, detected_labels)
        recall = recall_score(true_labels, detected_labels)
        f1 = f1_score(true_labels, detected_labels)

        precisions.append(precision)
        recalls.append(recall)
        f1_scores.append(f1)

    return np.mean(precisions), np.mean(recalls), np.mean(f1_scores)


thresholds = [1.0, 1.5, 2.0]
test_true_labels = [1 if idx in test_indices else 0 for idx in range(len(node_features))]

avg_precision, avg_recall, avg_f1_score = evaluate_anomalies(test_anomaly_scores, test_true_labels, thresholds)

# Print results
print(f"Average Precision: {avg_precision:.3f}")
print(f"Average Recall: {avg_recall:.3f}")
print(f"Average F1-score: {avg_f1_score:.3f}")
