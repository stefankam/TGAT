import csv
import os
import pickle
import networkx as nx

# Load the graph
graph = pickle.load(open('data/eth_block_18168871_18168890.pickle', 'rb'))

# Folder to save the processed data
output_folder = 'ETH_processed'
os.makedirs(output_folder, exist_ok=True)

# File to save the CSV
output_file = os.path.join(output_folder, 'eth_block_18168871_18168890_processed.csv')

# Headers for node features
node_feature_headers = [
    'outgoing_tx_count', 'incoming_tx_count', 'incoming_value_list', 'outgoing_value_list',
    'incoming_tx_volume', 'outgoing_tx_volume', 'incoming_value_variance', 'outgoing_value_variance',
    'activity_rate', 'change_in_activity', 'time_since_last', 'tx_volume', 'ico_participation',
    'flash_loan', 'token_airdrop', 'phishing', 'frequent_large_transfers', 'gas_price',
    'token_swaps', 'smart_contract_interactions', 'last_transaction_block'
]

# Function to extract node feature values as a comma-separated string
def get_node_features(graph, node, feature_headers):
    features = graph.nodes[node]  # Get features for the node
    feature_values = [str(features.get(header, "")) for header in feature_headers]  # Extract values for each feature
    return ",".join(feature_values)  # Return as a comma-separated string

# Open the CSV file for writing
with open(output_file, mode='w', newline='') as file:
    writer = csv.writer(file)

    # Write the header row
    header = ['user', 'item', 'timestamp', 'state_label'] + [f'source_{header}' for header in node_feature_headers] + [f'target_{header}' for header in node_feature_headers]
    writer.writerow(header)

    # Iterate over all edges and write the data to the CSV
    for edge in graph.edges(data=True):
        source, target, edge_attrs = edge  # Unpack the source, target, and attributes

        # Extract the edge attributes: timestamp is required, others are optional
        timestamp = edge_attrs.get('timestamp', "")
        state_label = edge_attrs.get('weight', "")  # Assuming 'weight' could be the state_label

        # Get the node feature values as comma-separated strings
        source_features = get_node_features(graph, source, node_feature_headers)
        target_features = get_node_features(graph, target, node_feature_headers)

        # Write the row to the CSV
        writer.writerow([source, target, timestamp, state_label] + source_features.split(",") + target_features.split(","))

print(f"CSV file saved at {output_file}")
