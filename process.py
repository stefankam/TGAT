import json
import numpy as np
import pandas as pd

def preprocess(data_name):
    u_list, i_list, ts_list, label_list = [], [], [], []
    feat_l = []
    idx_list = []
    
    with open(data_name) as f:
        s = next(f)
        print(s)
        for idx, line in enumerate(f):
            e = line.strip().split(',')
            u = int(e[0], 16)
            i = int(e[1], 16)
            
            
            
            ts = float(e[2])
            label = int(e[3])
            
            feat = np.array([x for x in e[4:]])
            
            u_list.append(u)
            i_list.append(i)
            ts_list.append(ts)
            label_list.append(label)
            idx_list.append(idx)
            
            feat_l.append(feat)
    return pd.DataFrame({'u': u_list, 
                         'i':i_list, 
                         'ts':ts_list, 
                         'label':label_list, 
                         'idx':idx_list}), np.array(feat_l)



def reindex(df):
    # Create mappings for user and item addresses to continuous indices
    user_mapping = {u: idx for idx, u in enumerate(df.u.unique())}
    item_mapping = {i: idx + len(user_mapping) for idx, i in enumerate(df.i.unique())}

    # Apply the mappings
    new_df = df.copy()
    new_df.u = new_df.u.map(user_mapping)
    new_df.i = new_df.i.map(item_mapping)
    new_df.idx += 1  # If necessary, adjust idx as well

    return new_df



def run(data_name):
    PATH = './ETH_processed/{}.csv'.format(data_name)
    OUT_DF = './ETH_processed/ml_{}.csv'.format(data_name)
    OUT_FEAT = './ETH_processed/ml_{}.npy'.format(data_name)
    OUT_NODE_FEAT = './ETH_processed/ml_{}_node.npy'.format(data_name)

    df, feat = preprocess(PATH)
    new_df = reindex(df)

    # Instead of max_idx, use the length of unique user and item indices
    total_nodes = len(new_df.u.unique()) + len(new_df.i.unique())

    # Create a zero matrix for node features
    rand_feat = np.zeros((total_nodes, feat.shape[1]))

    new_df.to_csv(OUT_DF, index=False)
    np.save(OUT_FEAT, feat)
    np.save(OUT_NODE_FEAT, rand_feat)
    
    
run('eth_block_18168871_18168890_processed')

#run('reddit')