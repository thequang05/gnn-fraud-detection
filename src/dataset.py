import os
import pandas as pd
import numpy as np
import torch
from torch_geometric.data import Data

_SRC_DIR = os.path.dirname(os.path.abspath(__file__))
_DEFAULT_DATA_DIR = os.path.join(_SRC_DIR, '..', 'data', 'raw')

def load_elliptic_data(data_dir=_DEFAULT_DATA_DIR):
    df_classes = pd.read_csv(f'{data_dir}/elliptic_txs_classes.csv')
    df_edges = pd.read_csv(f'{data_dir}/elliptic_txs_edgelist.csv')
    df_features = pd.read_csv(f'{data_dir}/elliptic_txs_features.csv',header=None)
    node=df_features[0].values
    map_id={j:i for i,j in enumerate(node)}
    features=df_features.iloc[:,1:].values
    x=torch.tensor(features,dtype=torch.float32)
    df_classes=df_classes.set_index('txId').reindex(node).reset_index()
    df_classes['class']=df_classes['class'].map({'1':1,'2':0,'unknown':-1})
    y=torch.tensor(df_classes['class'].values,dtype=torch.long)
    df_edges['txId1']=df_edges['txId1'].map(map_id)
    df_edges['txId2']=df_edges['txId2'].map(map_id)
    edge_index=torch.tensor(df_edges.values.T,dtype=torch.long)
    data = Data(x=x, edge_index=edge_index, y=y)
    print("Đang tạo Mask chia Train/Test theo Thời gian...")
    time_steps = data.x[:, 0]  
    data.train_mask = (time_steps <= 34)
    data.test_mask = (time_steps > 34)
    return data
if __name__ == "__main__":

    dataset_path = 'data/raw'
    graph_data = load_elliptic_data(dataset_path)
    
    print("\n--- KẾT QUẢ ---")
    print(f"Số lượng Nodes: {graph_data.num_nodes}")
    print(f"Số lượng Edges: {graph_data.num_edges}")
    print(f"Kích thước ma trận x: {graph_data.x.shape}")
    print(f"Kích thước vector y: {graph_data.y.shape}")
    
