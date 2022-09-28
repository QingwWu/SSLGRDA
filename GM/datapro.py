
import torch
from torch_sparse import coalesce
from torch_geometric.data import Data
from torch_geometric.utils import remove_self_loops, add_self_loops
from dataprocessing import dataload
import numpy as np

def read_data(name,num):

    dataload(name,num)
    edge = np.loadtxt('D:/bioinformatics/2_model/GCN_model/SSLGRDA/datasets/edge.txt')
    nodenum =int(edge.max()+1)
    # fea = np.loadtxt('D:/bioinformatics/2_model/GCN_model/SUGRL/datasets/Cora/raw/fea.txt')
    # fea = np.loadtxt('D:/bioinformatics/2_model/GCN_model/GraphMAE/data/feature.txt')
    # x = torch.Tensor(fea)
    x = torch.eye(nodenum)
    edge_index =  edge_index_from_dict(edge, num_nodes=nodenum)
    data = Data(x=x, edge_index=edge_index)
    return data

def edge_index_from_dict(graph_dict, num_nodes=None):
    row, col = [], []
    for i, j in graph_dict:
        row.append(i)
        col.append(j)
    edge_index = torch.stack([torch.tensor(row), torch.tensor(col)], dim=0).long()
    edge_index, _ = remove_self_loops(edge_index)
    # edge_index, _  = add_self_loops(edge_index)
    edge_index, _ = coalesce(edge_index, None, num_nodes, num_nodes)
    return edge_index






