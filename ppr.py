import torch
import numpy as np
import networkx as nx
from scipy.sparse import coo_matrix

def convert_sparse_to_tensor(adj_matrix, to_dense=True):
    """
    Convert a sparse CSR matrix to a PyTorch tensor.
    
    Args:
        adj_matrix (scipy.sparse.csr_matrix): Input sparse matrix.
        to_dense (bool): If True, return a dense tensor; otherwise, return a sparse tensor.
        
    Returns:
        torch.Tensor or torch.sparse.FloatTensor
    """
    if to_dense:
        return torch.tensor(adj_matrix.todense(), dtype=torch.float32)
    else:
        row, col = adj_matrix.nonzero()
        indices = torch.tensor(np.array([row, col]), dtype=torch.long)
        values = torch.tensor(adj_matrix.data, dtype=torch.float32)
        shape = adj_matrix.shape
        return torch.sparse.FloatTensor(indices, values, torch.Size(shape))


def att_walk_multigraph(G, i=0):
    """
    Compute attention-based adjacency for a MultiGraph.
    
    Args:
        G (networkx.MultiGraph): Input graph with node embeddings stored in 'embedding'.
        i (int): If 0, self-loops are added.
        
    Returns:
        torch.sparse.FloatTensor: Attention-based adjacency matrix.
    """
    # Collect node features
    features = torch.stack([data['embedding'] for _, data in G.nodes(data=True)])

    # Build sparse adjacency from edge weights
    adj = nx.to_scipy_sparse_array(G, weight='weight', format='coo')
    row, col = adj.row, adj.col
    values = torch.tensor(adj.data, dtype=torch.float32)
    edge_index = torch.sparse_coo_tensor(
        indices=torch.tensor(np.stack([row, col])),
        values=values,
        size=(features.shape[0], features.shape[0])
    ).coalesce()

    n_node = features.shape[0]
    row, col = edge_index.indices()

    # Compute similarity matrix
    adj_dense = edge_index.to_dense()
    sim_matrix = torch.tensor(estimated_similarity(features, adj_dense), dtype=torch.float32)
    sim_matrix_norm = torch.nn.functional.normalize(sim_matrix, p=1, dim=1)

    # Prune edges with lowest attention
    sim = sim_matrix_norm[row, col]
    C = 0.2
    cut_num = int(edge_index._nnz() * C)
    _, indices = torch.topk(sim, cut_num, largest=False)
    sim[indices] = 0

    att_dense = torch.sparse_coo_tensor(torch.stack([row, col]), sim, (n_node, n_node)).to_dense()

    if i == 0:
        att_dense -= torch.diag(att_dense.diagonal())

    att_dense_norm = torch.nn.functional.normalize(att_dense, p=1, dim=1)

    if i == 0:
        degree = (att_dense_norm != 0).sum(1)
        lam = 1 / (degree + 1)
        self_weight = torch.diag(lam)
        att = att_dense_norm + self_weight
    else:
        att = att_dense_norm

    row, col = att.nonzero().T
    att_edge_weight = torch.exp(att[row, col])
    new_adj = torch.sparse_coo_tensor(torch.stack([row, col]), att_edge_weight, (n_node, n_node)).coalesce()
    return new_adj


def estimated_similarity(fea, adj_matrix):
    """
    Estimate similarity-based attention matrix using attributed random walk.
    
    Args:
        fea (torch.Tensor): Node features [num_nodes, feature_dim].
        adj_matrix (torch.Tensor): Dense adjacency matrix [num_nodes, num_nodes].
        
    Returns:
        np.ndarray: Estimated attention matrix.
    """
    num_nodes = fea.shape[0]

    # Degree normalization
    degree_matrix = calculate_degree_matrix(adj_matrix, num_nodes)
    degree_matrix_inv = inverse_sparse_matrix(degree_matrix)
    topo_pro = torch.matmul(degree_matrix_inv, adj_matrix)

    # Feature-based similarity
    inner_products = torch.mm(fea, fea.t())
    all_sum = torch.sum(inner_products, dim=1).unsqueeze(1)
    normalized_weights = inner_products / (all_sum + 1e-10)
    att_pro = normalized_weights.detach().cpu().numpy().astype(np.float32)

    # Combine topology and feature-based matrices
    alpha = 0.2
    beta = 0.35
    transition_matrix = (1 - beta) * topo_pro.cpu().numpy() + beta * att_pro

    # Perform attributed random walk
    N = 25
    S_estimated = attributed_random_walk(N, num_nodes, alpha, transition_matrix)
    return S_estimated.astype(np.float32)


def calculate_degree_matrix(adj_matrix, num_nodes):
    """
    Compute diagonal degree matrix from adjacency matrix.
    """
    degrees = torch.sum(adj_matrix, dim=1) + 1
    indices = torch.stack([torch.arange(num_nodes), torch.arange(num_nodes)])
    return torch.sparse.FloatTensor(indices, degrees, (num_nodes, num_nodes))


def inverse_sparse_matrix(sparse_matrix):
    """
    Compute dense inverse of a sparse diagonal matrix.
    """
    return torch.inverse(sparse_matrix.to_dense())


def attributed_random_walk(N, num_nodes, alpha, transition_matrix):
    """
    Compute attributed random walk similarity matrix.
    """
    S_estimated = np.zeros((num_nodes, num_nodes))
    att_random_walk = alpha * np.eye(num_nodes)

    for i in range(N):
        if i == 0:
            S_estimated += att_random_walk
        else:
            att_random_walk = np.matmul((1 - alpha) * transition_matrix, att_random_walk)
            S_estimated += att_random_walk

    return S_estimated
