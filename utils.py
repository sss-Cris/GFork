import numpy as np
import networkx as nx
from numpy.linalg import norm
import pickle as pkl
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh
import sys
import random
import re
from tqdm import tqdm
import torch
import logging

def to_tensor_func(data, device):
    """
    将数据转换为张量并移动到指定设备，自动跳过图对象
    
    参数:
        data: 输入数据（可以是np.ndarray、list、torch.Tensor、nx.Graph等）
        device: 目标设备（如 'cuda' 或 'cpu'）
        
    返回:
        转换后的张量（如果是数值数据）或原始数据（如果是图对象或其他非数值数据）
    """
    # 如果是NetworkX图对象，直接返回（不转换）
    if isinstance(data, (nx.Graph, nx.MultiGraph)):
        return data
    
    # 如果是NumPy数组（但dtype=object，包含不等长序列等）
    if isinstance(data, np.ndarray) and data.dtype == object:
        try:
            # 尝试转换为float32类型的常规数组（如果内部数据形状一致）
            return torch.from_numpy(data.astype(np.float32)).to(device)
        except (ValueError, TypeError):
            # 如果转换失败（如不等长数组），返回原始数据
            return data
    
    # 如果是常规NumPy数组（非object类型）
    elif isinstance(data, np.ndarray):
        return torch.from_numpy(data).float().to(device)
    
    # 如果是PyTorch张量
    elif isinstance(data, torch.Tensor):
        return data.float().to(device)
    
    # 其他情况（如Python列表）
    else:
        try:
            return torch.tensor(data).float().to(device)
        except:
            # 转换失败时返回原始数据
            return data

# 设置随机种子
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

def setup_logger(logger_name, filename, delete_old=False):
    """
    设置日志记录器，避免重复添加处理器。
    
    Args:
        logger_name (str): 日志记录器的名称。
        filename (str): 日志文件的路径。
        delete_old (bool): 是否删除旧日志文件并重新创建。
    
    Returns:
        logging.Logger: 配置好的日志记录器。
    """
    # 获取或创建日志记录器
    logger = logging.getLogger(logger_name)
    
    # 如果日志记录器已经存在处理器，则直接返回
    if logger.handlers:
        return logger
    
    # 设置日志记录器的级别
    logger.setLevel(logging.DEBUG)
    
    # 创建控制台处理器（输出到 stderr）
    stderr_handler = logging.StreamHandler(sys.stderr)
    stderr_handler.setLevel(logging.INFO)
    
    # 创建文件处理器
    file_handler = logging.FileHandler(filename, mode='w' if delete_old else 'a')
    file_handler.setLevel(logging.DEBUG)
    
    # 设置日志格式
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    stderr_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)
    
    # 添加处理器到日志记录器
    logger.addHandler(stderr_handler)
    logger.addHandler(file_handler)
    
    return logger

# 解析索引文件
def parse_index_file(filename):
    """Parse index file."""
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index

# 生成样本掩码
def sample_mask(idx, l):
    """Create mask."""
    mask = np.zeros(l)
    mask[idx] = 1
    return np.array(mask, dtype=np.bool)
    
import numpy as np
import sys
import pickle as pkl
from sklearn.utils import shuffle as sklearn_shuffle
import networkx as nx  # Make sure to import networkx

def load_data(dataset_str, args):
    print(f"Loading data for dataset: {dataset_str}")
    names = ['all_graphs', 'all_features', 'all_labels']
    objects = []
    for i in range(len(names)):
        try:
            with open("data/ind.{}.{}".format(dataset_str, names[i]), 'rb') as f:
                objects.append(pkl.load(f, encoding='latin1'))
        except FileNotFoundError:
            print(f"错误：文件 data/ind.{dataset_str}.{names[i]} 不存在！")
            sys.exit(1)
        except pkl.UnpicklingError:
            print(f"错误：文件 data/ind.{dataset_str}.{names[i]} 反序列化失败！")
            sys.exit(1)

    graphs, features, labels = tuple(objects)
    
    # 获取允许的边类型
    allowed_edge_types = set(args.edges.split(','))
    print(f'使用边类型{allowed_edge_types}')

    # 初始化训练集、验证集和测试集的邻接矩阵、特征、标签和图
    adjs = []
    embeds = []
    Gs = []  # 新增：存储过滤后的训练图

    def filter_graph_and_extract_adjacency(G, allowed_edge_types):
        """
        根据 allowed_edge_types 筛选边，返回过滤后的图和邻接矩阵。
        """
        # 创建新的图对象，保持原图的属性
        filtered_G = G.__class__()
        filtered_G.add_nodes_from(G.nodes(data=True))  # 添加所有节点（保持节点属性）
        
        num_nodes = G.number_of_nodes()
        adj = np.zeros((num_nodes, num_nodes))  # 初始化邻接矩阵

        # 遍历所有边
        for u, v, key, data in G.edges(keys=True, data=True):
            edge_type = data.get('edge_type')
            weight = data.get('weight', 1.0)  # 默认权重为 1.0

            # 如果边的类型在允许的范围内，则添加到新图和邻接矩阵
            if edge_type in allowed_edge_types:
                filtered_G.add_edge(u, v, key=key, **data)  # 添加边到新图（保持所有属性）
                adj[u][v] += weight  # 累加权重（适用于多重边）

        return filtered_G, adj

    # 加载训练集的图、特征和过滤后的图
    for i in range(len(labels)):
        graph = graphs[i]  # 加载 MultiGraph 对象
        filtered_graph, adj = filter_graph_and_extract_adjacency(graph, allowed_edge_types)
        adjs.append(adj)
        embeds.append(np.array(features[i]))
        Gs.append(filtered_graph)  # 存储过滤后的图


    adjs = np.array(adjs, dtype=object)
    embeds = np.array(embeds, dtype=object)
    Gs = np.array(Gs, dtype=object)
    y = np.array(labels)

    print("Data loading completed.")
    # 返回过滤后的图对象
    return adjs, embeds, y, Gs

# 稀疏矩阵转换为元组
def sparse_to_tuple(sparse_mx):
    """Convert sparse matrix to tuple representation."""
    def to_tuple(mx):
        if not sp.isspmatrix_coo(mx):
            mx = mx.tocoo()
        coords = np.vstack((mx.row, mx.col)).transpose()
        values = mx.data
        shape = mx.shape
        return coords, values, shape

    if isinstance(sparse_mx, list):
        for i in range(len(sparse_mx)):
            sparse_mx[i] = to_tuple(sparse_mx[i])
    else:
        sparse_mx = to_tuple(sparse_mx)

    return sparse_mx

# COO矩阵转换为元组
def coo_to_tuple(sparse_coo):
    return (sparse_coo.coords.T, sparse_coo.data, sparse_coo.shape)

# 预处理特征矩阵
def preprocess_features(features):
    print("Preprocessing features...")
    max_length = max([len(f) for f in features])
    # print('max_length of feature', max_length)

    # for i, feature in enumerate(features):
    #     print(f"Feature {i} shape: {np.shape(feature)}")
    
    for i in tqdm(range(len(features))):
        feature = np.array(features[i])
        pad = max_length - feature.shape[0]
        
        if pad < 0:
            print(f"Feature at index {i} is longer than the maximum length: {feature.shape[0]}")
        else:
            feature = np.pad(feature, ((0, pad), (0, 0)), mode='constant')
        
        # if feature.shape[0] != max_length or feature.shape[1] != 300:
        #     print(f"Feature at index {i} has unexpected shape: {feature.shape}")
        
        features[i] = feature

    print("Feature preprocessing completed.")
    # print('np.array(list(features))', np.array(list(features)))
    return np.array(list(features))

# 归一化邻接矩阵
def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    rowsum = np.array(adj.sum(1))
    with np.errstate(divide='ignore'):
        d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = np.diag(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt)

# 预处理邻接矩阵
def preprocess_adj(adj):
    print("Preprocessing adjacency matrices...")
    max_length = max([a.shape[0] for a in adj])
    # print('max_length of adj', max_length)
    mask = np.zeros((adj.shape[0], max_length, 1))

    for i in tqdm(range(adj.shape[0])):
        adj_normalized = normalize_adj(adj[i])
        pad = max_length - adj_normalized.shape[0]
        adj_normalized = np.pad(adj_normalized, ((0, pad), (0, pad)), mode='constant')
        mask[i, :adj[i].shape[0], :] = 1.
        adj[i] = adj_normalized

    print("Adjacency matrix preprocessing completed.")
    return np.array(list(adj)), mask

# 构建feed字典
def construct_feed_dict(features, support, mask, labels, placeholders):
    """Construct feed dictionary."""
    feed_dict = dict()
    feed_dict.update({placeholders['labels']: labels})
    feed_dict.update({placeholders['features']: features})
    feed_dict.update({placeholders['support']: support})
    feed_dict.update({placeholders['mask']: mask})
    feed_dict.update({placeholders['num_features_nonzero']: features[1].shape})
    return feed_dict

# 计算切比雪夫多项式
def chebyshev_polynomials(adj, k):
    """Calculate Chebyshev polynomials up to order k. Return a list of sparse matrices (tuple representation)."""
    print("Calculating Chebyshev polynomials up to order {}...".format(k))

    adj_normalized = normalize_adj(adj)
    laplacian = sp.eye(adj.shape[0]) - adj_normalized
    largest_eigval, _ = eigsh(laplacian, 1, which='LM')
    scaled_laplacian = (2. / largest_eigval[0]) * laplacian - sp.eye(adj.shape[0])

    t_k = list()
    t_k.append(sp.eye(adj.shape[0]))
    t_k.append(scaled_laplacian)

    def chebyshev_recurrence(t_k_minus_one, t_k_minus_two, scaled_lap):
        s_lap = sp.csr_matrix(scaled_lap, copy=True)
        return 2 * s_lap.dot(t_k_minus_one) - t_k_minus_two

    for i in range(2, k + 1):
        t_k.append(chebyshev_recurrence(t_k[-1], t_k[-2], scaled_laplacian))

    print("Chebyshev polynomials calculation completed.")
    return sparse_to_tuple(t_k)

# 加载Word2Vec
def loadWord2Vec(filename):
    """Read Word Vectors"""
    print(f"Loading Word2Vec from {filename}")
    vocab = []
    embd = []
    word_vector_map = {}
    with open(filename, 'r') as file:
        for line in file.readlines():
            row = line.strip().split(' ')
            if len(row) > 2:
                vocab.append(row[0])
                vector = row[1:]
                length = len(vector)
                for i in range(length):
                    vector[i] = float(vector[i])
                embd.append(vector)
                word_vector_map[row[0]] = vector
    print('Loaded Word Vectors!')
    return vocab, embd, word_vector_map

# 清理字符串
def clean_str(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()



