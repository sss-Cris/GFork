import json
import sys
import torch
import argparse
from allennlp.common.util import import_module_and_submodules
from tqdm import tqdm
from allennlp.predictors import Predictor
from allennlp.models.archival import load_archive
import spacy
import random
import numpy as np
import itertools
from collections import Counter
import scipy.sparse as sp
import matplotlib.pyplot as plt
import networkx as nx
from typing import List, Dict
from transformers import BertTokenizer
import pickle as pkl
import os

from ppr import att_walk_multigraph


def save_to_file(data, file_path):
    """Save Python object to a file using pickle."""
    with open(file_path, 'wb') as f:
        pkl.dump(data, f)

def load_from_file(file_path):
    """Load Python object from a pickle file."""
    with open(file_path, 'rb') as f:
        return pkl.load(f)

def save_data(dataset, method_prefix, all_graphs, all_features, all_labels):
    """Save graphs, features, and labels to files."""
    file_template = "data/ind.{}.{}.{}"
    with open(file_template.format(dataset, method_prefix, 'all_graphs'), 'wb') as f:
        pkl.dump(all_graphs, f)
    with open(file_template.format(dataset, method_prefix, 'all_features'), 'wb') as f:
        pkl.dump(all_features, f)
    with open(file_template.format(dataset, method_prefix, 'all_labels'), 'wb') as f:
        pkl.dump(all_labels, f)

def predict_from_txt(model_path, doc_words_list):
    """
    Predict chunks and embeddings from a list of documents.
    Returns a list of dictionaries containing sentence info and predictions.
    """
    archive = load_archive(model_path)
    predictor = Predictor.from_archive(archive, 'chunk_predictor')
    nlp = spacy.load('en_core_web_trf', disable=['ner', 'textcat'])
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    sent_dict_list = []
    sent_id = 0

    for sent in tqdm(doc_words_list):
        sent_id += 1
        sentence = {}

        encoded_inputs = tokenizer(sent, padding=False, truncation=True, max_length=510, return_tensors="pt")
        input_ids = encoded_inputs["input_ids"][0]

        truncated_input_ids = input_ids[:510]
        truncated_sent = tokenizer.decode(truncated_input_ids, skip_special_tokens=True)
        sentence['sent'] = truncated_sent

        spacy_doc = nlp(truncated_sent)

        token_string_list, pos_list, dep_list, head_list, head_index_list = [], [], [], [], []
        for token in spacy_doc:
            token_string_list.append(token.orth_)
            pos_list.append(token.pos_)
            dep_list.append(token.dep_)
            head_list.append(token.head.orth_)
            head_index_list.append(token.head.i)

        sentence['spacy_pos'] = pos_list
        sentence['tokens'] = token_string_list
        sentence['sent_id'] = sent_id

        adj_dep_edges = [((head_index, child_index), dep_tag) 
                         for child_index, (head_index, dep_tag) in enumerate(zip(head_index_list, dep_list))]

        sentence['dep_graph_nodes'] = dep_list
        sentence['dep_graph_edges'] = adj_dep_edges

        result = predictor.predict_json(sentence)
        sentence['bounds'] = result['bound_tags']
        sentence['types'] = result['type_tags']
        sentence['word_embedding'] = result['word_embedding']

        sent_dict_list.append(sentence)

    return sent_dict_list

def plot_and_save_graph(adj, doc_words, save_path, file_name):
    """
    Plot a graph from adjacency matrix and save as an image.
    """
    graph = nx.from_scipy_sparse_array(adj)
    labels = {i: f"x{i}:{word}" for i, word in enumerate(doc_words)}
    pos = nx.spring_layout(graph, seed=42)

    plt.figure(figsize=(10, 8))
    nx.draw(
        graph,
        pos,
        with_labels=True,
        labels=labels,
        node_size=700,
        node_color="lightblue",
        font_size=10,
        font_color="black",
        font_weight="bold"
    )

    os.makedirs(save_path, exist_ok=True)
    plt.savefig(os.path.join(save_path, file_name), format="png", bbox_inches="tight")
    plt.close()

def get_coreference_clusters(doc_words):
    """
    Get coreference clusters for a list of words using a pre-trained model.
    Adjusts token indices to match the original document tokens.
    """
    text = ' '.join(doc_words)
    local_model_path = "/home/guest/workplace/gyf/textgraph/textGNN_config/trained_model/coreference/coref-spanbert-large-2021.03.10.tar.gz"

    try:
        coref_predictor = Predictor.from_path(local_model_path)
        coref_result = coref_predictor.predict(document=text)
        model_tokens = coref_result['document']
        original_tokens = doc_words

        if model_tokens == original_tokens:
            return coref_result['clusters']

        index_mapping = {}
        model_index = 0
        original_index = 0
        while model_index < len(model_tokens) and original_index < len(original_tokens):
            if model_tokens[model_index] == original_tokens[original_index]:
                index_mapping[model_index] = original_index
                model_index += 1
                original_index += 1
            else:
                combined_token = model_tokens[model_index]
                start_model_index = model_index
                model_index += 1
                while model_index < len(model_tokens) and combined_token != original_tokens[original_index]:
                    combined_token += model_tokens[model_index]
                    model_index += 1
                if combined_token == original_tokens[original_index]:
                    for i in range(start_model_index, model_index):
                        index_mapping[i] = original_index
                    original_index += 1
                else:
                    model_index = start_model_index + 1

        adjusted_clusters = []
        for cluster in coref_result['clusters']:
            adjusted_cluster = []
            for span in cluster:
                start, end = span
                if start in index_mapping and end in index_mapping:
                    adjusted_start = index_mapping[start]
                    adjusted_end = index_mapping[end]
                    adjusted_cluster.append((adjusted_start, adjusted_end))
            adjusted_clusters.append(adjusted_cluster)

        return adjusted_clusters
    except Exception as e:
        return []

def build_graph(method, label_list, edge_types, window_size, word_id_map, sent_dict_list, dataset, debug_sample=None):
    """
    Build graphs from sentences and edge types.
    Returns list of graphs, node features, labels, and document lengths.
    """
    graph_list, feature_list, y_list = [], [], []
    doc_len_list = []

    sample_range = [debug_sample] if debug_sample is not None else range(len(sent_dict_list))

    for i in tqdm(sample_range):
        doc_words = sent_dict_list[i]["tokens"]
        words_embedding = sent_dict_list[i]["word_embedding"]
        dep_graph_nodes = sent_dict_list[i]["dep_graph_nodes"]
        dep_graph_edges = sent_dict_list[i]["dep_graph_edges"]
        bounds = sent_dict_list[i]["bounds"]

        if len(doc_words) < 5 or len(doc_words) != len(words_embedding):
            continue

        doc_len = len(doc_words)
        doc_len_list.append(doc_len)

        edges_with_labels = build_edges(method, doc_words, words_embedding, dep_graph_edges, edge_types, window_size)

        G = nx.MultiGraph()
        for node in range(doc_len):
            G.add_node(node, word=doc_words[node], embedding=words_embedding[node])
        for (u, v), edge_type in edges_with_labels.items():
            G.add_edge(u, v, edge_type=edge_type)

        if method == "Chunk":
            G = convert_to_chunk_graph(G, bounds)
        else:
            G = convert_to_standard_graph(G)

        _ = att_walk_multigraph(G)

        final_embeddings = torch.stack([G.nodes[node]["embedding"] for node in G.nodes()], dim=0)

        graph_list.append(G)
        feature_list.append(final_embeddings)
        y_list.append(label_list[i])

    return graph_list, feature_list, y_list, doc_len_list

def convert_to_standard_graph(G):
    """Convert all edges in a graph to weight 1."""
    new_G = nx.MultiGraph()
    for node, data in G.nodes(data=True):
        new_G.add_node(node, **data)
    for u, v, key, data in G.edges(keys=True, data=True):
        new_G.add_edge(u, v, key=key, weight=1, **data)
    return new_G

def topk_pagerank(adj_ppr, alpha=0.85, k=4):
    """Compute top-k nodes by personalized PageRank from adjacency matrix."""
    G = nx.DiGraph()
    num_nodes = adj_ppr.shape[0]
    for i in range(num_nodes):
        for j in range(num_nodes):
            weight = adj_ppr[i, j].item()
            if weight > 0:
                G.add_edge(i, j, weight=weight)
    pagerank_values = nx.pagerank(G, alpha=alpha, weight='weight')
    top_k_nodes = sorted(pagerank_values.items(), key=lambda x: x[1], reverse=True)[:k]
    top_k_indices = [node for node, _ in top_k_nodes]
    return top_k_indices

def convert_to_chunk_graph(G, bounds):
    """Convert node-level graph to chunk-level graph using boundary info."""
    chunk_indices = []
    current_chunk = []
    for idx, is_boundary in enumerate(bounds):
        current_chunk.append(idx)
        if is_boundary:
            chunk_indices.append(current_chunk)
            current_chunk = []
    if current_chunk:
        chunk_indices.append(current_chunk)

    new_G = nx.MultiGraph()
    for chunk_idx, chunk in enumerate(chunk_indices):
        chunk_embedding = torch.mean(torch.stack([G.nodes[node]["embedding"] for node in chunk]), dim=0)
        chunk_word = ' '.join([G.nodes[node]["word"] for node in chunk])
        new_G.add_node(chunk_idx, word=chunk_word, embedding=chunk_embedding)

    edge_weight = {}
    for u, v, data in G.edges(data=True):
        chunk_u = next((idx for idx, ch in enumerate(chunk_indices) if u in ch))
        chunk_v = next((idx for idx, ch in enumerate(chunk_indices) if v in ch))
        edge_type = data["edge_type"]
        if chunk_u == chunk_v and edge_type == "self":
            key = (chunk_u, chunk_v, "self")
            edge_weight[key] = edge_weight.get(key, 0) + 1
        elif chunk_u != chunk_v:
            key = (chunk_u, chunk_v, edge_type)
            edge_weight[key] = edge_weight.get(key, 0) + 1

    for (chunk_u, chunk_v, edge_type), weight in edge_weight.items():
        new_G.add_edge(chunk_u, chunk_v, edge_type=edge_type, weight=weight)

    return new_G

def create_adjacency_matrix(edges, doc_len):
    """Create adjacency matrix from edge list."""
    edge_count = Counter(edges).items()
    row = [x for (x, y), c in edge_count]
    col = [y for (x, y), c in edge_count]
    weight = [c for (x, y), c in edge_count]
    return sp.csr_matrix((weight, (row, col)), shape=(doc_len, doc_len))

def build_edges(method, doc_words, words_embedding, dep_graph_edges, edge_types, window_size):
    """Build edges with types for a document."""
    doc_len = len(doc_words)
    edges_with_labels = {}

    if 'coreference' in edge_types:
        coref_clusters = get_coreference_clusters(doc_words)
        for cluster in coref_clusters:
            coref_nodes = {i for item in cluster for i in range(item[0], item[1] + 1)}
            for a, b in itertools.combinations(coref_nodes, 2):
                edges_with_labels[(a, b)] = "coreference"

    if 'window' in edge_types:
        windows = [(doc_words, list(range(doc_len)))] if doc_len <= window_size else [
            (doc_words[j:j+window_size], list(range(j, j+window_size))) for j in range(doc_len - window_size + 1)
        ]
        word_pair_count = {}
        for window_words, window_indices in windows:
            for p in range(1, len(window_words)):
                for q in range(p):
                    word_pair_key = tuple(sorted((window_indices[p], window_indices[q])))
                    word_pair_count[word_pair_key] = word_pair_count.get(word_pair_key, 0) + 1
        for key in word_pair_count:
            edges_with_labels[key] = "window"

    if 'syntax' in edge_types:
        for p, q in get_syntax_edges(dep_graph_edges):
            edges_with_labels[(p, q)] = "syntax"

    if 'same' in edge_types:
        for p, q in connect_same(doc_words):
            edges_with_labels[(p, q)] = "same"

    if 'self' in edge_types:
        for i in range(doc_len):
            edges_with_labels[(i, i)] = "self"

    return edges_with_labels

def get_syntax_edges(dep_graph_edges):
    """Extract syntax edges from dependency graph."""
    return [(edge[0][0], edge[0][1]) for edge in dep_graph_edges]

def connect_same(words):
    """Create edges between identical words."""
    edges = []
    for word in set(words):
        indices = [i for i, w in enumerate(words) if w == word]
        if len(indices) > 1:
            edges += list(itertools.permutations(indices, 2))
    return edges

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Build graph dataset.")
    parser.add_argument('--dataset', type=str, choices=['20ng','mr', 'ohsumed', 'R8', 'R52', 'TREC', 'ag_news', 'WebKB', 'SST1', 'SST2'])
    parser.add_argument('--method', type=str, default='original')
    parser.add_argument('--window-size', type=int, default=3)
    parser.add_argument('--weighted', action='store_true')
    parser.add_argument('--edges', type=str, default='coreference,window,same,syntax,self')
    parser.add_argument('--device', type=int, default=3)
    return parser.parse_args()

def load_documents(dataset):
    """Load document metadata from dataset file."""
    doc_name_list = []
    try:
        with open(f'datasets/data/{dataset}.txt', 'r', encoding='utf-8') as f:
            doc_name_list = [line.strip() for line in f]
    except FileNotFoundError:
        sys.exit(1)
    return doc_name_list

def load_corpus(dataset):
    """Load document text content from corpus file."""
    doc_content_list = []
    try:
        with open(f'datasets/data/corpus/{dataset}.clean.txt', 'r', encoding='utf-8') as f:
            doc_content_list = [line.strip() for line in f]
    except FileNotFoundError:
        sys.exit(1)
    return doc_content_list

def shuffle_documents(doc_name_list, doc_content_list):
    """Shuffle documents after filtering short ones (<10 characters)."""
    keep_indices = [i for i, content in enumerate(doc_content_list) if len(content) >= 10]
    doc_name_list = [doc_name_list[i] for i in keep_indices]
    doc_content_list = [doc_content_list[i] for i in keep_indices]
    ids = list(range(len(doc_name_list)))
    random.shuffle(ids)
    return [doc_name_list[i] for i in ids], [doc_content_list[i] for i in ids]

def create_onehot_labels(shuffle_doc_name_list):
    """Create one-hot encoded labels and list of classes."""
    label_set = set(doc_meta.split('\t')[2] for doc_meta in shuffle_doc_name_list)
    label_classes = list(label_set)
    y = []
    for doc_meta in shuffle_doc_name_list:
        one_hot = [0] * len(label_classes)
        label_index = label_classes.index(doc_meta.split('\t')[2])
        one_hot[label_index] = 1
        y.append(one_hot)
    return np.array(y), label_classes

def build_vocabulary(shuffle_doc_words_list):
    """Build vocabulary and word-to-id mapping."""
    word_set = set(word for doc in shuffle_doc_words_list for word in doc.split())
    vocab = list(word_set)
    word_id_map = {vocab[i]: i for i in range(len(vocab))}
    return vocab, word_id_map


if __name__ == '__main__':
    import_module_and_submodules("chunk")
    args = parse_arguments()
    dataset = args.dataset
    method = args.method
    window_size = args.window_size
    edge_types = args.edges.split(',')
    device = torch.device(f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu')
    cache_file = f"cache/{dataset}_cache_nosplit.pkl"

    if os.path.exists(cache_file):
        with open(cache_file, 'rb') as f:
            cached_data = pkl.load(f)
            label_list = cached_data['label_list']
            edge_types = cached_data['edge_types']
            window_size = cached_data['window_size']
            word_id_map = cached_data['word_id_map']
            sent_dict_list = cached_data['sent_dict_list']
    else:
        doc_name_list = load_documents(dataset)
        doc_content_list = load_corpus(dataset)
        shuffle_doc_name_list, shuffle_doc_words_list = shuffle_documents(doc_name_list, doc_content_list)
        label_list, label_classes = create_onehot_labels(shuffle_doc_name_list)
        model_path = "trained_model/2stage_chunker/model.tar.gz"
        sent_dict_list = predict_from_txt(model_path, shuffle_doc_words_list)
        vocab, word_id_map = build_vocabulary(shuffle_doc_words_list)
        cache_data = {
            'label_list': label_list,
            'edge_types': edge_types,
            'window_size': window_size,
            'word_id_map': word_id_map,
            'sent_dict_list': sent_dict_list,
        }
        save_to_file(cache_data, cache_file)

    method_prefix = 'Chunk' if method == 'Chunk' else 'standard'
    all_graphs, all_features, all_labels, doc_len_list = build_graph(
        method, label_list, edge_types, window_size, word_id_map, sent_dict_list, dataset
    )
    print('max_doc_length', max(doc_len_list), 'min_doc_length', min(doc_len_list), 'average {:.2f}'.format(np.mean(doc_len_list)))
