import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter
from transformers import BertModel, BertTokenizer
from layers import *


class G2(nn.Module):
    """Gradient gating module (memory-efficient)."""
    def __init__(self, convert, g2_p=2., activation=nn.ReLU()):
        super().__init__()
        self.convert = convert
        self.g2_p = g2_p
        self.activation = activation

    def forward(self, features, support, mask):
        """
        Args:
            features: node features [B, N, D]
            support: adjacency matrix [B, N, N]
            mask: node mask [B, N, 1]
        Returns:
            tau: gating coefficients [B, N, 1]
        """
        B, N, D = features.shape
        device = features.device

        X = self.activation(self.convert(features))  # transform features

        adj = (support > 0).float()  # binary adjacency
        mask = mask.squeeze(-1)
        valid_mask = mask.unsqueeze(1) * mask.unsqueeze(2)
        adj = adj * valid_mask

        diff_sum = torch.zeros(B, N, device=device)
        neighbor_count = torch.zeros(B, N, device=device)

        for b in range(B):
            idx_i, idx_j = adj[b].nonzero(as_tuple=True)
            xi = X[b, idx_i]
            xj = X[b, idx_j]

            diff = (xi - xj).abs().pow(self.g2_p).sum(dim=-1)

            diff_sum_b = torch.zeros(N, device=device).index_add_(0, idx_i, diff)
            count_b = torch.zeros(N, device=device).index_add_(0, idx_i, torch.ones_like(diff))

            diff_sum[b] = diff_sum_b
            neighbor_count[b] = count_b.clamp(min=1)

        tau = torch.tanh(diff_sum / neighbor_count).unsqueeze(-1)
        return tau


class BERTGNNWithG2(nn.Module):
    """BERT-GNN with optional G2 fusion."""
    def __init__(self, args, bert_model_name, output_dim, hidden_dim, gru_step, dropout_p, use_g2=True, g2_p=2):
        super().__init__()
        self.args = args
        self.hidden_dim = hidden_dim
        self.dropout_p = dropout_p

        self.bert = BertModel.from_pretrained('./trained_model/bert-base-uncased')
        self.bert.to(args.device)
        self.tokenizer = BertTokenizer.from_pretrained('./trained_model/bert-base-uncased')

        # freeze all BERT layers except last two
        for name, param in self.bert.named_parameters():
            if name.startswith("encoder.layer."):
                layer_num = int(name.split('.')[2])
                param.requires_grad = layer_num >= 10
            else:
                param.requires_grad = False

        self.linear = nn.Linear(self.bert.config.hidden_size, hidden_dim)
        self.act = nn.Tanh()
        self.dropout = nn.Dropout(dropout_p)

        self.graph_layer = GraphLayer(
            args=args,
            input_dim=hidden_dim,
            output_dim=hidden_dim,
            act=nn.Tanh(),
            dropout_p=dropout_p,
            gru_step=gru_step
        )

        self.readout = ReadoutLayer(
            args=args,
            input_dim=hidden_dim,
            output_dim=output_dim,
            dropout_p=dropout_p
        )

        self.use_g2 = use_g2
        if use_g2:
            self.G2 = G2(convert=self.linear, g2_p=g2_p, activation=nn.ReLU())

    def process_graph_to_bert_input(self, graph):
        """Convert graph nodes to BERT input and align subwords."""
        nodes = sorted(graph.nodes(data=True), key=lambda x: x[0])
        words = [node[1]['word'] for node in nodes]
        full_text = ' '.join(words)

        inputs = self.tokenizer(full_text, return_tensors='pt', truncation=True)
        inputs = {k: v.to(self.args.device) for k, v in inputs.items()}

        # map words to subword token positions
        chunk_to_subwords = []
        current_pos = 0
        for word in words:
            word_tokens = self.tokenizer.tokenize(word)
            chunk_to_subwords.append((current_pos, current_pos + len(word_tokens)))
            current_pos += len(word_tokens)

        return inputs, chunk_to_subwords

    def forward(self, graphs, support, mask):
        batch_size = len(graphs)
        max_nodes = mask.shape[1]

        all_embeddings = []

        for graph in graphs:
            bert_inputs, chunk_to_subwords = self.process_graph_to_bert_input(graph)
            bert_outputs = self.bert(**bert_inputs)
            subword_embeddings = bert_outputs.last_hidden_state[0]

            # aggregate subword embeddings
            chunk_embeddings = []
            for start, end in chunk_to_subwords:
                chunk_embeddings.append(subword_embeddings[start:end].mean(dim=0))
            all_embeddings.append(torch.stack(chunk_embeddings))

        # pad to max_nodes
        features = torch.zeros(batch_size, max_nodes, self.bert.config.hidden_size, device=self.bert.device)
        for i, emb in enumerate(all_embeddings):
            features[i, :len(emb)] = emb

        # project to hidden_dim
        linear_out = self.linear(features)
        linear_out = self.act(linear_out)
        linear_out = self.dropout(linear_out)

        graph_out = self.graph_layer(linear_out, support, mask)

        # apply gradient gating
        if self.use_g2:
            tau = self.G2(features, support, mask)
            if features.size(-1) != graph_out.size(-1):
                features = self.linear(features)
            graph_out = (1 - tau) * features + tau * graph_out

        return self.readout(graph_out, support, mask), graph_out
