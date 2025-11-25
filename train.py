import os
import time
import pickle
from argparse import ArgumentParser

from config import Config
import numpy as np
import torch
import torch.nn as nn
from sklearn import metrics
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from distutils.util import strtobool
from sklearn.model_selection import StratifiedKFold

from metrics import softmax_cross_entropy, accuracy
from models import BERTGNNWithG2
from utils import set_seed, setup_logger, load_data, preprocess_adj, preprocess_features, to_tensor_func

# Set random seed
seed = 123
set_seed(seed)

def parse_arguments():
    config = Config()

    parser = ArgumentParser()
    parser.add_argument('--method', type=str, default='original')
    parser.add_argument('--edges', type=str, default='coreference,window,same,syntax,self')
    parser.add_argument('--logger_name', type=str, default='TextING-pytorch')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints')
    parser.add_argument('--dataset', type=str, default='R8')
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--model', type=str, default='gru')
    parser.add_argument('--learning_rate', type=float, default=0.005)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--weight_decay', type=float, default=0.0)
    parser.add_argument('--shuffle', type=lambda x: bool(strtobool(x)), default=False)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--steps', type=int, default=2, help='Number of graph layers.')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--input_dim', type=int, default=768)
    parser.add_argument('--hidden_dim', type=int, default=96)
    parser.add_argument('--l1_lambda', type=float, default=0.001)
    parser.add_argument('--early_stopping', type=int, default=-1)
    parser.add_argument('--max_degree', type=int, default=3)
    parser.add_argument('--g2_p', type=float, default=2.0)
    parser.add_argument('--use_g2', type=lambda x: bool(strtobool(x)), default=True)
    
    args = parser.parse_args()

    # 根据命令行中的 dataset 参数获取对应的配置
    if args.dataset == '20ng':
        dataset_config = getattr(config, 'ng20', {})
    else:
        dataset_config = getattr(config, args.dataset, {})

    # 如果 dataset_config 存在，从中读取超参数
    if dataset_config:
        lr = dataset_config.get('lr', 0.001)
        dropout = dataset_config.get('dropout', 0.3)
        weight_decay = dataset_config.get('weight_decay', 0.0)
        hidden_dim = dataset_config.get('hidden_dim', 32)
        l1_lambda = dataset_config.get('l1_lambda', 0.0)

        # 更新命令行参数的默认值
        args.learning_rate = lr
        args.dropout = dropout
        args.weight_decay = weight_decay
        args.hidden_dim = hidden_dim
        args.l1_lambda = l1_lambda

    return args

def initialize_device(args):
    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    return device

def load_and_preprocess_data(args):

    print("Loading and preprocessing data...")
    dataset = args.dataset + ('.Chunk' if args.method == 'Chunk' else '.standard')
    adj, feature, y, G = load_data(dataset, args)
    adj, mask = preprocess_adj(adj)

    adj = to_tensor_func(adj, args.device)
    mask = to_tensor_func(mask, args.device)
    y = to_tensor_func(y, args.device)

    print("Data preprocessing completed and cached.")
    return adj, mask, feature, y, G

def build_model(args, output_dim):
    model = BERTGNNWithG2(
        args=args,
        bert_model_name=getattr(args, 'bert_model', 'bert-base-uncased'),
        output_dim=output_dim,
        hidden_dim=args.hidden_dim,
        gru_step=args.steps,
        dropout_p=args.dropout,
        use_g2=args.use_g2,
        g2_p=args.g2_p
    ).to(args.device)
    return model

def evaluate(model, graphs, support, mask, labels):
    model.eval()
    with torch.no_grad():
        outputs, embeddings = model(graphs, support, mask)
        cost = softmax_cross_entropy(nn.CrossEntropyLoss(), outputs, labels)
        acc = accuracy(outputs, labels)
        pred = torch.argmax(outputs, 1)
        labels = torch.argmax(labels, 1)
    return cost, acc, 0, embeddings, pred, labels

def train_and_evaluate(args, model, optimizer, train_data, val_data, test_data):
    train_adj, train_mask, _, train_y, train_G = train_data
    val_adj, val_mask, _, val_y, val_G = val_data
    test_adj, test_mask, _, test_y, test_G = test_data

    log_dir = os.path.join(args.checkpoint_dir, 'tensorboard_logs', f"{args.dataset}_{int(time.time())}")
    writer = SummaryWriter(log_dir=log_dir)
    print(f"[TensorBoard] Writing logs to: {log_dir}")


    best_val, best_acc, best_epoch = 0, 0, 0
    cost_val = []
    logger = setup_logger(args.logger_name, os.path.join(args.checkpoint_dir, 'train.log'))

    for epoch in range(args.epochs):
        t = time.time()
        indices = np.random.permutation(len(train_y))
        model.train()
        train_loss, train_acc = 0, 0

        for start in range(0, len(train_y), args.batch_size):
            end = start + args.batch_size
            idx = indices[start:end]
            batch_graphs = [train_G[i] for i in idx]
            batch_adj = train_adj[idx]
            batch_mask = train_mask[idx]
            batch_y = train_y[idx]

            
            outputs, _ = model(batch_graphs, batch_adj, batch_mask)
            loss = softmax_cross_entropy(nn.CrossEntropyLoss(), outputs, batch_y)

            if args.l1_lambda > 0:
                l1_reg = sum(torch.norm(param, p=1) for param in model.parameters())
                loss += args.l1_lambda * l1_reg

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            acc = accuracy(outputs, batch_y)
            train_loss += loss.item() * len(idx)
            train_acc += acc.item() * len(idx)

        train_loss /= len(train_y)
        train_acc /= len(train_y)

        val_cost, val_acc, _, _, _, _ = evaluate(model, val_G, val_adj, val_mask, val_y)
        test_cost, test_acc, _, _, _, _ = evaluate(model, test_G, test_adj, test_mask, test_y)

        cost_val.append(val_cost.item())
        print(f"\nEpoch: {epoch+1:04d} | Train Loss: {train_loss:.5f} | Train Acc: {train_acc:.5f} | "
                      f"Val Loss: {val_cost:.5f} | Val Acc: {val_acc:.5f} | "
                      f"Test Acc: {test_acc:.5f} | Time: {time.time()-t:.2f}s")

        if val_acc >= best_val:
            best_val = val_acc
            best_epoch = epoch
            best_acc = test_acc

        if args.early_stopping > 0 and epoch > args.early_stopping and val_cost > np.mean(cost_val[-(args.early_stopping+1):-1]):
            logger.info("Early stopping triggered")
            break

    writer.close()
    logger.info(f"Best epoch: {best_epoch} | Test Accuracy: {best_acc:.4f}")
    return best_val.item(), best_acc.item(), best_epoch

def main():
    args = parse_arguments()
    args.device = initialize_device(args)
    # print(args.dataset)
    
    all_adj, all_mask, all_feature, all_y, all_G = load_and_preprocess_data(args)

    all_adj = all_adj.cpu()
    all_mask = all_mask.cpu()
    # all_feature = all_feature.cpu()
    if len(all_y.shape) > 1 and all_y.shape[1] > 1:  # 判断是否是多类（独热编码）
        all_y_cpu = torch.argmax(all_y, dim=1).cpu().numpy()  # 选择最大值所在的索引作为类别标签
    kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
    all_acc, fold_metrics = [], []

    for fold, (train_idx, test_idx) in enumerate(kf.split(all_feature, all_y_cpu), 1):
        print(f"\n==== Fold {fold} ====")

        train_adj, test_adj = all_adj[train_idx], all_adj[test_idx]
        train_mask, test_mask = all_mask[train_idx], all_mask[test_idx]
        train_y, test_y = all_y[train_idx], all_y[test_idx]
        train_G = [all_G[i] for i in train_idx]
        test_G = [all_G[i] for i in test_idx]

        perm = np.random.permutation(len(train_y))
        val_split = int(len(train_y) * 0.1)
        val_adj, val_mask, val_y = train_adj[perm[:val_split]], train_mask[perm[:val_split]], train_y[perm[:val_split]]
        val_G = [train_G[i] for i in perm[:val_split]]
        train_adj, train_mask, train_y = train_adj[perm[val_split:]], train_mask[perm[val_split:]], train_y[perm[val_split:]]
        train_G = [train_G[i] for i in perm[val_split:]]

        train_adj = train_adj.to(args.device)
        train_mask = train_mask.to(args.device)
        val_adj = val_adj.to(args.device)
        val_mask = val_mask.to(args.device)
        test_adj = test_adj.to(args.device)
        test_mask = test_mask.to(args.device)

        model = build_model(args, output_dim=train_y.shape[1])
        optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

        best_val, best_acc, best_epoch = train_and_evaluate(
            args,
            model,
            optimizer,
            (train_adj, train_mask, all_feature, train_y, train_G),
            (val_adj, val_mask, all_feature, val_y, val_G),
            (test_adj, test_mask, all_feature, test_y, test_G)
        )

        fold_metrics.append({'fold': fold, 'best_val_acc': best_val, 'test_acc_at_best_val': best_acc, 'best_epoch': best_epoch})
        all_acc.append(best_acc)

    print("\n==== Cross Validation Results ====")
    for fm in fold_metrics:
        print(f"Fold {fm['fold']:02d}: Best Val Acc = {fm['best_val_acc']:.4f} | Test Acc = {fm['test_acc_at_best_val']:.4f} | Epoch = {fm['best_epoch']}")

    print(f"\nAverage Test Accuracy: {np.mean(all_acc.cpu().numpy()):.4f} ± {np.std(all_acc):.4f}")

if __name__ == '__main__':
    main()