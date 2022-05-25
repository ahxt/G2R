
import numpy as np
from torch_geometric.utils import to_dense_adj
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import add_remaining_self_loops
from torch_scatter import scatter_add
from torch_geometric.datasets import Planetoid, CitationFull
from torch_geometric.datasets import Planetoid
from torch_geometric.datasets import Coauthor, Amazon
import torch_geometric.transforms as T
from torch.nn import Parameter
import torch.nn.functional as F
import torch.nn as nn
import torch
import math
import argparse
import os
import logging

from eval import label_classification
from model import Model, GCNConv, Encoder
from utils import random_coauthor_amazon_splits, random_planetoid_splits, normalize_adj_row, normalize_adj



class MaximalCodingRateReduction(torch.nn.Module):
    ## This function is based on https://github.com/ryanchankh/mcr2/blob/master/loss.py
    def __init__(self, gam1=1.0, gam2=1.0, eps=0.01):
        super(MaximalCodingRateReduction, self).__init__()
        self.gam1 = gam1
        self.gam2 = gam2
        self.eps = eps

    def projection(self, z: torch.Tensor) -> torch.Tensor:
        z = F.normalize(z, dim=0)
        return z

    def compute_discrimn_loss_empirical(self, W):
        """Empirical Discriminative Loss."""
        p, m = W.shape
        I = torch.eye(p).cuda()
        scalar = p / (m * self.eps)
        logdet = torch.logdet(I + self.gam1 * scalar * W.matmul(W.T))
        return logdet / 2.

    def compute_compress_loss_empirical_all(self, W, Pi):
        """Empirical Compressive Loss."""
        p, m = W.shape
        k, _ = Pi.shape
        sum_trPi = torch.sum(Pi)

        I = torch.eye(p).cuda()
        compress_loss = 0.
        for j in range(k):
            trPi = torch.sum(Pi[j]) + 1e-8
            scalar = p / (trPi * self.eps)
            a = W.T * Pi[j].view(-1, 1)
            a = a.T
            log_det = torch.logdet(I + scalar * a.matmul(W.T))
            compress_loss += log_det * trPi / m
        num = data.x.shape[0]
        compress_loss = compress_loss / 2 * (num / sum_trPi)
        return compress_loss

    def forward(self, X, A):
        i = np.random.randint(A.shape[0], size=args.num_node_batch)
        A = A[i,::]
        A = A.cpu().numpy()
        W = X.T
        Pi = A
        Pi = torch.tensor(Pi, dtype=torch.float32).cuda()

        discrimn_loss_empi = self.compute_discrimn_loss_empirical(W)
        compress_loss_empi = self.compute_compress_loss_empirical_all(W, Pi)
        total_loss_empi = - self.gam2 * discrimn_loss_empi + compress_loss_empi
        return total_loss_empi



def train(model: Model, x, edge_index, A, MaximalCodingRateReduction: MaximalCodingRateReduction):
    model.train()
    optimizer.zero_grad()
    z = model(x, edge_index)
    loss = MaximalCodingRateReduction(z, A)
    loss.backward()
    optimizer.step()
    return loss.item()


def test(model: Model, x, edge_index, y, train_mask=None, test_mask=None):
    model.eval()
    z = model(x, edge_index)
    x = z.detach().cpu().numpy()
    res = label_classification(z, y, train_mask=train_mask, test_mask=test_mask)
    return res






if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='Cora')
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--num_hidden', type=int, default=512)
    parser.add_argument('--num_out', type=int, default=512)
    parser.add_argument('--gam1', type=float, default=0.5)
    parser.add_argument('--gam2', type=float, default=0.5)
    parser.add_argument('--eps', type=float, default=0.05)
    parser.add_argument('--activation', type=str, default="relu")
    parser.add_argument('--num_epochs', type=int, default=40)
    parser.add_argument('--num_layers', type=int, default=1)
    parser.add_argument('--num_node_batch', type=int, default=100)
    parser.add_argument('--weight_decay', type=float, default=0.0001)
    parser.add_argument('--input_dir', type=str, default="../data")
    parser.add_argument('--seed', type=int, default=31415)
    parser.add_argument('--round', type=int, default=0)
    parser.add_argument('--exp_name', type=str, default="G2R")
    parser.add_argument('--split', type=str, default="random")
    args = parser.parse_args()

    print("Args:{}".format(args))

    seed = args.seed
    learning_rate = args.learning_rate
    num_hidden = args.num_hidden
    num_out = args.num_out
    activation = ({'relu': F.relu, 'prelu': nn.PReLU()})[args.activation]
    base_model = GCNConv
    num_epochs = args.num_epochs
    weight_decay = args.weight_decay

    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.set_num_threads(3)


    path = os.path.join(args.input_dir)

    if args.dataset == "Cora":
        dataset = Planetoid(path, args.dataset, transform=T.NormalizeFeatures())
        data = dataset[0]
        data.num_classes = dataset.num_classes
        if args.split == "random":
            data = random_planetoid_splits(data, dataset.num_classes, lcc_mask=None)
    
    elif args.dataset == "CiteSeer":
        dataset = Planetoid(path, args.dataset, transform=T.NormalizeFeatures())
        # dataset = Planetoid(path, args.dataset)
        data = dataset[0]
        data.num_classes = dataset.num_classes
        if args.split == "random":
            data = random_planetoid_splits(data, dataset.num_classes, lcc_mask=None)
    
    elif args.dataset == "PubMed":
        dataset = Planetoid(path, args.dataset, transform=T.NormalizeFeatures())
        # dataset = Planetoid(path, args.dataset)
        data = dataset[0]
        data.num_classes = dataset.num_classes
        if args.split == "random":
            data = random_planetoid_splits(data, dataset.num_classes, lcc_mask=None)
    
    elif args.dataset == "CoraFull":
        dataset = CitationFull(path, "cora")
        data = dataset[0]
        data.num_classes = dataset.num_classes
        data = random_coauthor_amazon_splits(data, dataset.num_classes, lcc_mask=None)
    
    elif args.dataset == "Photo":
        dataset = Amazon(path, args.dataset, transform=T.NormalizeFeatures())
        data = dataset[0]
        data.num_classes = dataset.num_classes
        data = random_coauthor_amazon_splits(data, dataset.num_classes, lcc_mask=None)
    
    elif args.dataset == "Computers":
        dataset = Amazon(path, args.dataset, transform=T.NormalizeFeatures())
        # dataset = Amazon(path, args.dataset)
        data = dataset[0]
        data.num_classes = dataset.num_classes
        data = random_coauthor_amazon_splits(data, dataset.num_classes, lcc_mask=None)
    
    elif args.dataset == "CS":
        dataset = Coauthor(path, args.dataset, transform=T.NormalizeFeatures())
        data = dataset[0]
        data.num_classes = dataset.num_classes
        data = random_coauthor_amazon_splits(data, dataset.num_classes, lcc_mask=None)
    
    elif args.dataset == "Physics":
        dataset = Coauthor(path, args.dataset, transform=T.NormalizeFeatures())
        data = dataset[0]
        data.num_classes = dataset.num_classes
        data = random_coauthor_amazon_splits(data, dataset.num_classes, lcc_mask=None)
    else:
        print("Input dataset name!!")
        raise NotImplementedError
    
    print("Dataset:", args.dataset)
    print("Number of Nodes:", data.x.shape[0])
    print("Number of Nodes Features:", data.x.shape[1])
    print("Number of Edges:", data.edge_index.shape[1])


    node_num = data.x.shape[0]
    num_features = data.x.shape[1]

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data = data.to(device)

    A = to_dense_adj(data.edge_index)[0].cpu()
    A = normalize_adj_row(A)
    A = torch.from_numpy(A.todense())


    encoder = Encoder(in_channels=num_features,out_channels=num_out, hidden_channels=num_hidden, activation=activation,base_model=base_model, k=args.num_layers).to(device)
    model = Model(encoder=encoder).to(device)
    coding_rate_loss = MaximalCodingRateReduction(gam1=args.gam1, gam2=args.gam2, eps=args.eps).to(device)
    optimizer = torch.optim.Adam( list(model.parameters()) + list(coding_rate_loss.parameters()), lr=learning_rate, weight_decay=weight_decay)


    for epoch in range(1, num_epochs + 1):
        loss = train(model, data.x, data.edge_index, A, coding_rate_loss)
        val_res = test(model, data.x, data.edge_index, data.y, train_mask=data.train_mask.cpu().numpy(), test_mask=data.val_mask.cpu().numpy())
        test_res = test(model, data.x, data.edge_index, data.y, train_mask=data.train_mask.cpu().numpy(), test_mask=data.test_mask.cpu().numpy())
        print("Epoch: {:03d}, val_acc: {:.4f}, test_acc:{:.4f}".format(epoch, val_res["acc"], test_res["acc"] ))