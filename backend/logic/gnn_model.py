import torch
import torch.nn.functional as F
from torch.nn import Linear
from torch_geometric.nn import SAGEConv, HeteroConv

def drop_edges(edge_index, p=0.0):
    if p <= 0.0:
        return edge_index
    E = edge_index.size(1)
    if E == 0:
        return edge_index
    keep_mask = torch.rand(E, device=edge_index.device) >= p
    if keep_mask.all():
        return edge_index
    return edge_index[:, keep_mask]

# Focal loss hyperparameters
# FOCAL_GAMMA = 2.0
# FOCAL_ALPHA = 0.25
# EDGE_DROPOUT_P = 0.10

# Architectural capacity
HIDDEN_DIM = 32
CONV_LAYERS = 2
DROPOUT = 0.3
EDGE_DROPOUT_P = 0.10
CHUNK_SIZE = 65536

# Optimization-stability
LR = 1e-3
WEIGHT_DECAY = 1e-5
EPOCHS = 30
PATIENCE = 6

# Data-dependent
P_POS_MULTIPLIER = 8
FOCAL_GAMMA = 1.5
FOCAL_ALPHA = 0.4

class BinaryFocalLoss(torch.nn.Module):
    def __init__(self, gamma=FOCAL_GAMMA, alpha=FOCAL_ALPHA, reduction='mean'):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction
        
    def forward(self, logits, targets):
        p = torch.sigmoid(logits)
        targets = targets.float()
        p_t = p * targets + (1 - p) * (1 - targets)
        alpha_factor = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        focal_weight = alpha_factor * (1.0 - p_t).pow(self.gamma)
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        loss = focal_weight * bce
        return loss.mean() if self.reduction == 'mean' else loss.sum()

class EdgeClassifier(torch.nn.Module):
    def __init__(self, tx_in, acc_in, hidden_dim, conv_layers, dropout, metadata):
        super().__init__()
        self.tx_lin = Linear(tx_in, hidden_dim)
        self.acc_lin = Linear(acc_in, hidden_dim)
        
        self.convs = torch.nn.ModuleList()
        self.bns = torch.nn.ModuleList()
        
        edge_types = metadata[1]
        for _ in range(conv_layers):
            conv_dict = {edge: SAGEConv((-1, -1), hidden_dim) for edge in edge_types}
            self.convs.append(HeteroConv(conv_dict, aggr='sum'))
            self.bns.append(torch.nn.BatchNorm1d(hidden_dim))
            
        # MLP for final classification
        self.edge_mlp = torch.nn.Sequential(
            Linear(2 * hidden_dim + tx_in, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            Linear(hidden_dim, 1)
        )
        self.edge_dropout_p = EDGE_DROPOUT_P

    def forward(self, x_dict, edge_index_dict, sender_indices=None, receiver_indices=None, tx_raw=None, apply_edge_dropout=True):
        # Fail-safe device detection
        device = self.tx_lin.weight.device
        
        # 1. Edge Dropout
        if apply_edge_dropout and self.edge_dropout_p > 0.0:
            edge_index_dict_mod = {k: drop_edges(ei.to(device), p=self.edge_dropout_p) for k, ei in edge_index_dict.items()}
        else:
            edge_index_dict_mod = {k: ei.to(device) for k, ei in edge_index_dict.items()}

        # 2. Base Embeddings
        x = {k: v.to(device) for k, v in x_dict.items()}
        x['account'] = self.acc_lin(x['account']).relu()
        x['transaction'] = self.tx_lin(x['transaction']).relu()
        
        # 3. Message Passing (GNN)
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index_dict_mod)
            x = {k: F.dropout(self.bns[i](v).relu(), p=DROPOUT, training=self.training) for k, v in x.items()}
        
        # 4. Final Prediction (MLP)
        # Required for GNNExplainer to see the end-to-end impact
        if sender_indices is not None and receiver_indices is not None and tx_raw is not None:
            h_sender = x['account'][sender_indices.to(device)]
            h_receiver = x['account'][receiver_indices.to(device)]
            tx_features = tx_raw.to(device)
            
            # Create Tri-Vector: [Sender_GNN, Receiver_GNN, Transaction_Tabular]
            edge_repr = torch.cat([h_sender, h_receiver, tx_features], dim=1)
            return self.edge_mlp(edge_repr).squeeze(1)
            
        return x # Fallback: returns account/transaction embeddings