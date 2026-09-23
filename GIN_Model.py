import torch
import torch.nn as nn
import torch.nn.functional as F
from Temporal_Convolutional_Network import TCNLayer


class GIN_Layer(nn.Module):

    def __init__(self, in_features, out_features, eps, dropout):
        super(GIN_Layer, self).__init__()
        self.eps=eps
        self.mlp=nn.Sequential(
            nn.Linear(in_features, out_features),
            nn.ReLU(),
            nn.Linear(out_features, out_features)
        )
        self.dropout = dropout

    def forward(self, x, A_norm):
        #A è la matrice di adiacenza con i self_loops
        A=A_norm.unsqueeze(0).expand(x.size(0),-1,-1)
        neighbor_Agg=torch.bmm(A, x)
        #formula_GIN
        out=self.mlp((1.0+self.eps)*x+neighbor_Agg)
        out=F.relu(out)
        return F.dropout(out, p=self.dropout, training=self.training)


class GIN(nn.Module):
    def __init__(self, input_dim, out_dim, hidden_dim, eps, adj_matrix, kernel_size=3, num_layers=3, num_blocks=3, dropout=0.3):
        super(GIN, self).__init__()

        self.num_layers = num_layers
        self.dropout = dropout
        self.eps = eps

        # Normalizzazione dell'adiacenza per GIN: A_norm = A + I
        N = adj_matrix.shape[0]
        A_tilde = adj_matrix + torch.eye(N, device=adj_matrix.device)
        D_inv = torch.diag(1.0 / (torch.sum(A_tilde, dim=1) + 1e-8))
        A_norm = D_inv @ A_tilde
        self.register_buffer("A_norm", A_norm)

        # Layer GIN
        gin_layers = []
        for i in range(num_layers):
            in_f = input_dim if i == 0 else hidden_dim
            gin_layers.append(GIN_Layer(in_f, hidden_dim, dropout, eps=eps))
        self.gin_layers = nn.ModuleList(gin_layers)

        # Strati TCN Temporali (Identici a GCN/Temporal_Model)
        tcn_layers = []
        for i in range(num_layers):
            out_f = out_dim if i == num_layers - 1 else hidden_dim
            tcn_layers.append(TCNLayer(hidden_dim, out_f, num_blocks, kernel_size))
        self.tcn_layers = nn.ModuleList(tcn_layers)

        # Teste Quantiliche
        self.linear1 = nn.Linear(out_dim, 3)
        self.linear2 = nn.Linear(out_dim, 3)
        self.linear3 = nn.Linear(out_dim, 3)

    def forward(self, x):
        # x: (B, T, N, F)
        B, T, N, _ = x.shape

        for i in range(self.num_layers):

            x_gin_in = x.reshape(B * T, N, -1)
            x_gin_out = self.gin_layers[i](x_gin_in, self.A_hat)
            F_hidden = x_gin_out.shape[-1]
            x = x_gin_out.reshape(B, T, N, F_hidden).permute(0, 2, 1, 3)

            x_tcn_in = x.reshape(B * N, T, F_hidden).permute(0, 2, 1)
            x_tcn_out = self.tcn_layers[i](x_tcn_in)
            F_out = x_tcn_out.shape[1]
            x = x_tcn_out.reshape(B, N, F_out, T).permute(0, 3, 1, 2)

        out = x[:, -1, :, :]

        median = self.linear1(out)
        low_quantile = F.softplus(self.linear2(out))
        high_quantile = F.softplus(self.linear3(out))

        return median - low_quantile, median, median + high_quantile








