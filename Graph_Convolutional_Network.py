import torch 
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from Temporal_Convolutional_Network import TCNLayer

def chebyshev_pol(L, K):
    L_shape=L.shape[0]
    T_0= np.eye(L_shape, dtype=L.dtype)
    T_1=L
    T=[T_0, T_1]
    for i in range(2, K+1):
        T_i= 2*L @ T[i-1] - T[i-2]
        T.append(T_i)
    return T 

class GCNLayer(nn.Module):

    def __init__(self, in_features, out_features, K, dropout):
        super(GCNLayer, self).__init__()
        self.K = K
        self.linears = nn.ModuleList(
            [nn.Linear(in_features, out_features) for _ in range(K + 1)]  # un peso per ordine
        )
        self.dropout = dropout

    def forward(self, x, T):
        out_list = []
        for k, T_k in enumerate(T):
            t_batched = T_k.unsqueeze(0).expand(x.size(0), -1, -1)   # (batch, N, N)
            out = torch.bmm(t_batched, x)
            out = self.linears[k](out)                                # peso specifico per k
            out_list.append(out)
        out_sum=torch.sum(torch.stack(out_list, dim=0), dim=0)         # somma SOLO sui K+1 ordini
        out_sum=F.relu(out_sum)
        return  F.dropout(out_sum, p=self.dropout, training=self.training)


class GCN(nn.Module):
  
    
    def __init__(self, input_dim,out_dim, hidden_dim,T_list, kernel_size=2, num_layers=3, dropout=0.5,):
        super(GCN, self).__init__()
        
        # Store model hyperparameters.
        self.num_layers = num_layers
        self.dropout = dropout
        self.K = len(T_list)-1
        for k, T_k in enumerate(T_list):
            self.register_buffer(f"T_{k}", T_k)

        layers=[]
        for i in range(num_layers):
            if i== 0:
                layers.append(GCNLayer(input_dim, hidden_dim, self.K, dropout))
            elif i==num_layers-1:
                layers.append(GCNLayer(hidden_dim, hidden_dim, self.K, dropout))
                
            else:
                layers.append(GCNLayer(hidden_dim, hidden_dim, self.K, dropout))

        self.gcn_layers= nn.ModuleList(
            [layers[i] for i in range(num_layers)]
        )
        layers=[]
        for i in range(num_layers):
            if i== num_layers-1:
                layers.append(TCNLayer(hidden_dim, out_dim, num_layers, kernel_size))
            else:
                layers.append(TCNLayer(hidden_dim, hidden_dim, num_layers, kernel_size))
                

        self.tcn_layers= nn.ModuleList(
            [layers[i] for i in range(num_layers)]
        )
        
        
        
  
        
        

     
    def forward(self, x):
        # x: (B, N, T, F)
        B, N, T, _ = x.shape
        T_list = [getattr(self, f"T_{k}") for k in range(self.K + 1)]

        for i in range(self.num_layers):
        
            x_gcn_in = x.permute(0, 2, 1, 3).reshape(B * T, N, -1)   # (B*T, N, F)
            x_gcn_out = self.gcn_layers[i](x_gcn_in, T_list)          # (B*T, N, hidden_dim)
            F_hidden = x_gcn_out.shape[-1]
            x = x_gcn_out.reshape(B, T, N, F_hidden).permute(0, 2, 1, 3)  # torna a (B, N, T, hidden_dim)

            # --- TCN: fondi batch e N ---
            x_tcn_in = x.reshape(B * N, T, F_hidden).permute(0, 2, 1)     # (B*N, F, T)
            x_tcn_out = self.tcn_layers[i](x_tcn_in)                       # (B*N, F', T)
            F_out = x_tcn_out.shape[1]
            x = x_tcn_out.permute(0, 2, 1).reshape(B, N, T, F_out)          # torna a (B, N, T, F')

        return x  



    def graph_to_matrices(self, node_features, adj_matrix):
        
 
        n_nodes = adj_matrix.shape[0]
        A_tilde = adj_matrix.astype(np.float32)
        #La matrice non è simmetrica, va resa tale sacrificando informazioni sulla direzionalità
        A_sim= (A_tilde+ A_tilde.T) / 2
        D = np.diag(np.sum(A_sim, axis=1))
        D_inv_sqrt = np.diag(np.power(np.diag(D), -0.5))
        
    
        A_norm = D_inv_sqrt @ A_sim @ D_inv_sqrt
        L=np.eye(n_nodes)-A_norm
        eigenvalues= np.linalg.eigvalsh(L)
        eigenvalues.sort()
        max_eig=eigenvalues[-1]
        L_norm= 2*L/max_eig - np.eye(n_nodes)

        
        X = node_features.astype(np.float32)
        
        return torch.Tensor(L_norm), X





   