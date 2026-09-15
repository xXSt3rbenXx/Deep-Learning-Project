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

    def __init__(self, in_features, out_features, T):
        super(GCNLayer, self).__init__()
        self.K = len(T) - 1
        for k, T_k in enumerate(T):
            self.register_buffer(f"T_{k}", T_k)          # ogni T_k si sposta su GPU correttamente
        self.linears = nn.ModuleList(
            [nn.Linear(in_features, out_features) for _ in range(self.K + 1)]  # un peso per ordine
        )

    def forward(self, x):
        out_list = []
        for k in range(self.K + 1):
            T_k = getattr(self, f"T_{k}")
            t_batched = T_k.unsqueeze(0).expand(x.size(0), -1, -1)   # (batch, N, N)
            out = torch.bmm(t_batched, x)
            out = self.linears[k](out)                                # peso specifico per k
            out_list.append(out)

        return torch.sum(torch.stack(out_list, dim=0), dim=0)         # somma SOLO sui K+1 ordini

class GCN(nn.Module):
  
    
    def __init__(self, input_dim,out_dim, hidden_dim,L,kernel_size=2,K=2, num_layers=3, dropout=0.5,):
        super(GCN, self).__init__()
        
        # Store model hyperparameters.
        self.num_layers = num_layers
        self.dropout = dropout
        self.K = K
        self.cheb=chebyshev_pol(L, K)
        
        
        self.gcn_layers = nn.ModuleList()
        
       
        self.gcn_layers.append(GCNLayer(input_dim, hidden_dim,K))
        self.gcn_layers.append(TCNLayer(hidden_dim, hidden_dim, kernel_size=kernel_size, blocks=3))
        
        
        for i in range(num_layers - 1):
            if i==num_layers-2:
                self.gcn_layers.append(GCNLayer(hidden_dim, hidden_dim,K))
                self.gcn_layers.append(TCNLayer(hidden_dim, out_dim, kernel_size=kernel_size, blocks=3))
            else:
                self.gcn_layers.append(GCNLayer(hidden_dim, hidden_dim,K))
                self.gcn_layers.append(TCNLayer(hidden_dim, hidden_dim, kernel_size=kernel_size, blocks=3))
        
        

     
    def forward(self, x, adj):
 
        
        # Apply all GCN layers sequentially.
        for i, gcn_layer in enumerate(self.gcn_layers):
            
           
            x = gcn_layer(x, adj)
            
           
            if i < len(self.gcn_layers) - 1:
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        

    
      
       
        
        
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





   