import torch 
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from Temporal_Convolutional_Network import TCNLayer


class GCNLayer(nn.Module):
    
    
    def __init__(self, in_features, out_features):
        super(GCNLayer, self).__init__()
        self.linear = nn.Linear(in_features, out_features)
        
    def forward(self, x, adj):
        
        x = self.linear(x)
        x = torch.bmm(adj, x)
        
        # Return the updated node representations.
        return x



class GCN(nn.Module):
  
    
    def __init__(self, input_dim, hidden_dim, output_dim,kernel_size, dilation, num_layers=3, dropout=0.5,):
        super(GCN, self).__init__()
        
        # Store model hyperparameters.
        self.num_layers = num_layers
        self.dropout = dropout
        
        
        self.gcn_layers = nn.ModuleList()
        
       
        self.gcn_layers.append(GCNLayer(input_dim, hidden_dim))
        self.gcn_layers.append(TCNLayer(hidden_dim, hidden_dim, kernel_size, dilation))
        
        for _ in range(num_layers - 2):
            self.gcn_layers.append(GCNLayer(hidden_dim, hidden_dim))
            self.gcn_layers.append(TCNLayer(hidden_dim, hidden_dim, kernel_size, dilation))
        
        
        if num_layers > 1:
            self.gcn_layers.append(GCNLayer(hidden_dim, hidden_dim))
            self.gcn_layers.append(TCNLayer(hidden_dim, hidden_dim, kernel_size, dilation))
        

        self.predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, output_dim)
        )
        
    def forward(self, x, adj, mask):
 
        
        # Apply all GCN layers sequentially.
        for i, gcn_layer in enumerate(self.gcn_layers):
            
           
            x = gcn_layer(x, adj)
            
           
            if i < len(self.gcn_layers) - 1:
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        

        mask_expanded = mask.unsqueeze(-1).expand_as(x)
       
        x_masked = x * mask_expanded
        
      
        graph_repr = x_masked.sum(dim=1) / mask.sum(dim=1, keepdim=True)
        
      
        output = self.predictor(graph_repr)
        
        
        return output



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
        
        return L_norm, X


    def cheb_polynomial(self, x, K):
        return 0.5*((x+np.sqrt(x**2-1))**K + (x-np.sqrt(x**2-1))**K)



   