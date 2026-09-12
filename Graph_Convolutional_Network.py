import torch 
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


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
  
    
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=3, dropout=0.5):
        super(GCN, self).__init__()
        
        # Store model hyperparameters.
        self.num_layers = num_layers
        self.dropout = dropout
        
        
        self.gcn_layers = nn.ModuleList()
        
       
        self.gcn_layers.append(GCNLayer(input_dim, hidden_dim))
        
        for _ in range(num_layers - 2):
            self.gcn_layers.append(GCNLayer(hidden_dim, hidden_dim))
        
        
        if num_layers > 1:
            self.gcn_layers.append(GCNLayer(hidden_dim, hidden_dim))
        

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



    def graph_to_matrices(self,graph, node_features, adj_matrix):
        
 
        n_nodes = len(graph.nodes())
        A_tilde = adj_matrix.astype(np.float32) + np.eye(n_nodes)
        D = np.diag(np.sum(A_tilde, axis=1))
        D_inv_sqrt = np.diag(np.power(np.diag(D), -0.5))
        
    
        A_norm = D_inv_sqrt @ A_tilde @ D_inv_sqrt
        X = node_features.astype(np.float32)
        
        return A_norm, X


   