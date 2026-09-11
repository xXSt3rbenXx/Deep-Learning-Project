import torch 
import torch.nn as nn
import torch.nn.functional as F


# Define a single Graph Convolutional Network layer.
# A GCN layer performs two main operations:
# 1. it transforms node features using a learnable linear layer;
# 2. it propagates and aggregates information through the graph structure
#    using the normalized adjacency matrix.
class GCNLayer(nn.Module):
    """
    Single Graph Convolutional Layer.

    This layer implements the operation:

        H' = A_norm X W

    where:
    - X is the input node feature matrix;
    - W is a learnable weight matrix;
    - A_norm is the normalized adjacency matrix;
    - H' is the updated node representation matrix.
    """
    
    def __init__(self, in_features, out_features):
        super(GCNLayer, self).__init__()
        
        # Learnable linear transformation.
        # This maps each node feature vector from in_features to out_features.
        self.linear = nn.Linear(in_features, out_features)
        
    def forward(self, x, adj):
        """
        Forward pass.

        x: node features with shape:
           (batch_size, max_nodes, in_features)

        adj: normalized adjacency matrix with shape:
             (batch_size, max_nodes, max_nodes)
        """
        
        # Apply the learnable linear transformation to each node independently.
        # This corresponds to XW in the GCN formula.
        #
        # Shape before: (batch_size, max_nodes, in_features)
        # Shape after:  (batch_size, max_nodes, out_features)
        x = self.linear(x)
        
        # Perform graph convolution using the normalized adjacency matrix.
        #
        # torch.bmm performs batch matrix multiplication.
        # For each graph in the batch, it computes:
        #
        #     A_norm @ XW
        #
        # This means that each node updates its representation by combining:
        # - its own transformed features;
        # - the transformed features of its neighboring nodes.
        #
        # The normalized adjacency matrix controls how much information
        # is received from each connected node.
        x = torch.bmm(adj, x)
        
        # Return the updated node representations.
        return x


# Define the full Graph Convolutional Network.
# This model takes an entire graph as input and produces a graph-level prediction.
# In this example, it is used for molecular property prediction.
class GCN(nn.Module):
    """
    Graph Convolutional Network for molecular property prediction.

    The model consists of:
    1. multiple GCN layers to learn node embeddings;
    2. a graph-level pooling operation to obtain one vector per graph;
    3. a predictor to predict the final molecular label.
    """
    
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=2, dropout=0.5):
        super(GCN, self).__init__()
        
        # Store model hyperparameters.
        self.num_layers = num_layers
        self.dropout = dropout
        
        # ModuleList is used to store multiple GCN layers.
        # Unlike a regular Python list, ModuleList registers the layers
        # so that PyTorch can train their parameters.
        self.gcn_layers = nn.ModuleList()
        
        # First GCN layer.
        # It maps the original node features to the hidden dimension.
        self.gcn_layers.append(GCNLayer(input_dim, hidden_dim))
        
        # Intermediate hidden GCN layers.
        # These layers keep the same hidden dimensionality.
        for _ in range(num_layers - 2):
            self.gcn_layers.append(GCNLayer(hidden_dim, hidden_dim))
        
        # Final GCN layer.
        # This produces the final node embeddings before graph-level pooling.
        if num_layers > 1:
            self.gcn_layers.append(GCNLayer(hidden_dim, hidden_dim))
        
        # classifier applied after graph-level pooling.
        # It maps the graph representation to the final output.
        #
        # For binary classification, output_dim is usually 1.
        self.predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, output_dim)
        )
        
    def forward(self, x, adj, mask):
        """
        Forward pass.

        x: node features with shape:
           (batch_size, max_nodes, input_dim)

        adj: normalized adjacency matrix with shape:
             (batch_size, max_nodes, max_nodes)

        mask: node mask with shape:
              (batch_size, max_nodes)

              mask = 1 for real nodes
              mask = 0 for padded nodes
        """
        
        # Apply all GCN layers sequentially.
        for i, gcn_layer in enumerate(self.gcn_layers):
            
            # Update node representations using graph convolution.
            x = gcn_layer(x, adj)
            
            # Apply ReLU and dropout after every GCN layer except the last one.
            # ReLU introduces non-linearity.
            # Dropout helps reduce overfitting during training.
            if i < len(self.gcn_layers) - 1:
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        
        # ---------------------------------------------------------
        # Graph-level representation using masked average pooling
        # ---------------------------------------------------------
        
        # Since graphs may have different numbers of nodes,
        # they were padded to the same size.
        # The mask tells the model which nodes are real and which are padding.
        
        # Expand the mask so that it matches the shape of x.
        #
        # mask shape before: (batch_size, max_nodes)
        # mask shape after:  (batch_size, max_nodes, hidden_dim)
        mask_expanded = mask.unsqueeze(-1).expand_as(x)
        
        # Set padded node embeddings to zero.
        # This ensures that padded nodes do not contribute to the graph representation.
        x_masked = x * mask_expanded
        
        # Compute the average of real node embeddings only.
        #
        # x_masked.sum(dim=1) sums node embeddings for each graph.
        # mask.sum(dim=1, keepdim=True) counts the number of real nodes.
        #
        # The result is one vector representation for each graph.
        graph_repr = x_masked.sum(dim=1) / mask.sum(dim=1, keepdim=True)
        
        # Pass the graph-level representation through the predictor.
        # The output represents the prediction for the whole graph/molecule.
        output = self.predictor(graph_repr)
        
        # Return the final prediction.
        return output



    