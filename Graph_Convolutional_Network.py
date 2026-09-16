import torch 
import torch.nn as nn
import torch.nn.functional as F
from Temporal_Convolutional_Network import TCNLayer



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
            elif i==num_layers-1 and num_layers > 1:
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

        self.linear1=nn.Linear(out_dim, 3)
        self.linear2=nn.Linear(out_dim, 3)
        self.linear3=nn.Linear(out_dim, 3)
        
        
        
  
        
        

     
    def forward(self, x):
        # x: (B, T, N, F)
        B, T, N, _ = x.shape
        T_list = [getattr(self, f"T_{k}") for k in range(self.K + 1)]

        for i in range(self.num_layers):
        
            x_gcn_in = x .reshape(B * T, N, -1)   # (B*T, N, F)
            x_gcn_out = self.gcn_layers[i](x_gcn_in, T_list)          # (B*T, N, hidden_dim)
            F_hidden = x_gcn_out.shape[-1]
            x = x_gcn_out.reshape(B, T, N, F_hidden).permute(0, 2, 1, 3)  # torna a (B, N, T, hidden_dim)

            # --- TCN: fondi batch e N ---
            x_tcn_in = x.reshape(B * N, T, F_hidden).permute(0, 2, 1)     # (B*N, F, T)
            x_tcn_out = self.tcn_layers[i](x_tcn_in)                       # (B*N, F', T)
            F_out = x_tcn_out.shape[1]
            x = x_tcn_out.reshape(B, N, F_out,T ).permute(0, 3, 1,2)       # torna a (B,  T,N, F')



        #Adesso genero i  quantili finali su 3/6/12 step temporali
        #eseguo la previsione unicamente sull'ultimo istante temporale generato (B, N, F')

        out=x[:, -1, :,:]

        median= self.linear1(out)
        #Gli altri quantili sono scritti solo in forma di incremento e decremento in modo che rimangano sopra/sotto il valore della mediana
        #softplus(x) = log(1 + e^x) consente di ottenere valori sempre positivi 
        low_quantile= F.softplus(self.linear2(out))
        high_quantile= F.softplus(self.linear3(out))    



        return median-low_quantile, median, median+high_quantile 



    


    




   