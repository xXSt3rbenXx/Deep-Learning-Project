#Questo modello è senza grafo -> dalla traccia il nostro obiettivo è fare #
#un confronto equo tra i modelli
#questo modello utilizza ancora la TCN -> caso di ablazione quindi serve per forza

#Architettura modello: lascio la TCN identica, tolgo la GCN da quello di prima
#isolo totalmente il grafo
#ci aspettiamo risultati che dipendono esclusivamente dal passaggio di informazioni spaziale tra le strade e i sensori

#COME LAVORA??
#lavora sfruttando gli ultimi 12 passi temporali di ciascun sensore in modo totalmente isolato ed indipendente, senza sapere cosa
#sta succedendo nei sensori adiacenti o nelle strade vicine
#in sintesi: NO GRAPH = TCN PURA che lavora solo sull'asse dei tempo, sensore per sensore


import torch
import torch.nn as nn
import torch.nn.functional as F
from Temporal_Convolutional_Network import TCNLayer

class Temporal_Model(nn.Module):
    def __init__(self, input_dim=1, out_dim=3, hidden_dim=32, kernel_size=2, num_layers=3,num_blocks=3 dropout=0.3):
        super(Temporal_Model, self).__init__()

        self.num_layers=num_layers
        self.num_blocks=num_blocks
        self.dropout=dropout

        #Creazione strati TCN
        tcn_layers=[]
        for i in range(num_layers):
            in_ch=input_dim if i==0 else hidden_dim
            #l'ultimo strato mantiene hidden_dim per la testa di output
            out_ch=hidden_dim
            tcn_layers.append(TCNLayer(in_ch, out_ch, blocks=num_blocks, kernel_size=kernel_size))
        self.tcn_layers=nn.ModuleList(tcn_layers)
        self.drop=nn.Dropout(dropout)
        self.l1=nn.Linear(hidden_dim, out_dim) #mediana
        self.l2=nn.Linear(hidden_dim, out_dim) #quantile inferiore
        self.l3=nn.Linear(hidden_dim, out_dim) #quantile superiore

    def forward(self, x):
        #input x shape: (B, T, N, F) oppure (B, T, N)
        #output q10, q50, q90 di shape ciascuno (B, N, 3)
        if x.ndim==3:
            #vedo se manca la dimensione F, se manca la aggiungo
            x=x.unsqueeze(-1)
        B, T, N, F_in=x.shape

        #permutiamo e uniamo Batch e Nodi per la Conv1D
        #(B, N, T, F)-> (B, N, F, T) -> (B*N, F, T)
        x_tcn_in=x.permute(0, 2, 3, 1).reshape(B*N, F_in, T)

        #passaggio attraverso gli strati TCN
        for layer in self.tcn_layers:
            x_tcn_in=layer(x_tcn_in)
            x_tcn_in=self.drop(x_tcn_in)

        #estrazione dell'ultimo step temporale: shape (B*N, hidden_dim)
        out_last=x_tcn_in[:, :, -1]

        #ripristino formula (B, N, hidden_dim)
        out=out_last.reshape(B, N,-1 )

        #calcolo mediana e quantili
        median=self.l1(out)
        low_quantile=F.softplus(self.l2(out))
        high_quantile=F.softplus(self.l3(out))

        q10=median-low_quantile
        q50=median
        q90=median+high_quantile

        return q10, q50, q90
