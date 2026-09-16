import torch
from preprocessing import Preprocessing as pre
import numpy as np




def chebyshev_pol( L, K):
        L_shape=L.shape[0]
        T_0= np.eye(L_shape, dtype=L.dtype)
        T_1=L
        T=[T_0, T_1]
        for i in range(2, K+1):
            T_i= 2*L @ T[i-1] - T[i-2]
            T.append(T_i)
        return T 


def graph_to_matrices(node_features, adj_matrix):
        
 
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

#q=[0.1,0.5,0.9] per quando verrà sviluppato
def pinball_loss(quantiles, y, y_pred):
      total_loss = 0
      for q, y_p in zip(quantiles, y_pred):
            loss=torch.max(q*(y-y_p), (1-q)*(y_p-y))
            total_loss += loss.mean()
      return total_loss

def historical_average_baseline(train_grouped, datetime):
      #Implementare la ricerca nella lookup table già pronta con MSE loss
      pass


if torch.cuda.is_available():
    device=torch.device('cuda')
else:
    device=torch.device('cpu')


X_train, Y_train,X_val, Y_val, X_test, Y_test,adj,train_grouped = pre(data_path='Dataset/metr-la.csv',adj_path='Dataset/adj_Metr-LA.pkl').normalization()
X_train, Y_train,X_val, Y_val, X_test, Y_test=torch.from_numpy(X_train).float().to(device), torch.from_numpy(Y_train).float().to(device), torch.from_numpy(X_val).float().to(device), torch.from_numpy(Y_val).float().to(device), torch.from_numpy(X_test).float().to(device), torch.from_numpy(Y_test).float().to(device)