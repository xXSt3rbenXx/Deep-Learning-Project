import torch
from preprocessing import Preprocessing as pre

if torch.cuda.is_available():
    device=torch.device('cuda')
else:
    device=torch.device('cpu')


X_train, Y_train,X_val, Y_val, X_test, Y_test,adj = pre(data_path='Dataset/metr-la.csv',adj_path='Dataset/adj_Metr-LA.pkl').normalization()
X_train, Y_train,X_val, Y_val, X_test, Y_test=torch.from_numpy(X_train).float().to(device), torch.from_numpy(Y_train).float().to(device), torch.from_numpy(X_val).float().to(device), torch.from_numpy(Y_val).float().to(device), torch.from_numpy(X_test).float().to(device), torch.from_numpy(Y_test).float().to(device)