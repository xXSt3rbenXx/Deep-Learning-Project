import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from itertools import product

from preprocessing import Preprocessing as pre
from Graph_Convolutional_Network import GCN
from Temporal_Model import Temporal_Model
from Early_Stopping import EarlyStopping


def make_loaders(train_dataset, val_dataset, batch_size=32):
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader


def chebyshev_pol(L, K):
    L_shape = L.shape[0]
    T_0 = torch.eye(L_shape, dtype=L.dtype).to(device)
    T_1 = L
    T = [T_0, T_1]
    for i in range(2, K + 1):
        T_i = 2 * L @ T[i - 1] - T[i - 2]
        T.append(T_i)
    return T


def graph_to_matrices(adj_matrix):
    n_nodes = adj_matrix.shape[0]
    A_tilde = adj_matrix.astype(np.float32)
    A_sim = (A_tilde + A_tilde.T) / 2
    D = np.diag(np.sum(A_sim, axis=1))
    D_inv_sqrt = np.diag(np.power(np.diag(D), -0.5, where=np.diag(D)>0)) #I valori 0 non verranno messi sotto radice per evitare np.inf

    A_norm = D_inv_sqrt @ A_sim @ D_inv_sqrt
    L = np.eye(n_nodes) - A_norm
    eigenvalues = np.linalg.eigvalsh(L)
    max_eig = eigenvalues[-1]
    L_norm = 2 * L / max_eig - np.eye(n_nodes)

    return torch.Tensor(L_norm)


def pinball_loss(quantiles, y, y_pred):
    total_loss = 0
    for q, y_p in zip(quantiles, y_pred):
        loss = torch.max(q * (y - y_p), (1 - q) * (y_p - y))
        total_loss += loss.mean()
    return total_loss


def historical_average_baseline(y_true, y_pred_ha, quantiles=[0.1, 0.5, 0.9]):
    # Replichiamo il valore deterministico sui 3 quantili per la Pinball Loss
    ha_quantiles = (y_pred_ha, y_pred_ha, y_pred_ha)
    pb_loss = pinball_loss(quantiles, y_true, ha_quantiles).item()

    #definisco i tre orizzonti temporali
    horizons_idx = [0, 1, 2]
    horizon_names = ["Step 3 (15m)", "Step 6 (30m)", "Step 12 (60m)"]
    mae_list, rmse_list = [], []

    #calcolo le metriche per ogni orizzonte temporale - MAE e RMSE
    for idx in horizons_idx:
        yt = y_true[:, :, idx]
        yp = y_pred_ha[:, :, idx]
        mae = torch.abs(yt - yp).mean().item()
        rmse = torch.sqrt(torch.mean((yt - yp) ** 2)).item()
        mae_list.append(mae)
        rmse_list.append(rmse)
    return pb_loss, horizon_names, mae_list, rmse_list


def train_and_eval_model(model, train_loader, val_loader, optimizer, scaler, quantiles, epochs, device, early_stopping, model_name='Model'):
    print(f"\n=================== Inizio Addestramento: {model_name} ===================")
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        for x_train, y_train in train_loader:
            x_train, y_train = x_train.to(device), y_train.to(device)
            optimizer.zero_grad()

            with torch.amp.autocast(device_type=device.type):
                output = model(x_train)
                loss = pinball_loss(quantiles, y_train, output)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item() * x_train.size(0)


        #Train loss come media pesata dei valori
        train_loss /= len(train_loader.dataset)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x_val, y_val in val_loader:
                x_val, y_val = x_val.to(device), y_val.to(device)

                with torch.amp.autocast(device_type=device.type):
                    output = model(x_val)
                    loss = pinball_loss(quantiles, y_val, output)

                val_loss += loss.item() * x_val.size(0)

            val_loss /= len(val_loader.dataset)
                # Check condition
            early_stopping.check_early_stop(val_loss)
    
            if early_stopping.stop_training:
                print(f"Early stopping all'epoca {epoch}")
                break

        print(f'{model_name} | Epoca {epoch:02d}/{epochs:02d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}')


# Device Setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Preprocessing e Grafo
prep = pre(data_path='Dataset/metr-la.csv', adj_path='Dataset/adj_Metr-LA.pkl')
X_train, Y_train, X_val, Y_val, X_test, Y_test, Y_val_ha, Y_test_ha, adj_matrix, train_grouped = prep.normalization(use_graph=True)

#Matrice Laplaciana e Polinomi di Chebyshev
L_norm = graph_to_matrices(adj_matrix).to(device)
T = chebyshev_pol(L_norm, K=2)

# 3. Trasformazione Tensori (Shape: B, T, N, F_in=1 e Target su orizzonti 3, 6, 12)
X_train_t = torch.from_numpy(X_train).float().unsqueeze(-1)
Y_train_t = torch.from_numpy(Y_train).float().permute(0, 2, 1)[:, :, [2, 5, 11]]

X_val_t = torch.from_numpy(X_val).float().unsqueeze(-1)
Y_val_t = torch.from_numpy(Y_val).float().permute(0, 2, 1)[:, :, [2, 5, 11]]

X_test_t = torch.from_numpy(X_test).float().unsqueeze(-1)
Y_test_t = torch.from_numpy(Y_test).float().permute(0, 2, 1)[:, :, [2, 5, 11]]

#DataLoaders
train_loader, val_loader = make_loaders(
    TensorDataset(X_train_t, Y_train_t),
    TensorDataset(X_val_t, Y_val_t),
    batch_size=32
)

epochs = 10
quantiles = [0.1, 0.5, 0.9]

#Temporal Model
model_temporal = Temporal_Model(input_dim=1, out_dim=3, hidden_dim=32, kernel_size=2, num_layers=3, dropout=0.3).to(device)
optimizer_temporal = optim.Adam(model_temporal.parameters(), lr=0.001)
scaler_temporal = torch.amp.GradScaler('cuda' if device.type == 'cuda' else 'cpu')
early_stopping = EarlyStopping(verbose=True)

#Eseguo l'hyperparameter tuning solo sul modello con grafo, ci basterà semplicemente utilizzare gli stessi parametri del modello con grafo sul modello senza grafo ai fini di confronto
train_and_eval_model(
    model_temporal, train_loader, val_loader, optimizer_temporal, scaler_temporal,
    quantiles, epochs, device, early_stopping, model_name="Temporal Model (No Graph)"
)

#GCN Model hp tuning
h_dim=[32, 64] #Tengo la hd bassa
K=[2,3]
learning_rates = [1e-4, 1e-3, 3e-3]
prod=product(h_dim, K, learning_rates)
best_hp,best_model,best_val_loss=None, None, np.inf
for h,k,lr in prod:
    T = chebyshev_pol(L_norm, K=k)
    model_gcn = GCN(input_dim=1, hidden_dim=h, out_dim=1, T_list=T, kernel_size=3, num_layers=3, dropout=0.3).to(device)
    optimizer_gcn = optim.Adam(model_gcn.parameters(), lr=lr)
    scaler_gcn = torch.amp.GradScaler('cuda' if device.type == 'cuda' else 'cpu')
    early_stopping = EarlyStopping(verbose=True)

    train_and_eval_model(
        model_gcn, train_loader, val_loader, optimizer_gcn, scaler_gcn,
        quantiles, epochs, device, early_stopping, model_name="GCN Model (With Graph)"
    )
    if early_stopping.best_loss < best_val_loss:
        best_val_loss = early_stopping.best_loss
        best_model = model_gcn
        best_hp = (h, k, lr)