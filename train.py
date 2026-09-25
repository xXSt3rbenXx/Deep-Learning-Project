import torch
import torch.optim as optim
from torch.utils.data import TensorDataset
import numpy as np
from itertools import product

from utils import *

from preprocessing import Preprocessing as pre
from Graph_Convolutional_Network import GCN
from Temporal_Model import Temporal_Model
from Early_Stopping import EarlyStopping
from GIN_Model import GIN

# Riproducibilità
torch.manual_seed(67)
np.random.seed(67)



# Device Setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device utilizzato: {device}")

# Preprocessing e Grafo
prep = pre(data_path='Dataset/metr-la.csv', adj_path='Dataset/adj_Metr-LA.pkl')
X_train, Y_train, X_val, Y_val, X_test, Y_test, Y_val_ha, Y_test_ha, adj_matrix, train_grouped = prep.normalization(use_graph=True)

# Matrice Laplaciana e Polinomi di Chebyshev
L_norm, A_norm = graph_to_matrices(adj_matrix)
L_norm = L_norm.to(device)
A_norm = A_norm.to(device)

# Trasformazione Tensori (Shape: B, T, N, F_in=1 e Target su orizzonti 3, 6, 12)
X_train_t = torch.from_numpy(X_train).float().unsqueeze(-1)
Y_train_t = torch.from_numpy(Y_train).float().permute(0, 2, 1)[:, :, [2, 5, 11]]

X_val_t = torch.from_numpy(X_val).float().unsqueeze(-1)
Y_val_t = torch.from_numpy(Y_val).float().permute(0, 2, 1)[:, :, [2, 5, 11]]

X_test_t = torch.from_numpy(X_test).float().unsqueeze(-1)
Y_test_t = torch.from_numpy(Y_test).float().permute(0, 2, 1)[:, :, [2, 5, 11]]

Y_test_ha = torch.from_numpy(Y_test_ha).float().permute(0, 2, 1)[:, :, [2, 5, 11]]   

# DataLoaders (Incluso test_loader)
train_loader, val_loader, test_loader = make_loaders(
    TensorDataset(X_train_t, Y_train_t),
    TensorDataset(X_val_t, Y_val_t),
    TensorDataset(X_test_t, Y_test_t),
    batch_size=32
)

epochs = 50
quantiles = [0.1, 0.5, 0.9]

# --- 1. BASELINE HISTORICAL AVERAGE ---
ha_pb, ha_mae, ha_rmse = historical_average_baseline(Y_test_t, Y_test_ha, quantiles=quantiles)

# --- 2. GCN MODEL HP TUNING ---

K = [2, 3]
learning_rates = [1e-4, 1e-3]
layers = [2, 3]
prod = product( K, learning_rates, layers)
best_hp, best_val_loss = None, np.inf

for  k, lr, l in prod:
    print(f"\nIperparametri attuali GCN: K={k}, lr={lr}, layers={l}")
    T = chebyshev_pol(L_norm, K=k, device=device)
    model_gcn = GCN(input_dim=1, hidden_dim=32, out_dim=1, T_list=T, kernel_size=3, num_layers=l, dropout=0.3).to(device)
    optimizer_gcn = optim.Adam(model_gcn.parameters(), lr=lr)
    scaler_gcn = torch.amp.GradScaler('cuda' if device.type == 'cuda' else 'cpu')
    early_stopping = EarlyStopping(delta=0.001, verbose=True)

    train_and_eval_model(
        model_gcn, train_loader, val_loader, optimizer_gcn, scaler_gcn,
        quantiles, epochs, device, early_stopping, model_name="GCN Model Tuning"
    )
    if early_stopping.best_loss < best_val_loss:
        best_val_loss = early_stopping.best_loss
        best_hp = (k, lr, l)
        best_epochs = early_stopping.best_epoch

(k, lr, l) = best_hp
print(f"\nMigliori Iperparametri GCN: K={k}, lr={lr}, layers={l} | Epoche: {best_epochs}")
gcn_hp = {"K": k, "num_layers": l, "lr": lr}

# Addestramento Finale e Valutazione Test GCN
T = chebyshev_pol(L_norm, K=k, device=device)
model_gcn = GCN(input_dim=1, hidden_dim=32, out_dim=1, T_list=T, kernel_size=3, num_layers=l, dropout=0.3).to(device)
optimizer_gcn = optim.Adam(model_gcn.parameters(), lr=lr)
scaler_gcn = torch.amp.GradScaler('cuda' if device.type == 'cuda' else 'cpu')

train_tuned_model(model_gcn, train_loader, optimizer_gcn, scaler_gcn, quantiles, best_epochs, device, model_name="GCN Tuned")
gcn_pb, gcn_mae, gcn_rmse = evaluate_model_test(model_gcn, test_loader, quantiles, device, model_name="GCN Model (Spectral)")

# --- 3. TEMPORAL MODEL (NO GRAPH) ---
model_temporal = Temporal_Model(input_dim=1, out_dim=3, hidden_dim=32, kernel_size=3, num_layers=l, dropout=0.3).to(device)
optimizer_temporal = optim.Adam(model_temporal.parameters(), lr=lr)
scaler_temporal = torch.amp.GradScaler('cuda' if device.type == 'cuda' else 'cpu')
early_stopping_temp = EarlyStopping(delta=0.001, verbose=True)

train_and_eval_model(
    model_temporal, train_loader, val_loader, optimizer_temporal, scaler_temporal,
    quantiles, epochs, device, early_stopping_temp, model_name="Temporal Model (No Graph)"
)
temp_pb, temp_mae, temp_rmse = evaluate_model_test(model_temporal, test_loader, quantiles, device, model_name="Temporal Model (No Graph)")
print(f"\nMigliori Iperparametri Temporal Model: lr={lr}, layers={l}.")
temporal_hp = { "num_layers": l, "lr": lr}
# --- 4. GIN MODEL (SPATIAL GRAPH) ---
eps_candidates = [1e-4, 1e-3, 1e-2]
best_gin_val_loss = np.inf
best_gin_hp = None
best_gin_epochs = 0

prod = product(eps_candidates, learning_rates, layers)
for eps, lr, l in prod:
    print(f"\nIperparametri attuali GIN, eps={eps}, lr={lr}, layers={l}")
    torch.manual_seed(67)
    np.random.seed(67)
    model_gin = GIN(input_dim=1, hidden_dim=32, out_dim=1, A=A_norm,
                     kernel_size=3, num_layers=l, dropout=0.3, eps=eps).to(device)
    optimizer_gin = optim.Adam(model_gin.parameters(), lr=lr)
    scaler_gin = torch.amp.GradScaler('cuda' if device.type == 'cuda' else 'cpu')
    early_stopping = EarlyStopping(delta=0.001, verbose=True)

    train_and_eval_model(
        model_gin, train_loader, val_loader, optimizer_gin, scaler_gin,
        quantiles, epochs, device, early_stopping, model_name="GIN Model Tuning"
    )
    if early_stopping.best_loss < best_gin_val_loss:
        best_gin_val_loss = early_stopping.best_loss
        best_gin_hp = (eps, lr, l)
        best_gin_epochs = early_stopping.best_epoch

eps, lr, l = best_gin_hp
print(f"\nMigliori Iperparametri GIN: Eps={eps}, lr={lr}, layers={l} | Epoche: {best_gin_epochs}")
gin_hp = {"eps": eps, "num_layers": l, "lr": lr}


model_gin = GIN(input_dim=1, out_dim=1, hidden_dim=32, A=A_norm,
                kernel_size=3, num_layers=l, dropout=0.3, eps=eps).to(device)
optimizer_gin = optim.Adam(model_gin.parameters(), lr=lr)
scaler_gin = torch.amp.GradScaler('cuda' if device.type == 'cuda' else 'cpu')

train_tuned_model(model_gin, train_loader, optimizer_gin, scaler_gin, quantiles, best_gin_epochs, device, model_name="GIN Tuned")
gin_pb, gin_mae, gin_rmse = evaluate_model_test(model_gin, test_loader, quantiles, device, model_name="GIN Model (Spatial)")
# --- TABELLA RIASSUNTIVA FINALE SUL TEST SET ---
print("\n" + "="*80)
print("                       RIEPILOGO METRICHE TEST SET")
print("="*80)
print(f"{'Modello':<25} | {'Pinball Loss':<12} | {'MAE (15m/30m/60m)':<22} | {'RMSE (15m/30m/60m)'}")
print("-" * 80)
print(f"{'Historical Average':<25} | {ha_pb:<12.4f} | {f'{ha_mae[0]:.2f}/{ha_mae[1]:.2f}/{ha_mae[2]:.2f}':<22} | {f'{ha_rmse[0]:.2f}/{ha_rmse[1]:.2f}/{ha_rmse[2]:.2f}'}")
print(f"{'Temporal (No Graph)':<25} | {temp_pb:<12.4f} | {f'{temp_mae[0]:.2f}/{temp_mae[1]:.2f}/{temp_mae[2]:.2f}':<22} | {f'{temp_rmse[0]:.2f}/{temp_rmse[1]:.2f}/{temp_rmse[2]:.2f}'}")
print(f"{'GCN (Spectral)':<25} | {gcn_pb:<12.4f} | {f'{gcn_mae[0]:.2f}/{gcn_mae[1]:.2f}/{gcn_mae[2]:.2f}':<22} | {f'{gcn_rmse[0]:.2f}/{gcn_rmse[1]:.2f}/{gcn_rmse[2]:.2f}'}")
print(f"{'GIN (Spatial)':<25} | {gin_pb:<12.4f} | {f'{gin_mae[0]:.2f}/{gin_mae[1]:.2f}/{gin_mae[2]:.2f}':<22} | {f'{gin_rmse[0]:.2f}/{gin_rmse[1]:.2f}/{gin_rmse[2]:.2f}'}")
print("="*80)

# Salvataggio Pesi
torch.save({"state_dict": model_gcn.state_dict(), "hyperparameters": gcn_hp}, "gcn_final_complete.pt")
torch.save({"state_dict": model_temporal.state_dict(), "hyperparameters": temporal_hp}, "temporal_final.pt")
torch.save({"state_dict": model_gin.state_dict(), "hyperparameters": gin_hp}, "gin_final_complete.pt")
