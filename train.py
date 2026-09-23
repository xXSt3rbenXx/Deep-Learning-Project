import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from itertools import product

from preprocessing import Preprocessing as pre
from Graph_Convolutional_Network import GCN
from Temporal_Model import Temporal_Model
from Early_Stopping import EarlyStopping
from GIN_Model import GIN

# Riproducibilità
torch.manual_seed(67)
np.random.seed(67)


def make_loaders(train_dataset, val_dataset, test_dataset, batch_size=32):
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, test_loader


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
    D_inv_sqrt = np.diag(np.power(np.diag(D), -0.5, where=np.diag(D) > 0))

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
    # Convertiamo in tensori PyTorch se sono array numpy
    if isinstance(y_true, np.ndarray):
        y_true = torch.from_numpy(y_true).float()
    if isinstance(y_pred_ha, np.ndarray):
        y_pred_ha = torch.from_numpy(y_pred_ha).float()

    ha_quantiles = (y_pred_ha, y_pred_ha, y_pred_ha)
    pb_loss = pinball_loss(quantiles, y_true, ha_quantiles).item()

    horizons_idx = [2, 5, 11]  # Corrispondono a 15m, 30m, 60m
    horizon_names = ["Step 3 (15m)", "Step 6 (30m)", "Step 12 (60m)"]
    mae_list, rmse_list = [], []

    print(f"\n=================== Baseline: Historical Average (Test Set) ===================")
    print(f"Pinball Loss (Test): {pb_loss:.4f}")
    for idx, h_name in zip(horizons_idx, horizon_names):
        yt = y_true[:, :, idx] if y_true.ndim == 3 else y_true[:, idx, :]
        yp = y_pred_ha[:, :, idx] if y_pred_ha.ndim == 3 else y_pred_ha[:, idx, :]
        mae = torch.abs(yt - yp).mean().item()
        rmse = torch.sqrt(torch.mean((yt - yp) ** 2)).item()
        mae_list.append(mae)
        rmse_list.append(rmse)
        print(f"[{h_name}] MAE: {mae:.4f} | RMSE: {rmse:.4f}")

    return pb_loss, mae_list, rmse_list


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

            early_stopping.check_early_stop(val_loss, epoch)

            if early_stopping.stop_training:
                print(f"Early stopping all'epoca {epoch}")
                break

        print(f'{model_name} | Epoca {epoch:02d}/{epochs:02d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}')


def train_tuned_model(model, train_loader, optimizer, scaler, quantiles, epochs, device, model_name='Model'):
    print(f"\n=================== Inizio Addestramento Finale: {model_name} ===================")
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

        train_loss /= len(train_loader.dataset)
        print(f'{model_name} | Epoca {epoch:02d}/{epochs:02d} | Train Loss: {train_loss:.4f}')


def evaluate_model_test(model, test_loader, quantiles, device, model_name="Model"):
    """Valuta il modello addestrato sul Test Set calcolando Pinball Loss, MAE e RMSE sui 3 orizzonti."""
    model.eval()
    total_pb_loss = 0.0
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for x_test, y_test in test_loader:
            x_test, y_test = x_test.to(device), y_test.to(device)
            with torch.amp.autocast(device_type=device.type):
                output = model(x_test)
                loss = pinball_loss(quantiles, y_test, output)

            total_pb_loss += loss.item() * x_test.size(0)
            all_preds.append(output[1].cpu())  # Usiamo il quantile mediano (q50) per MAE/RMSE
            all_targets.append(y_test.cpu())

    test_pb_loss = total_pb_loss / len(test_loader.dataset)
    preds_cat = torch.cat(all_preds, dim=0)
    targets_cat = torch.cat(all_targets, dim=0)

    horizons_idx = [0, 1, 2]
    horizon_names = ["Step 3 (15m)", "Step 6 (30m)", "Step 12 (60m)"]

    print(f"\n---------------- Valutazione Test Set: {model_name} ----------------")
    print(f"Pinball Loss (Test): {test_pb_loss:.4f}")

    mae_list, rmse_list = [], []
    for idx, h_name in zip(horizons_idx, horizon_names):
        yt = targets_cat[:, :, idx]
        yp = preds_cat[:, :, idx]
        mae = torch.abs(yt - yp).mean().item()
        rmse = torch.sqrt(torch.mean((yt - yp) ** 2)).item()
        mae_list.append(mae)
        rmse_list.append(rmse)
        print(f"[{h_name}] MAE: {mae:.4f} | RMSE: {rmse:.4f}")

    return test_pb_loss, mae_list, rmse_list


# Device Setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device utilizzato: {device}")

# Preprocessing e Grafo
prep = pre(data_path='Dataset/metr-la.csv', adj_path='Dataset/adj_Metr-LA.pkl')
X_train, Y_train, X_val, Y_val, X_test, Y_test, Y_val_ha, Y_test_ha, adj_matrix, train_grouped = prep.normalization(use_graph=True)

# Matrice Laplaciana e Polinomi di Chebyshev
L_norm = graph_to_matrices(adj_matrix).to(device)

# Trasformazione Tensori (Shape: B, T, N, F_in=1 e Target su orizzonti 3, 6, 12)
X_train_t = torch.from_numpy(X_train).float().unsqueeze(-1)
Y_train_t = torch.from_numpy(Y_train).float().permute(0, 2, 1)[:, :, [2, 5, 11]]

X_val_t = torch.from_numpy(X_val).float().unsqueeze(-1)
Y_val_t = torch.from_numpy(Y_val).float().permute(0, 2, 1)[:, :, [2, 5, 11]]

X_test_t = torch.from_numpy(X_test).float().unsqueeze(-1)
Y_test_t = torch.from_numpy(Y_test).float().permute(0, 2, 1)[:, :, [2, 5, 11]]

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
ha_pb, ha_mae, ha_rmse = historical_average_baseline(Y_test, Y_test_ha, quantiles=quantiles)

# --- 2. GCN MODEL HP TUNING ---
h_dim = [32, 64]
K = [2, 3]
learning_rates = [1e-4, 1e-3, 3e-3]
layers = [2, 3]
prod = product(h_dim, K, learning_rates, layers)
best_hp, best_model, best_val_loss = None, None, np.inf

for h, k, lr, l in prod:
    T = chebyshev_pol(L_norm, K=k)
    model_gcn = GCN(input_dim=1, hidden_dim=h, out_dim=1, T_list=T, kernel_size=3, num_layers=l, dropout=0.3).to(device)
    optimizer_gcn = optim.Adam(model_gcn.parameters(), lr=lr)
    scaler_gcn = torch.amp.GradScaler('cuda' if device.type == 'cuda' else 'cpu')
    early_stopping = EarlyStopping(delta=0.001, verbose=True)

    train_and_eval_model(
        model_gcn, train_loader, val_loader, optimizer_gcn, scaler_gcn,
        quantiles, epochs, device, early_stopping, model_name="GCN Model Tuning"
    )
    if early_stopping.best_loss < best_val_loss:
        best_val_loss = early_stopping.best_loss
        best_hp = (h, k, lr, l)
        best_epochs = early_stopping.best_epoch

(h, k, lr, l) = best_hp
print(f"\nMigliori Iperparametri GCN: h_dim={h}, K={k}, lr={lr}, layers={l} | Epoche: {best_epochs}")

# Addestramento Finale e Valutazione Test GCN
T = chebyshev_pol(L_norm, K=k)
model_gcn = GCN(input_dim=1, hidden_dim=h, out_dim=1, T_list=T, kernel_size=3, num_layers=l, dropout=0.3).to(device)
optimizer_gcn = optim.Adam(model_gcn.parameters(), lr=lr)
scaler_gcn = torch.amp.GradScaler('cuda' if device.type == 'cuda' else 'cpu')

train_tuned_model(model_gcn, train_loader, optimizer_gcn, scaler_gcn, quantiles, best_epochs, device, model_name="GCN Tuned")
gcn_pb, gcn_mae, gcn_rmse = evaluate_model_test(model_gcn, test_loader, quantiles, device, model_name="GCN Model (Spectral)")

# --- 3. TEMPORAL MODEL (NO GRAPH) ---
model_temporal = Temporal_Model(input_dim=1, out_dim=3, hidden_dim=h, kernel_size=2, num_layers=l, dropout=0.3).to(device)
optimizer_temporal = optim.Adam(model_temporal.parameters(), lr=lr)
scaler_temporal = torch.amp.GradScaler('cuda' if device.type == 'cuda' else 'cpu')
early_stopping_temp = EarlyStopping(delta=0.001, verbose=True)

train_and_eval_model(
    model_temporal, train_loader, val_loader, optimizer_temporal, scaler_temporal,
    quantiles, epochs, device, early_stopping_temp, model_name="Temporal Model (No Graph)"
)
temp_pb, temp_mae, temp_rmse = evaluate_model_test(model_temporal, test_loader, quantiles, device, model_name="Temporal Model (No Graph)")

# --- 4. GIN MODEL (SPATIAL GRAPH) ---
adj_tensor = torch.from_numpy(adj_matrix).float().to(device)
eps_candidates = [1e-4, 1e-3, 1e-2]

best_gin_eps = None
best_gin_val_loss = np.inf
best_gin_epochs = 0

for eps in eps_candidates:
    model_gin = GIN(
        input_dim=1, out_dim=1, hidden_dim=h, adj_matrix=adj_tensor,
        kernel_size=3, num_layers=l, dropout=0.3, eps=eps
    ).to(device)

    optimizer_gin = optim.Adam(model_gin.parameters(), lr=lr)
    scaler_gin = torch.amp.GradScaler('cuda' if device.type == 'cuda' else 'cpu')
    early_stopping_gin = EarlyStopping(delta=0.001, verbose=True)

    train_and_eval_model(
        model_gin, train_loader, val_loader, optimizer_gin, scaler_gin,
        quantiles, epochs, device, early_stopping_gin, model_name=f"GIN Model (eps={eps})"
    )

    if early_stopping_gin.best_loss < best_gin_val_loss:
        best_gin_val_loss = early_stopping_gin.best_loss
        best_gin_eps = eps
        best_gin_epochs = early_stopping_gin.best_epoch

# Addestramento Finale e Valutazione Test GIN
model_gin = GIN(
    input_dim=1, out_dim=1, hidden_dim=h, adj_matrix=adj_tensor,
    kernel_size=3, num_layers=l, dropout=0.3, eps=best_gin_eps
).to(device)

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
torch.save({"state_dict": model_gcn.state_dict(), "hyperparameters": {"hidden_dim": h, "K": k, "num_layers": l, "lr": lr}}, "gcn_final_complete.pt")
torch.save(model_temporal.state_dict(), "temporal_final.pt")
torch.save({"state_dict": model_gin.state_dict(), "hyperparameters": {"hidden_dim": h, "eps": best_gin_eps, "num_layers": l, "lr": lr}}, "gin_final_complete.pt")