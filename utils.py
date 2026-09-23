import torch
import numpy as np
from torch.utils.data import DataLoader






def make_loaders(train_dataset, val_dataset, test_dataset, batch_size=32):
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, test_loader


def chebyshev_pol(L, K,device):
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

    return torch.Tensor(L_norm).float(), torch.Tensor(A_norm).float()


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
        yt = y_true[:, idx, :]      # (num_samples, num_sensors) -- ora seleziona il TIMESTEP giusto
        yp = y_pred_ha[:, idx, :]   # stessa correzione per la predizione HA
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


