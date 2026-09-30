import torch
import numpy as np
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import os

def denormalize(x, mean, std):
    return x*std+mean

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

    horizons_idx = [0,1,2] 
    horizon_names = ["Step 3 (15m)", "Step 6 (30m)", "Step 12 (60m)"]
    mae_list, rmse_list = [], []

    print(f"\n=================== Baseline: Historical Average (Test Set) ===================")
    print(f"Pinball Loss (Test): {pb_loss:.4f}")
    for idx, h_name in zip(horizons_idx, horizon_names):
        yt = y_true[:, :,idx]      # (num_sensori, num_samples) 
        yp = y_pred_ha[:, :,idx]   # stessa correzione per la predizione HA
        mae = torch.abs(yt - yp).mean().item()
        rmse = torch.sqrt(torch.mean((yt - yp) ** 2)).item()
        mae_list.append(mae)
        rmse_list.append(rmse)
        print(f"[{h_name}] MAE: {mae:.4f} | RMSE: {rmse:.4f}")

    return pb_loss, mae_list, rmse_list


def train_and_eval_model(
    model, train_loader, val_loader, optimizer, scaler,
    quantiles, epochs, device, early_stopping, model_name='Model'
):
    print(f"\n=================== Inizio Addestramento: {model_name} ===================")
    train_losses = []
    val_losses = []
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        for x_train, y_train in train_loader:
            x_train = x_train.to(device)
            y_train = y_train.to(device)

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
                x_val = x_val.to(device)
                y_val = y_val.to(device)
                with torch.amp.autocast(device_type=device.type):
                    output = model(x_val)
                    loss = pinball_loss(quantiles, y_val, output)
                val_loss += loss.item() * x_val.size(0)
        val_loss /= len(val_loader.dataset)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        early_stopping.check_early_stop(val_loss, epoch)
        print(
            f'{model_name} | '
            f'Epoca {epoch:02d}/{epochs:02d} | '
            f'Train Loss: {train_loss:.4f} | '
            f'Val Loss: {val_loss:.4f}'
        )
        if early_stopping.stop_training:
            print(f"Early stopping all'epoca {epoch}")
            break

    return train_losses, val_losses

def train_tuned_model(model, train_loader, optimizer, scaler, quantiles, epochs, device, model_name='Model'):
    print(f"\n=================== Inizio Addestramento Finale: {model_name} ===================")
    train_losses=[]
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
        train_losses.append(train_loss)
        print(f'{model_name} | Epoca {epoch:02d}/{epochs:02d} | Train Loss: {train_loss:.4f}')
    return train_losses


def evaluate_model_test(model,test_loader,quantiles,device,model_name="Model"):
    model.eval()
    total_pb_loss = 0.0
    all_q10 = []
    all_q50 = []
    all_q90 = []
    all_targets = []
    with torch.no_grad():
        for x_test, y_test in test_loader:
            x_test = x_test.to(device)
            y_test = y_test.to(device)
            with torch.amp.autocast(device_type=device.type):
                output = model(x_test)
                loss = pinball_loss(
                    quantiles,
                    y_test,
                    output
                    )

            total_pb_loss += loss.item() * x_test.size(0)
            # Salviamo tutti i quantili
            all_q10.append(output[0].cpu())
            all_q50.append(output[1].cpu())
            all_q90.append(output[2].cpu())

            all_targets.append(y_test.cpu())
    test_pb_loss = total_pb_loss / len(test_loader.dataset)
    q10 = torch.cat(all_q10, dim=0)
    q50 = torch.cat(all_q50, dim=0)
    q90 = torch.cat(all_q90, dim=0)
    targets = torch.cat(all_targets, dim=0)
    horizons_idx = [0, 1, 2]
    horizon_names = [
        "Step 3 (15m)",
        "Step 6 (30m)",
        "Step 12 (60m)"
    ]
    print(
        f"\n---------------- Valutazione Test Set: "
        f"{model_name} ----------------"
    )
    print(f"Pinball Loss (Test): {test_pb_loss:.4f}")
    mae_list = []
    rmse_list = []
    for idx, h_name in zip(horizons_idx, horizon_names):
        yt = targets[:, :, idx]
        yp = q50[:, :, idx]
        mae = torch.abs(yt - yp).mean().item()
        rmse = torch.sqrt(
            torch.mean((yt - yp) ** 2)
        ).item()
        mae_list.append(mae)
        rmse_list.append(rmse)
        print(
            f"[{h_name}] "
            f"MAE: {mae:.4f} | "
            f"RMSE: {rmse:.4f}"
        )
    return (
        test_pb_loss,
        mae_list,
        rmse_list,
        targets,
        q10,
        q50,
        q90
        )

def plot_training_curves(
    train_losses,
    val_losses=None,
    model_name='Modello',
    save_path=None
):
    epochs = np.arange(1, len(train_losses) + 1)
    fig=plt.figure(figsize=(9, 5))
    plt.plot(
        epochs,
        train_losses,
        label='Train Loss',
        linewidth=2
    )
    plt.plot(
        epochs,
        val_losses,
        label='Validation Loss',
        linestyle='--',
        linewidth=2
    )
    plt.xlabel('Epoca')
    plt.ylabel('Pinball Loss')
    plt.title(f'Training e Validation Loss - {model_name}')
    plt.xticks(epochs[::max(1, len(epochs)//10)])
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend()
    plt.tight_layout()
    if save_path:
        folder=os.path.dirname(save_path)
        if folder:
            os.makedirs(folder, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f'Grafico Salvato in: {save_path}')
    plt.show()
    plt.close(fig)
def plot_model_comparison(metrics_dict,  horizon_steps=[3, 6, 12], save_path=None):
    #confronto il modello su più metriche contemporaneamente
    num_metrics=len(metrics_dict)
    fig, axes=plt.subplots(1, num_metrics, figsize=(6*num_metrics, 5), sharex=True)
    if num_metrics==1:
        axes=[axes]
    width=0.2
    x=np.arange(len(horizon_steps))
    for ax, (metric_name, models) in zip(axes, metrics_dict.items()):
        num_models=len(models)
        for i, (model_name, values) in enumerate(models.items()):
            ax.bar(
                x+(i - (num_models-1)/2) * width,
                values, width, label=model_name
            )
        ax.set_xlabel("Orizzonte (Step Temporali)")
        ax.set_ylabel(metric_name)
        ax.set_title(f"Confronto {metric_name}")
        ax.set_xticks(x)
        ax.set_xticklabels([f"Step {s} ({s * 5}m)" for s in horizon_steps])
        ax.legend()
        ax.grid(axis="y", linestyle=":", alpha=0.7)
    plt.tight_layout()
    if save_path:
        folder=os.path.dirname(save_path)
        if folder:
            os.makedirs(folder, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f'Grafico Salvato in: {save_path}')
    plt.show()
def plot_quantile_predictions(y_true, q10, q50, q90, sensor_idx=0, horizon_idx=0,
                               num_steps=100, start_idx=0, model_name="Model",
                               mean=None, std=None, period_label='Normal', save_path=None):
    if torch.is_tensor(y_true): y_true = y_true.numpy()
    if torch.is_tensor(q10): q10 = q10.numpy()
    if torch.is_tensor(q50): q50 = q50.numpy()
    if torch.is_tensor(q90): q90 = q90.numpy()

    end_idx = min(start_idx + num_steps, y_true.shape[0])
    time_axis = np.arange(start_idx, end_idx)
    true_values = y_true[start_idx:end_idx, sensor_idx, horizon_idx]
    lower = q10[start_idx:end_idx, sensor_idx, horizon_idx]
    median = q50[start_idx:end_idx, sensor_idx, horizon_idx]
    upper = q90[start_idx:end_idx, sensor_idx, horizon_idx]


    if mean is not None and std is not None:
        true_values=denormalize(true_values, mean, std)
        lower=denormalize(lower, mean, std)
        median=denormalize(median, mean, std)
        upper=denormalize(upper, mean, std)
        ylabel='Velocità'
    else:
        ylabel='Valore'

    horizon_names = [
        "15 minuti",
        "30 minuti",
        "60 minuti"
    ]

    fig=plt.figure(figsize=(12, 6))
    plt.plot(time_axis, true_values, label="Valore reale",linewidth=1.5)
    plt.plot( time_axis,median, label="Q50", linewidth=2)
    plt.fill_between(time_axis,lower,upper, alpha=0.2,label="Intervallo 80% (Q10-Q90)")

    plt.title( f"{model_name} - Sensore {sensor_idx} - "
               f"Orizzonte {horizon_names[horizon_idx]} - Periodo {period_label}"
    )
    plt.xlabel("Passo temporale")
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()
    if save_path:
        import os
        folder=os.path.dirname(save_path)
        if folder:
            os.makedirs(folder, exist_ok=True)
        plt.savefig(save_path, dpi=300,bbox_inches="tight")
        print(f'Grafico salvato in :{save_path}')
    plt.show()
    plt.close()


def compute_calibration_metrics(y_true, q10, q90, target_coverage=80.0):
    if not torch.is_tensor(y_true): y_true = torch.tensor(y_true)
    if not torch.is_tensor(q10): q10 = torch.tensor(q10)
    if not torch.is_tensor(q90): q90 = torch.tensor(q90)
    in_bounds = (y_true >= q10) & (y_true <= q90)
    coverage = torch.mean(in_bounds.float()).item() * 100.0
    width = torch.mean(q90 - q10).item()
    calibration_error = abs(coverage - target_coverage)
    return coverage, width, calibration_error

def count_parameters(model):
    #conta i parametri del modello
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


#STRESS TEST

