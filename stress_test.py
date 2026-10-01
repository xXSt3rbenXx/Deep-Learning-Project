import torch
import numpy as np
from utils import *
from preprocessing import Preprocessing as pre
from Graph_Convolutional_Network import GCN
from Temporal_Model import Temporal_Model
from GIN_Model import GIN
from torch.utils.data import TensorDataset
from utils import pinball_loss, compute_calibration_metrics

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.manual_seed(67)
np.random.seed(67)

prep = pre(data_path='Dataset/metr-la.csv', adj_path='Dataset/adj_Metr-LA.pkl')
X_train, Y_train, X_val, Y_val, X_test, Y_test, _, _, adj_matrix, _, data_mean, data_std = prep.normalization(use_graph=True)
L_norm, A_norm = graph_to_matrices(adj_matrix)
L_norm, A_norm = L_norm.to(device), A_norm.to(device)

X_val_t = torch.from_numpy(X_val).float().unsqueeze(-1)
Y_val_t = torch.from_numpy(Y_val).float().permute(0, 2, 1)[:, :, [2, 5, 11]]
X_test_t = torch.from_numpy(X_test).float().unsqueeze(-1)
Y_test_t = torch.from_numpy(Y_test).float().permute(0, 2, 1)[:, :, [2, 5, 11]]

_, val_loader, test_loader = make_loaders(
    TensorDataset(X_val_t, Y_val_t),
    TensorDataset(X_val_t, Y_val_t),
    TensorDataset(X_test_t, Y_test_t),
    batch_size=32
)

quantiles = [0.1, 0.5, 0.9]
# GCN
ckpt_gcn = torch.load('./Parameters/gcn_final_complete.pt', map_location=device)
hp_gcn = ckpt_gcn['hyperparameters']
T_gcn = chebyshev_pol(L_norm, K=hp_gcn['K'], device=device)
model_gcn = GCN(input_dim=1, hidden_dim=32, out_dim=1, T_list=T_gcn, kernel_size=3, num_layers=hp_gcn['num_layers'], dropout=0.3).to(device)
model_gcn.load_state_dict(ckpt_gcn['state_dict'])
model_gcn.eval()

# TEMPORAL
ckpt_temp = torch.load("./Parameters/temporal_final.pt", map_location=device)
hp_temp = ckpt_temp["hyperparameters"]
model_temporal = Temporal_Model(input_dim=1, out_dim=3, hidden_dim=32, kernel_size=3, num_layers=hp_temp["num_layers"], dropout=0.3).to(device)
model_temporal.load_state_dict(ckpt_temp["state_dict"])
model_temporal.eval()

# GIN
ckpt_gin = torch.load("./Parameters/gin_final_complete.pt", map_location=device)
hp_gin = ckpt_gin["hyperparameters"]
model_gin = GIN(input_dim=1, hidden_dim=32, out_dim=1, A=A_norm, kernel_size=3, num_layers=hp_gin["num_layers"], dropout=0.3, eps=hp_gin["eps"]).to(device)
model_gin.load_state_dict(ckpt_gin["state_dict"])
model_gin.eval()

print("Tutti i checkpoint caricati con successo in modalità .eval()!\n")

# 3. Funzione di Valutazione degli Stress Test
def evaluate_traffic_stress_scenarios(model, test_loader, quantiles, device, data_mean, data_std, mode, threshold_percentile=10):
    model.eval()

    window_speeds = []
    with torch.no_grad():
        for x_batch, _ in test_loader:
            real_speeds = denormalize(x_batch, data_mean, data_std)
            batch_means = real_speeds.mean(dim=(1, 2,3)).cpu().numpy()
            window_speeds.extend(batch_means)
    window_speeds = np.array(window_speeds)

    if mode == 'high_congestion':
        cutoff_speed = np.percentile(window_speeds, threshold_percentile)
        selected_mask = window_speeds <= cutoff_speed
        scenario_label = f'High Congestion (v <= {cutoff_speed:.1f} mph)'
    elif mode == 'zero_congestion':
        cutoff_speed = np.percentile(window_speeds, 100 - threshold_percentile)
        selected_mask = window_speeds >= cutoff_speed
        scenario_label = f'Zero Congestion (v >= {cutoff_speed:.1f} mph)'
    else:
        raise ValueError('Mode error: usa "high_congestion" o "zero_congestion"')

    all_q10, all_q50, all_q90, all_targets = [], [], [], []
    sample_idx = 0
    total_pb_loss = 0.0
    selected_samples_count = 0

    with torch.no_grad():
        for x_test, y_test in test_loader:
            batch_size = x_test.size(0)
            batch_mask = selected_mask[sample_idx : sample_idx + batch_size]
            sample_idx += batch_size

            # Fix: Controllo array booleano NumPy
            if not batch_mask.any():
                continue

            x_sub = x_test[batch_mask].to(device)
            y_sub = y_test[batch_mask].to(device)

            with torch.amp.autocast(device_type=device.type):
                output = model(x_sub)
                loss = pinball_loss(quantiles, y_sub, output)

            total_pb_loss += loss.item() * x_sub.size(0)
            selected_samples_count += x_sub.size(0)

            all_q10.append(output[0].cpu())
            all_q50.append(output[1].cpu())
            all_q90.append(output[2].cpu())
            all_targets.append(y_sub.cpu())

    if selected_samples_count == 0:
        print(f"Nessun campione trovato per lo scenario: {scenario_label}")
        return None

    test_pb_loss = total_pb_loss / selected_samples_count
    q10 = torch.cat(all_q10, dim=0)
    q50 = torch.cat(all_q50, dim=0)
    q90 = torch.cat(all_q90, dim=0)
    targets = torch.cat(all_targets, dim=0)

    horizons_idx = [0, 1, 2]
    horizon_names = ["Step 3 (15m)", "Step 6 (30m)", "Step 12 (60m)"]

    mae_real_list, rmse_real_list = [], []
    for idx in horizons_idx:
        mae_norm = torch.abs(targets[:, :, idx] - q50[:, :, idx]).mean().item()
        rmse_norm = torch.sqrt(torch.mean((targets[:, :, idx] - q50[:, :, idx]) ** 2)).item()
        mae_real_list.append(mae_norm * data_std)
        rmse_real_list.append(rmse_norm * data_std)
    coverage, width, ace = compute_calibration_metrics(targets, q10, q90, target_coverage=80.0)
    return {
        'label': scenario_label,
        'pb_loss': test_pb_loss,
        'mae_real': mae_real_list,      # ora una lista di 3 valori, non uno scalare
        'rmse_real': rmse_real_list,    # idem
        'coverage': coverage,
        'ace': ace,
        'mpiw': width,
        'targets': targets, 'q10': q10, 'q50': q50, 'q90': q90
    }

models = {
    'GCN': model_gcn,
    'Temporal': model_temporal,
    'GIN': model_gin
}

scenarios = ['high_congestion', 'zero_congestion']
results = {}

for model_name, model_obj in models.items():
    results[model_name] = {}
    print(f"==================== STRESS TEST: {model_name} ====================")
    for mode in scenarios:
        res = evaluate_traffic_stress_scenarios(
            model=model_obj,
            test_loader=test_loader,
            quantiles=quantiles,
            device=device,
            data_mean=data_mean,
            data_std=data_std,
            mode=mode,
            threshold_percentile=10
        )
        results[model_name][mode] = res
        if res:
            print(f"Scenario: {res['label']}")
            print(f"  - Pinball Loss: {res['pb_loss']:.4f}")
            for h_name, mae_h, rmse_h in zip(horizon_names, res['mae_real'], res['rmse_real']):
                print(f"  - [{h_name}] MAE: {mae_h:.2f} mph | RMSE: {rmse_h:.2f} mph")
            print(f"  - Coverage    : {res['coverage']:.2f}%")
            print(f"  - ACE         : {res['ace']:.4f}")
            print(f"  - MPIW        : {res['mpiw']:.4f}\n")