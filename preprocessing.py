import pandas as pd
import numpy as np

class Preprocessing:
    def __init__(self, data_path, adj_path):
        self.data_path = data_path
        self.adj_path = adj_path

    def normalization(self, use_graph=False):
        # Caricamento Dati e Indicizzazione Temporale
        df = pd.read_csv(self.data_path)
        threshold = 0.1
        if use_graph:
            if self.adj_path is None:
                raise ValueError("use_graph=True ma non è stato fornito adj_path.")
            adj = pd.read_pickle(self.adj_path)
            adj_matrix = adj[2].copy()
            adj_matrix[adj_matrix < threshold] = 0.0
        else:
            adj_matrix = None

        datetime_col = df.columns[0]
        df[datetime_col] = pd.to_datetime(df[datetime_col])
        df = df.set_index(datetime_col)

        sensor_cols = df.columns.tolist()

        # Distinzione Spaziale degli Zeri
        network_spatial_mean = df[sensor_cols].mean(axis=1)

        is_zero = (df[sensor_cols] == 0)
        is_global_blackout = (network_spatial_mean == 0)
        is_isolated_fault_mask = (network_spatial_mean > 10).values[:, None]
        is_fault = is_zero & (is_isolated_fault_mask | is_global_blackout.values[:, None])

        df_cleaned = df.copy()
        df_cleaned[is_fault] = np.nan

        # Creazione Metadati Temporali
        dow = df_cleaned.index.dayofweek
        time_slot = df_cleaned.index.hour * 12 + df_cleaned.index.minute // 5
        meta_df = pd.DataFrame({'dow': dow, 'time_slot': time_slot}, index=df_cleaned.index)

        train_ratio = 0.70
        train_end = int(len(df_cleaned) * train_ratio)

        df_train = df_cleaned.iloc[:train_end]
        meta_train = meta_df.iloc[:train_end]

        # Calcolo medie storiche SOLO sui dati di Training
        train_grouped = df_train.groupby([meta_train['dow'], meta_train['time_slot']]).mean()

        # Mappatura ed Imputazione sul Dataset Completo
        mapped_means = meta_df.merge(
            train_grouped,
            on=['dow', 'time_slot'],
            how='left'
        ).set_index(df_cleaned.index)[sensor_cols]

        df_imputed = df_cleaned.fillna(mapped_means)
        df = df_imputed.interpolate(method='time').ffill().bfill()

        print("Numero di NaN residui:", df.isna().sum().sum())

        # Splitting
        n = len(df)
        train_end = int(n * 0.70)
        val_end = int(n * 0.80)

        x_train = df.iloc[:train_end]
        x_val   = df.iloc[train_end:val_end]
        x_test  = df.iloc[val_end:]

        # Z-Score Globale su Train
        mean = x_train.values.mean()
        std  = x_train.values.std()
        x_train_norm = (x_train - mean) / std
        x_val_norm   = (x_val - mean) / std
        x_test_norm  = (x_test - mean) / std

        print(f"Global Mean (Train): {mean:.4f}")
        print(f"Global Std (Train):  {std:.4f}")

        # Normalizzazione delle medie storiche per la baseline HA
        mapped_means_norm = (mapped_means - mean) / std
        ha_val_norm  = mapped_means_norm.iloc[train_end:val_end]
        ha_test_norm = mapped_means_norm.iloc[val_end:]

        # Finestre Scorrevoli
        X_train, Y_train = self.sliding_windows(x_train_norm)
        X_val, Y_val     = self.sliding_windows(x_val_norm)
        X_test, Y_test   = self.sliding_windows(x_test_norm)

        _, Y_val_ha  = self.sliding_windows(ha_val_norm)
        _, Y_test_ha = self.sliding_windows(ha_test_norm)

        return X_train, Y_train, X_val, Y_val, X_test, Y_test, Y_val_ha, Y_test_ha, adj_matrix, train_grouped

    def sliding_windows(self, data, window_in=12, window_out=12):
        if hasattr(data, 'values'):
            data = data.values

        X, Y = [], []
        num_samples = len(data) - window_in - window_out + 1

        for i in range(num_samples):
            X.append(data[i: i + window_in])
            Y.append(data[i + window_in: i + window_in + window_out])

        return np.array(X), np.array(Y)