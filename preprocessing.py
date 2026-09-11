import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
# Caricamento Dati e Indicizzazione Temporale
data_path = 'Dataset/metr-la.csv'  # oppure 'METR-LA.csv'
df = pd.read_csv(data_path)

datetime_col = df.columns[0]
df[datetime_col] = pd.to_datetime(df[datetime_col])
df = df.set_index(datetime_col)

# Lista pulita di sole colonne sensori (C1, C2... o ID numerici)
sensor_cols = df.columns.tolist()

#  Distinzione Spaziale degli Zeri
network_spatial_mean = df[sensor_cols].mean(axis=1)

is_zero = (df[sensor_cols] == 0)
is_global_blackout = (network_spatial_mean == 0)
is_isolated_fault = is_zero.apply(lambda col: network_spatial_mean > 10, axis=0)

# Un valore è un GUASTO se è un errore isolato O un blackout totale di rete
is_fault = is_zero & (is_isolated_fault | is_global_blackout.values[:, None])

# Sostituiamo con NaN SOLO i veri guasti/mancanze
df_cleaned = df.copy()
df_cleaned[is_fault] = np.nan

# 3. Creazione Metadati Temporali SEPARATI (senza inquinare df)
dow = df_cleaned.index.dayofweek
time_slot = df_cleaned.index.hour * 12 + df_cleaned.index.minute // 5
meta_df = pd.DataFrame({'dow': dow, 'time_slot': time_slot}, index=df_cleaned.index)

# 4. Divisione Train per evitare DATA LEAKAGE
train_ratio = 0.70
train_end = int(len(df_cleaned) * train_ratio)

df_train = df_cleaned.iloc[:train_end]
meta_train = meta_df.iloc[:train_end]

# Calcolo medie storiche SOLO sui dati di Training
train_grouped = df_train.groupby([meta_train['dow'], meta_train['time_slot']]).mean()

#  Mappatura ed Imputazione sul Dataset Completo
mapped_means = meta_df.merge(
    train_grouped,
    on=['dow', 'time_slot'],
    how='left'
).set_index(df_cleaned.index)[sensor_cols]

# Imputazione con Media Storica
df_imputed = df_cleaned.fillna(mapped_means)

# Interpolazione Temporale per micro-gap o slot vuoti residui
df = df_imputed.interpolate(method='time').ffill().bfill()

print("Numero di NaN residui:", df.isna().sum().sum())

#SPLITTING
n = len(df)
train_end = int(n * 0.70)
val_end = int(n * 0.80)

x_train = df.iloc[:train_end]
x_val   = df.iloc[train_end:val_end]
x_test  = df.iloc[val_end:]

# Z-SCORE GLOBALE
mean = x_train.values.mean()
std  = x_train.values.std()
x_train_norm = (x_train - mean) / std
x_val_norm   = (x_val - mean) / std
x_test_norm  = (x_test - mean) / std

print(f"Global Mean (Train): {mean:.4f}")
print(f"Global Std (Train):  {std:.4f}")


#ANALISI FINESTRE SCROREVOLI
def sliding_windows(data, window_in=12, window_out=12):
    #windows_in: valori nel passato
    #windows_out: valori nel futuro
    if hasattr(data, 'values'):
        data = data.values

    X, Y = [], []
    num_samples = len(data) - window_in - window_out + 1

    for i in range(num_samples):
        X.append(data[i: i + window_in])
        Y.append(data[i + window_in: i + window_in + window_out])

    return np.array(X), np.array(Y)


# Applicazione separata sui set già normalizzati
X_train, Y_train = sliding_windows(x_train_norm)
X_val, Y_val = sliding_windows(x_val_norm)
X_test, Y_test = sliding_windows(x_test_norm)

#se vogliamo cambiare il time step e passare da 12 a 24, ci conviene lavorare sul passato:
#windows_in=24, windows_out=12

