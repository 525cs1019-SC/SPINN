import os
import gc
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import RobustScaler

SEQ_LEN = 24
PRED_LEN = 6
NUM_COLS = ['SLP', 'WND_RATE', 'TMP', 'DEW']

def load_and_preprocess_data(data_path=None):
    if data_path and os.path.exists(data_path):
        df_raw = pd.read_parquet(data_path)
        df_raw['DATE'] = pd.to_datetime(df_raw['DATE'])
    else:
        date_range = pd.date_range(start='2020-01-01', end='2026-12-31', freq='h')
        np.random.seed(42)
        df_raw = pd.DataFrame({
            'DATE': date_range,
            'SLP': 1010.0 + np.sin(np.arange(len(date_range))/100)*10.0 + np.random.normal(0, 2, len(date_range)),
            'WND_RATE': 15.0 + np.cos(np.arange(len(date_range))/200)*8.0 + np.random.normal(0, 3, len(date_range)),
            'TMP': 28.0 + np.sin(np.arange(len(date_range))/500)*5.0 + np.random.normal(0, 1, len(date_range)),
            'DEW': 22.0 + np.cos(np.arange(len(date_range))/500)*4.0 + np.random.normal(0, 1, len(date_range))
        })

    df_raw = df_raw.sort_values('DATE').reset_index(drop=True)
    df_raw[NUM_COLS] = df_raw[NUM_COLS].astype(np.float32)

    # Partitioning
    split_idx = int(len(df_raw) * 0.8)
    df_train = df_raw.iloc[:split_idx].copy()
    df_test_clean = df_raw.iloc[split_idx:].copy()

    # Sparse condition (25% missing data)
    df_test_sparse = df_test_clean.copy()
    mask = np.random.rand(*df_test_sparse[NUM_COLS].shape) < 0.25
    df_test_sparse[NUM_COLS] = df_test_sparse[NUM_COLS].where(~mask, np.nan).ffill().bfill()

    # OOD condition (synthetic extremes)
    df_test_ood = df_test_clean.copy()
    test_dates = df_test_ood['DATE'].values
    cyc_indices = [int(len(test_dates) * 0.3), int(len(test_dates) * 0.7)]
    for idx in cyc_indices:
        center = pd.to_datetime(test_dates[idx])
        c_mask = (df_test_ood['DATE'] >= center - pd.Timedelta(days=2)) & (df_test_ood['DATE'] <= center + pd.Timedelta(days=3))
        df_test_ood.loc[c_mask, 'SLP'] -= np.random.uniform(35.0, 50.0)
        df_test_ood.loc[c_mask, 'WND_RATE'] += np.random.uniform(40.0, 60.0)
        df_test_ood.loc[c_mask, 'TMP'] -= np.random.uniform(4.0, 6.0)

    scaler = RobustScaler()
    train_scaled = scaler.fit_transform(df_train[NUM_COLS].values).astype(np.float32)
    test_cl_scaled = scaler.transform(df_test_clean[NUM_COLS].values).astype(np.float32)
    test_sp_scaled = scaler.transform(df_test_sparse[NUM_COLS].values).astype(np.float32)
    test_od_scaled = scaler.transform(df_test_ood[NUM_COLS].values).astype(np.float32)

    del df_raw, df_train, df_test_clean, df_test_sparse, df_test_ood
    gc.collect()

    return train_scaled, test_cl_scaled, test_sp_scaled, test_od_scaled

class LazyTimeSeriesDataset(Dataset):
    def __init__(self, data_2d, indices):
        self.data = torch.from_numpy(data_2d)
        self.idx = indices

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, k):
        i = self.idx[k]
        return self.data[i : i + SEQ_LEN], self.data[i + SEQ_LEN : i + SEQ_LEN + PRED_LEN]

def get_valid_indices(data_len):
    return np.arange(0, data_len - SEQ_LEN - PRED_LEN + 1)
