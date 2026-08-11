import gc
import warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb

from dataset import load_and_preprocess_data, LazyTimeSeriesDataset, get_valid_indices, SEQ_LEN, PRED_LEN
from models import SPINN_Engine, GRU_Baseline, physics_loss_kernel

warnings.filterwarnings('ignore')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BATCH_SIZE = 1024

def build_flat_subset(data_2d, indices):
    X = np.empty((len(indices), SEQ_LEN * 4), dtype=np.float32)
    y = np.empty((len(indices), PRED_LEN * 4), dtype=np.float32)
    for k, i in enumerate(indices):
        X[k] = data_2d[i : i + SEQ_LEN].flatten()
        y[k] = data_2d[i + SEQ_LEN : i + SEQ_LEN + PRED_LEN].flatten()
    return X, y

def chunked_eval(nn_model, data_2d, indices):
    nn_model.eval()
    preds, targets = [], []
    chunk_size = 2000
    with torch.no_grad():
        for start in range(0, len(indices), chunk_size):
            end = min(start + chunk_size, len(indices))
            chunk_idx = indices[start:end]
            X_chunk = np.empty((len(chunk_idx), SEQ_LEN, 4), dtype=np.float32)
            y_chunk = np.empty((len(chunk_idx), PRED_LEN, 4), dtype=np.float32)
            for k, i in enumerate(chunk_idx):
                X_chunk[k] = data_2d[i : i + SEQ_LEN]
                y_chunk[k] = data_2d[i + SEQ_LEN : i + SEQ_LEN + PRED_LEN]
                
            t_in = torch.from_numpy(X_chunk).to(device)
            preds.append(nn_model(t_in).cpu().numpy())
            targets.append(y_chunk)
            
    return np.vstack(preds), np.vstack(targets)

def chunked_xgb_eval(model, data_2d, indices):
    preds = []
    chunk_size = 5000
    for start in range(0, len(indices), chunk_size):
        end = min(start + chunk_size, len(indices))
        X_flat, _ = build_flat_subset(data_2d, indices[start:end])
        pred_flat = model.predict(X_flat)
        preds.append(pred_flat.reshape(-1, PRED_LEN, 4))
    return np.vstack(preds)

def main():
    print(f"Executing pipeline on device: {device}")
    train_s, test_cl_s, test_sp_s, test_od_s = load_and_preprocess_data()

    idx_tr = get_valid_indices(len(train_s))
    idx_cl = get_valid_indices(len(test_cl_s))
    idx_sp = get_valid_indices(len(test_sp_s))
    idx_od = get_valid_indices(len(test_od_s))

    train_loader = DataLoader(LazyTimeSeriesDataset(train_s, idx_tr), batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

    # Model Init
    snn_model = SPINN_Engine().to(device)
    gru_model = GRU_Baseline().to(device)

    opt_snn = torch.optim.AdamW(snn_model.parameters(), lr=2e-3, weight_decay=1e-4)
    opt_gru = torch.optim.Adam(gru_model.parameters(), lr=2e-3)
    criterion = nn.MSELoss()

    print("\n--- Training Neural Frameworks ---")
    for epoch in range(1, 4):
        snn_model.train()
        total_loss = 0
        for bx, by in train_loader:
            bx, by = bx.to(device), by.to(device)
            
            # SNN-PINODE Step
            opt_snn.zero_grad()
            p_snn = snn_model(bx)
            loss = criterion(p_snn, by) + 0.05 * physics_loss_kernel(p_snn, device)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(snn_model.parameters(), max_norm=1.0)
            opt_snn.step()
            
            # GRU Step
            opt_gru.zero_grad()
            l_gru = criterion(gru_model(bx), by)
            l_gru.backward()
            opt_gru.step()
            
            total_loss += loss.item()
            
        print(f"Epoch [{epoch}/3] | Loss: {total_loss/len(train_loader):.4f}")
        gc.collect()

    print("\n--- Training Sub-Sampled XGBoost Baseline ---")
    np.random.seed(42)
    sub_idx = np.random.choice(idx_tr, size=min(25000, len(idx_tr)), replace=False)
    X_tr_xgb, y_tr_xgb = build_flat_subset(train_s, sub_idx)
    xgb_reg = xgb.XGBRegressor(n_estimators=30, max_depth=4, n_jobs=-1)
    xgb_reg.fit(X_tr_xgb, y_tr_xgb)
    del X_tr_xgb, y_tr_xgb
    gc.collect()

    print("\n--- Running Safe Evaluation ---")
    p_pin_cl, y_cl = chunked_eval(snn_model, test_cl_s, idx_cl)
    p_pin_sp, y_sp = chunked_eval(snn_model, test_sp_s, idx_sp)
    p_pin_od, y_od = chunked_eval(snn_model, test_od_s, idx_od)

    p_gru_cl, _ = chunked_eval(gru_model, test_cl_s, idx_cl)
    p_gru_sp, _ = chunked_eval(gru_model, test_sp_s, idx_sp)
    p_gru_od, _ = chunked_eval(gru_model, test_od_s, idx_od)

    p_xgb_cl = chunked_xgb_eval(xgb_reg, test_cl_s, idx_cl)
    p_xgb_sp = chunked_xgb_eval(xgb_reg, test_sp_s, idx_sp)
    p_xgb_od = chunked_xgb_eval(xgb_reg, test_od_s, idx_od)

    # Reporting
    frameworks = ['SPINN', 'GRU', 'XGBoost']
    scenarios = ['CLEAN', 'SPARSE', 'OOD']
    preds = {
        'SPINN': (p_pin_cl, p_pin_sp, p_pin_od),
        'GRU': (p_gru_cl, p_gru_sp, p_gru_od),
        'XGBoost': (p_xgb_cl, p_xgb_sp, p_xgb_od)
    }
    targets = {'CLEAN': y_cl, 'SPARSE': y_sp, 'OOD': y_od}

    r2_summary, rmse_summary, mae_summary = [], [], []

    for sc in scenarios:
        r2_row, rmse_row, mae_row = {'Condition': sc}, {'Condition': sc}, {'Condition': sc}
        yt = targets[sc].flatten()
        for name in frameworks:
            yp = preds[name][scenarios.index(sc)].flatten()
            r2_row[name] = max(0.0, r2_score(yt, yp))
            rmse_row[name] = np.sqrt(mean_squared_error(yt, yp))
            mae_row[name] = mean_absolute_error(yt, yp)
            
        r2_summary.append(r2_row)
        rmse_summary.append(rmse_row)
        mae_summary.append(mae_row)

    print("\n" + "="*50)
    print("  COMPREHENSIVE EXPERIMENTAL RESULTS")
    print("="*50)
    print("\n--- R² SCORE ---")
    print(pd.DataFrame(r2_summary).to_string(index=False, float_format="%.4f"))
    print("\n--- RMSE ---")
    print(pd.DataFrame(rmse_summary).to_string(index=False, float_format="%.4f"))
    print("\n--- MAE ---")
    print(pd.DataFrame(mae_summary).to_string(index=False, float_format="%.4f"))

if __name__ == '__main__':
    main()
