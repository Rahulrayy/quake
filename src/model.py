import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
import pickle

# tunable parameters
BATCH_SIZE    = 256
LEARNING_RATE = 1e-3
EPOCHS        = 1000
PATIENCE      = 10
HIDDEN_DIM    = 128
DROPOUT       = 0.1
PENALTY_ALPHA = 0.5

FEATURE_COLS = [
    'euclid_dist', 'height_diff', 'horiz_dist', 'height_ratio',
    'dx', 'dy', 'dz',
    'src_density', 'src_degree', 'src_avg_edge', 'src_max_edge',
    'goal_density', 'goal_degree', 'goal_avg_edge', 'goal_max_edge',
    'density_ratio', 'degree_ratio'
]

BASE_DIR = Path(__file__).parent.parent


class PathDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class HeuristicNet(nn.Module):
    def __init__(self, input_dim, hidden_dim=HIDDEN_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(DROPOUT),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(DROPOUT),
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        # kaiming init works well with relu activations
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.net(x).squeeze(-1)

    def predict_cf(self, x):
        # returns correction factor clamped to >= 1.0 for admissibility
        with torch.no_grad():
            log_cf = self.forward(x)
        return torch.clamp(torch.exp(log_cf), min=1.0)

    def predict_single(self, features):
        x  = torch.tensor(features, dtype=torch.float32).unsqueeze(0)
        cf = self.predict_cf(x)
        return float(cf.item())


def admissibility_loss(pred, true, alpha=PENALTY_ALPHA):
    mse     = F.mse_loss(pred, true)
    # penalise predictions below 0 (cf < 1.0 is impossible)
    penalty = torch.clamp(-pred, min=0).mean()
    return mse + alpha * penalty


def train_model(train_loader, val_loader, input_dim):
    checkpoints = BASE_DIR / 'checkpoints'
    checkpoints.mkdir(exist_ok=True)
    model_path  = checkpoints / 'best_model.pt'

    model     = HeuristicNet(input_dim=input_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

    best_val_loss    = float('inf')
    patience_counter = 0
    history          = {'train': [], 'val': []}

    for epoch in range(EPOCHS):
        # training pass
        model.train()
        train_losses = []
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            pred = model(X_batch)
            loss = admissibility_loss(pred, y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_losses.append(loss.item())

        # validation pass
        model.eval()
        val_losses = []
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                pred = model(X_batch)
                loss = F.mse_loss(pred, y_batch)
                val_losses.append(loss.item())

        train_loss = np.mean(train_losses)
        val_loss   = np.mean(val_losses)
        scheduler.step(val_loss)
        history['train'].append(train_loss)
        history['val'].append(val_loss)

        if (epoch + 1) % 10 == 0:
            print(f"  epoch {epoch+1:3d} — train: {train_loss:.4f}  val: {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss    = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), model_path)
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"  early stopping at epoch {epoch+1}")
                break

    model.load_state_dict(torch.load(model_path))
    return model, history


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    features_path = BASE_DIR / "data" / "features.parquet"
    checkpoints   = BASE_DIR / "checkpoints"
    checkpoints.mkdir(exist_ok=True)

    print("loading features")
    df = pd.read_parquet(features_path)
    print(f"  {len(df)} samples, {len(FEATURE_COLS)} features")

    train_maps = [m for m in df['map_name'].unique() if m.startswith('e1') or m.startswith('e2')]
    val_maps   = [m for m in df['map_name'].unique() if m.startswith('e3')]
    test_maps  = [m for m in df['map_name'].unique() if m.startswith('e4') or m.startswith('dm') or m in ['start', 'end']]

    train_df = df[df['map_name'].isin(train_maps)]
    val_df   = df[df['map_name'].isin(val_maps)]
    test_df  = df[df['map_name'].isin(test_maps)]

    print(f"\ntrain: {len(train_df)} samples ({len(train_maps)} maps)")
    print(f"val:   {len(val_df)} samples ({len(val_maps)} maps)")
    print(f"test:  {len(test_df)} samples ({len(test_maps)} maps)")

    # fit scaler on train only to avoid data leakage
    scaler  = StandardScaler()
    X_train = scaler.fit_transform(train_df[FEATURE_COLS].values)
    X_val   = scaler.transform(val_df[FEATURE_COLS].values)
    X_test  = scaler.transform(test_df[FEATURE_COLS].values)

    y_train = train_df['log_cf'].values.astype(np.float32)
    y_val   = val_df['log_cf'].values.astype(np.float32)
    y_test  = test_df['log_cf'].values.astype(np.float32)

    # save scaler for inference time
    with open(checkpoints / 'scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    print(f"\n saved")

    train_loader = DataLoader(PathDataset(X_train, y_train), batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(PathDataset(X_val,   y_val),   batch_size=BATCH_SIZE)
    test_loader  = DataLoader(PathDataset(X_test,  y_test),  batch_size=BATCH_SIZE)

    print(f"\ntraining")
    model, history = train_model(train_loader, val_loader, input_dim=len(FEATURE_COLS))

    # evaluate on test set
    model.eval()
    test_losses = []
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            pred = model(X_batch)
            test_losses.append(F.mse_loss(pred, y_batch).item())
    print(f"\ntest MSE {np.mean(test_losses):.4f}")

    # plot training curves
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(history['train'], label='train loss')
    ax.plot(history['val'],   label='val loss')
    ax.set_xlabel('epoch')
    ax.set_ylabel('MSE loss')
    ax.set_title('training curves')
    ax.legend()
    plt.tight_layout()
    out = BASE_DIR / "plots" / "training_curves.png"
    plt.savefig(out, dpi=150)
    print(f"saved: {out}")
    plt.show()