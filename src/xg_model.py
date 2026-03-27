import numpy as np
import pandas as pd
import pickle
import xgboost as xgb
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

BASE_DIR = Path(__file__).parent.parent

FEATURE_COLS = [
    'euclid_dist', 'height_diff', 'horiz_dist', 'height_ratio',
    'dx', 'dy', 'dz',
    'src_density', 'src_degree', 'src_avg_edge', 'src_max_edge',
    'goal_density', 'goal_degree', 'goal_avg_edge', 'goal_max_edge',
    'density_ratio', 'degree_ratio'
]

# tunable parameters
N_ESTIMATORS  = 1000    # number of trees, raise for more accuracy
MAX_DEPTH     = 8      # tree depth, raise for more complex patterns
LEARNING_RATE = 0.1    # shrinkage, lower = slower but more accurate
SUBSAMPLE     = 0.8    # fraction of samples per tree
MIN_CHILD_W   = 5      # minimum samples in a leaf node


if __name__ == "__main__":
    features_path = BASE_DIR / "data" / "features.parquet"
    checkpoints   = BASE_DIR / "checkpoints"
    checkpoints.mkdir(exist_ok=True)


    df = pd.read_parquet(features_path)
    print(f"  {len(df)} samples, {len(FEATURE_COLS)} features")

    # same train/val/test split as the MLP - split by map not by sample
    train_maps = [m for m in df['map_name'].unique() if m.startswith('e1') or m.startswith('e2')]
    val_maps   = [m for m in df['map_name'].unique() if m.startswith('e3')]
    test_maps  = [m for m in df['map_name'].unique()
                  if m.startswith('e4') or m.startswith('dm') or m in ['start', 'end']]

    train_df = df[df['map_name'].isin(train_maps)]
    val_df   = df[df['map_name'].isin(val_maps)]
    test_df  = df[df['map_name'].isin(test_maps)]

    print(f"\ntrain: {len(train_df)} samples ({len(train_maps)} maps)")
    print(f"val:   {len(val_df)} samples ({len(val_maps)} maps)")
    print(f"test:  {len(test_df)} samples ({len(test_maps)} maps)")

    # fit scaler on train only
    scaler  = StandardScaler()
    X_train = scaler.fit_transform(train_df[FEATURE_COLS].values)
    X_val   = scaler.transform(val_df[FEATURE_COLS].values)
    X_test  = scaler.transform(test_df[FEATURE_COLS].values)

    y_train = train_df['log_cf'].values
    y_val   = val_df['log_cf'].values
    y_test  = test_df['log_cf'].values

    # save xgboost scaler separately so it doesn't overwrite the MLP scaler
    with open(checkpoints / 'xg_scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    print("\nxg scaler saved")

    # train xgboost with early stopping on val set
    print("\ntraining ")
    model = xgb.XGBRegressor(
        n_estimators      = N_ESTIMATORS,
        max_depth         = MAX_DEPTH,
        learning_rate     = LEARNING_RATE,
        subsample         = SUBSAMPLE,
        min_child_weight  = MIN_CHILD_W,
        tree_method       = 'hist',
        random_state      = 42,
        early_stopping_rounds = 20,
        eval_metric       = 'rmse',
        verbosity         = 1,
    )

    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=50  # print every 50 trees
    )

    # save model
    model.save_model(str(checkpoints / 'xg_model.json'))


    # evaluate
    val_preds  = model.predict(X_val)
    test_preds = model.predict(X_test)
    val_mse    = mean_squared_error(y_val,  val_preds)
    test_mse   = mean_squared_error(y_test, test_preds)

    print(f"\nval MSE  {val_mse:.4f}")
    print(f"test MSE {test_mse:.4f}")
    print(f"best iteration {model.best_iteration}")

    #  feature importance
    fig, ax = plt.subplots(figsize=(10, 6))
    importance = pd.Series(model.feature_importances_, index=FEATURE_COLS)
    importance.sort_values().plot(kind='barh', ax=ax, color='steelblue')
    ax.set_xlabel('feature importance (gain)')
    ax.set_title('xgboost feature importance')
    plt.tight_layout()
    plt.savefig(BASE_DIR / "plots" / "xgboost_feature_importance.png", dpi=150)
    print("\nsaved xgboost_feature_importance.png")
    plt.show()


    fig, ax = plt.subplots(figsize=(8, 8))
    cf_true = np.exp(y_test)
    cf_pred = np.exp(test_preds).clip(min=1.0)
    sample  = np.random.choice(len(cf_true), size=min(5000, len(cf_true)), replace=False)
    ax.scatter(cf_true[sample], cf_pred[sample], alpha=0.1, s=3, c='steelblue')
    max_val = min(cf_true.max(), 8)
    ax.plot([1, max_val], [1, max_val], 'r--', linewidth=1.5, label='perfect prediction')
    ax.set_xlabel('true correction factor')
    ax.set_ylabel('predicted correction factor')
    ax.set_title('xgboost predicted vs true correction factor\n(test maps only)')
    ax.set_xlim(1, max_val)
    ax.set_ylim(1, max_val)
    ax.legend()
    plt.tight_layout()
    plt.savefig(BASE_DIR / "plots" / "xgboost_predictions.png", dpi=150)
    print("saved: xgboost_predictions.png")
    plt.show()

    # training curve (val rmse over iterations)
    results = model.evals_result()
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(results['validation_0']['rmse'], color='steelblue', label='val rmse')
    ax.axvline(model.best_iteration, color='red', linestyle='--',
               linewidth=1, label=f'best iteration ({model.best_iteration})')
    ax.set_xlabel('iteration (number of trees)')
    ax.set_ylabel('rmse')
    ax.set_title('xgboost training curve')
    ax.legend()
    plt.tight_layout()
    plt.savefig(BASE_DIR / "plots" / "xgboost_training_curve.png", dpi=150)
    print("saved xgboost_training_curve.png")
    plt.show()


    """
    "C:\Users\rahul\OneDrive\Desktop\LEIDEN SEM 2\games\assignment 3\.venv\Scripts\python.exe" "C:\Users\rahul\OneDrive\Desktop\LEIDEN SEM 2\games\assignment 3\src\xg_model.py" 
  625740 samples, 17 features

train: 261090 samples (15 maps)
val:   127500 samples (7 maps)
test:  237150 samples (16 maps)

xg scaler saved

training 
[0]	validation_0-rmse:0.32623
[50]	validation_0-rmse:0.30348
[54]	validation_0-rmse:0.30361

val MSE  0.0919
test MSE 0.0582
best iteration 34

saved xgboost_feature_importance.png
saved: xgboost_predictions.png
saved xgboost_training_curve.png

Process finished with exit code 0

    """