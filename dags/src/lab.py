import pickle
import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

_SRC_DIR  = os.path.dirname(__file__)
DATA_PATH = os.path.join(_SRC_DIR, '..', 'data', 'stock_data.csv')
MODEL_DIR = os.path.join(_SRC_DIR, '..', 'model')


# ── Task 1 ────────────────────────────────────────────────────────────────────
def generate_data():
    """
    Synthetically generates OHLCV data for 200 fictional stocks over 252 trading days.
    Each stock belongs to one of 4 hidden market regimes (growth, defensive, volatile, low-vol).
    Derived features: daily_return, volatility, avg_volume, price_range are used for clustering.
    """
    np.random.seed(42)
    n_stocks, n_days = 200, 252

    # 4 hidden regimes with different return/volatility profiles
    regimes = {
        'growth':    {'mu': 0.001,  'sigma': 0.012, 'vol_scale': 1.2},
        'defensive': {'mu': 0.0003, 'sigma': 0.006, 'vol_scale': 0.8},
        'volatile':  {'mu': 0.0005, 'sigma': 0.025, 'vol_scale': 2.0},
        'low_vol':   {'mu': 0.0002, 'sigma': 0.004, 'vol_scale': 0.5},
    }

    records = []
    regime_names = list(regimes.keys())
    for i in range(n_stocks):
        reg = regime_names[i % 4]
        p   = regimes[reg]
        # Simulate daily close prices via geometric Brownian motion
        returns  = np.random.normal(p['mu'], p['sigma'], n_days)
        prices   = 100 * np.exp(np.cumsum(returns))
        highs    = prices * (1 + np.abs(np.random.normal(0, 0.005, n_days)))
        lows     = prices * (1 - np.abs(np.random.normal(0, 0.005, n_days)))
        volume   = np.random.normal(1e6 * p['vol_scale'], 1e5, n_days).clip(1e4)

        records.append({
            'stock_id':      f'STOCK_{i:03d}',
            'daily_return':  round(float(np.mean(returns)), 6),
            'volatility':    round(float(np.std(returns)), 6),
            'avg_volume':    round(float(np.mean(volume)), 2),
            'price_range':   round(float(np.mean(highs - lows)), 4),
            'final_price':   round(float(prices[-1]), 2),
            'sharpe_approx': round(float(np.mean(returns) / (np.std(returns) + 1e-9)), 4),
        })

    df = pd.DataFrame(records)
    os.makedirs(os.path.dirname(DATA_PATH), exist_ok=True)
    df.to_csv(DATA_PATH, index=False)
    print(f"[generate_data] Synthetic OHLCV dataset saved → {df.shape[0]} stocks, {df.shape[1]} features")
    return pickle.dumps(df)


# ── Task 2 ────────────────────────────────────────────────────────────────────
def load_data():
    df = pd.read_csv(DATA_PATH)
    print(f"[load_data] Loaded: {df.shape}")
    return pickle.dumps(df)


# ── Task 3 ────────────────────────────────────────────────────────────────────
def data_preprocessing(data):
    """Drop non-numeric stock_id and standardise all feature columns."""
    df = pickle.loads(data)
    features = ['daily_return', 'volatility', 'avg_volume', 'price_range', 'sharpe_approx']
    X = StandardScaler().fit_transform(df[features])
    print(f"[data_preprocessing] Scaled shape: {X.shape}  features: {features}")
    return pickle.dumps(X)


# ── Task 4 ────────────────────────────────────────────────────────────────────
def build_save_model(data, filename):
    """
    Train KMeans for k = 2..8, select best k via Silhouette Score.
    Saves the best model to disk and returns scores dict.
    """
    X = pickle.loads(data)
    os.makedirs(MODEL_DIR, exist_ok=True)

    best_k, best_score, best_model = 2, -1, None
    sil_scores, sse_values = {}, {}

    for k in range(2, 9):
        km     = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(X)
        sil    = silhouette_score(X, labels)
        sil_scores[k] = round(sil, 6)
        sse_values[k] = round(km.inertia_, 4)
        print(f"  k={k}  SSE={km.inertia_:.2f}  Silhouette={sil:.4f}")
        if sil > best_score:
            best_score, best_k, best_model = sil, k, km

    model_path = os.path.join(MODEL_DIR, filename)
    with open(model_path, 'wb') as f:
        pickle.dump(best_model, f)

    print(f"[build_save_model] Best k={best_k} (silhouette={best_score:.4f}) → saved to {model_path}")
    return pickle.dumps({'best_k': best_k, 'sil_scores': sil_scores, 'sse_values': sse_values})


# ── Task 5 ────────────────────────────────────────────────────────────────────
def load_model_summary(filename, results_data):
    """
    Load best model, print a market-regime cluster dashboard:
    score table + avg return/volatility per cluster + ASCII size bars.
    """
    model_path = os.path.join(MODEL_DIR, filename)
    with open(model_path, 'rb') as f:
        model = pickle.load(f)

    results    = pickle.loads(results_data)
    best_k     = results['best_k']
    sil_scores = results['sil_scores']
    sse_values = results['sse_values']

    df       = pd.read_csv(DATA_PATH)
    features = ['daily_return', 'volatility', 'avg_volume', 'price_range', 'sharpe_approx']
    X        = StandardScaler().fit_transform(df[features])
    df['cluster'] = model.predict(X)

    W = 60
    print("\n" + "=" * W)
    print("   STOCK MARKET CLUSTER DASHBOARD  –  Synthetic OHLCV")
    print("=" * W)
    print(f"  Optimal k            : {best_k}")
    print(f"  Best Silhouette Score: {sil_scores[best_k]:.4f}  (range: -1 → +1)")

    print(f"\n  {'k':<4} {'SSE':>12}  {'Silhouette':>12}  Note")
    print("  " + "-" * 50)
    for k in sorted(sil_scores):
        note = "<-- BEST" if k == best_k else ""
        print(f"  {k:<4} {sse_values[k]:>12.2f}  {sil_scores[k]:>12.4f}  {note}")

    print(f"\n  Per-cluster market regime profile:")
    print(f"  {'Cluster':<10} {'Count':>6}  {'Avg Return':>12}  {'Avg Volatility':>16}  {'Avg Sharpe':>12}")
    print("  " + "-" * 62)
    for c in sorted(df['cluster'].unique()):
        sub = df[df['cluster'] == c]
        print(f"  {c:<10} {len(sub):>6}  {sub['daily_return'].mean():>12.6f}"
              f"  {sub['volatility'].mean():>16.6f}  {sub['sharpe_approx'].mean():>12.4f}")

    print(f"\n  Cluster size bars (each # ≈ 2 stocks):")
    for c in sorted(df['cluster'].unique()):
        cnt = len(df[df['cluster'] == c])
        bar = "#" * (cnt // 2)
        print(f"    Cluster {c}: {cnt:>4} stocks  {bar}")

    print("=" * W + "\n")

    summary = f"Optimal clusters: {best_k}, Silhouette: {sil_scores[best_k]:.4f}"
    print(summary)
    return summary