"""Previsao temporal: primeira coluna = timestamp; ultima = target.
Teste de um passo: usa observacoes anteriores reais. Futuro recursivo:
colunas auxiliares mantidas no ultimo valor conhecido, sem dados futuros.
Dependencias: numpy pandas scikit-learn matplotlib (xgboost/lightgbm opcionais).
"""
import argparse
import time
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor, HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
RANDOM_STATE = 42
TIMINGS = []
LAGS = [1, 2, 3, 4, 6, 12, 26, 52]
ROLLING_WINDOWS = [3, 6, 12, 26, 52]
def format_duration(seconds):
    return f"{seconds / 60:.2f} min"

def load_series(path):
    raw = pd.read_csv(path, sep=None, engine="python")
    if raw.shape[1] < 2:
        raise ValueError("CSV deve ter pelo menos timestamp e target.")
    names = list(raw.columns)
    data = pd.DataFrame({"timestamp": pd.to_datetime(raw.iloc[:, 0], errors="raise"),
                         "target": pd.to_numeric(raw.iloc[:, -1], errors="coerce")})
    for i in range(1, len(names) - 1):
        data[f"aux_{i}"] = pd.to_numeric(raw.iloc[:, i], errors="raise")
    data = data.sort_values("timestamp").reset_index(drop=True)
    if data.timestamp.isna().any() or data.timestamp.duplicated().any():
        raise ValueError("Timestamp ausente ou duplicado.")
    if len(data) < 3:
        raise ValueError("Serie insuficiente.")
    freq = pd.infer_freq(data.timestamp)
    if freq is None:
        raise ValueError("Datas irregulares: regularize a serie antes de executar.")
    data = data.replace([np.inf, -np.inf], np.nan)
    print(f"Timestamp: {names[0]} | Target: {names[-1]} | Frequencia: {freq}")
    print(f"Linhas: {len(data)} | Targets ausentes: {data.target.isna().sum()}")
    return data, freq, names[-1]

def add_time_features(df):
    out = df[["timestamp", "target"]].copy()
    dates = out.timestamp
    out["t"] = np.arange(len(out))
    out["ano"] = dates.dt.year
    out["sin_ano"] = np.sin(2 * np.pi * dates.dt.dayofyear / 365.25)
    out["cos_ano"] = np.cos(2 * np.pi * dates.dt.dayofyear / 365.25)
    past = df.target.shift(1)
    for lag in LAGS:
        out[f"lag_{lag}"] = df.target.shift(lag)
    for window in ROLLING_WINDOWS:
        roll = past.rolling(window)
        for name in ("mean", "std", "min", "max"):
            out[f"roll_{name}_{window}"] = getattr(roll, name)()
    out["diff_1"] = past.diff()
    out["diff_52"] = past.diff(52)
    for col in df.columns:
        if col.startswith("aux_"):
            # Preenchimento exclusivamente com valores anteriores.
            out[f"{col}_lag1"] = df[col].ffill().shift(1)
    return out

def get_feature_columns(df):
    return [c for c in df.columns if c not in {"timestamp", "target"}]

def build_models(model_names=None):
    models = {
        "RIDGE": Pipeline([
            ("scaler", StandardScaler()),
            ("model", Ridge(alpha=1.0))
        ]),
        "ELASTICNET": Pipeline([
            ("scaler", StandardScaler()),
            ("model", ElasticNet(alpha=0.05, l1_ratio=0.3, max_iter=20000, random_state=RANDOM_STATE))
        ]),
        "RF": RandomForestRegressor(
            n_estimators=500,
            min_samples_leaf=2,
            random_state=RANDOM_STATE,
            n_jobs=-1
        ),
        "EXTRATREES": ExtraTreesRegressor(
            n_estimators=500,
            min_samples_leaf=1,
            random_state=RANDOM_STATE,
            n_jobs=-1
        ),
        "GBR": GradientBoostingRegressor(
            n_estimators=400,
            learning_rate=0.03,
            max_depth=3,
            random_state=RANDOM_STATE
        ),
        "HISTGBR": HistGradientBoostingRegressor(
            max_iter=400,
            learning_rate=0.03,
            max_leaf_nodes=15,
            random_state=RANDOM_STATE
        ),
    }

    try:
        from xgboost import XGBRegressor
        models["XGBOOST"] = XGBRegressor(
            n_estimators=500,
            learning_rate=0.03,
            max_depth=3,
            subsample=0.9,
            colsample_bytree=0.9,
            objective="reg:squarederror",
            random_state=RANDOM_STATE,
            n_jobs=-1
        )
    except Exception as e:
        print(f"XGBoost nao disponivel. Pulando. Motivo: {e}")

    try:
        from lightgbm import LGBMRegressor
        models["LIGHTGBM"] = LGBMRegressor(
            n_estimators=300,
            learning_rate=0.03,
            min_data_in_leaf=2,
            min_data_in_bin=1,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            verbose=-1
        )
    except Exception as e:
        print(f"LightGBM nao disponivel. Pulando. Motivo: {e}")

    if model_names is not None:
        requested = [m.strip().upper() for m in model_names if m.strip()]
        models = {name: model for name, model in models.items() if name in requested}
        missing = sorted(set(requested) - set(models.keys()))
        if missing:
            print(f"Modelos solicitados nao disponiveis e serao ignorados: {missing}")

    return models


# ============================================================
# METRICAS
# ============================================================

def smape(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denom = (np.abs(y_true) + np.abs(y_pred)) / 2
    mask = denom != 0
    if mask.sum() == 0:
        return np.nan
    return np.mean(np.abs(y_true[mask] - y_pred[mask]) / denom[mask]) * 100


def mape_safe(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = y_true != 0
    if mask.sum() == 0:
        return np.nan
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100


def evaluate_model(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return {
        "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
        "MAE": mean_absolute_error(y_true, y_pred),
        "MAPE_percent": mape_safe(y_true, y_pred),
        "SMAPE_percent": smape(y_true, y_pred),
        "R2": r2_score(y_true, y_pred),
    }


# ============================================================
# TREINO E TESTE
# ============================================================

def train_and_evaluate(feature_df, test_months=52, model_names=None):
    clean = feature_df.copy()
    clean["timestamp"] = pd.to_datetime(clean["timestamp"])
    clean = clean.dropna(subset=["target"] + get_feature_columns(clean)).reset_index(drop=True)

    if len(clean) <= test_months + 6:
        raise ValueError(
            f"Serie curta demais apos criacao dos lags. Linhas uteis: {len(clean)}. "
            f"Reduza --test-steps ou reduza LAGS."
        )

    feature_cols = get_feature_columns(clean)
    train = clean.iloc[:-test_months].copy()
    test = clean.iloc[-test_months:].copy()

    X_train = train[feature_cols].astype(float).to_numpy()
    y_train = train["target"].astype(float).to_numpy()
    X_test = test[feature_cols].astype(float).to_numpy()
    y_test = test["target"].astype(float).to_numpy()

    models = build_models(model_names=model_names)
    if not models:
        raise RuntimeError("Nenhum modelo disponivel para treinamento.")

    metrics_rows = []
    predictions_test = test[["timestamp", "target"]].copy()
    fitted_models = {}
    failed_models = []
    model_times = []

    for name, model in models.items():
        t0 = time.perf_counter()
        print(f"Treinando {name}...")
        try:
            model.fit(X_train, y_train)
            pred = model.predict(X_test)
            pred = np.asarray(pred, dtype=float)
            pred = pred

            elapsed = time.perf_counter() - t0
            model_times.append({"modelo": name, "tempo_segundos": elapsed, "tempo_formatado": format_duration(elapsed)})
            TIMINGS.append({"etapa": f"Treinamento {name}", "tempo_segundos": elapsed, "tempo_formatado": format_duration(elapsed)})

            fitted_models[name] = model
            predictions_test[f"pred_{name}"] = pred

            met = evaluate_model(y_test, pred)
            met["modelo"] = name
            met["tempo_treino_teste_segundos"] = elapsed
            met["tempo_treino_teste"] = format_duration(elapsed)
            metrics_rows.append(met)
            print(f"  -> OK | Tempo: {format_duration(elapsed)}")

        except Exception as e:
            elapsed = time.perf_counter() - t0
            failed_models.append({"modelo": name, "erro": str(e), "tempo_segundos": elapsed})
            TIMINGS.append({"etapa": f"Falha treinamento {name}", "tempo_segundos": elapsed, "tempo_formatado": format_duration(elapsed)})
            print(f"Modelo {name} falhou e sera ignorado. Erro: {e}")

    if len(metrics_rows) == 0:
        raise RuntimeError("Nenhum modelo treinou corretamente.")

    metrics = pd.DataFrame(metrics_rows)
    metrics = metrics[["modelo", "RMSE", "MAE", "MAPE_percent", "SMAPE_percent", "R2", "tempo_treino_teste"]]
    metrics = metrics.sort_values("RMSE").reset_index(drop=True)

    return (
        fitted_models,
        metrics,
        predictions_test,
        feature_cols,
        clean,
        pd.DataFrame(failed_models),
        pd.DataFrame(model_times),
    )



def recursive_forecast(model, history, columns, steps, freq):
    hist = history.copy()
    if hist.target.isna().any():
        raise ValueError("Historico futuro deve ter target preenchido com previsoes causais.")
    offset = pd.tseries.frequencies.to_offset(freq)
    rows = []
    for step in range(1, steps + 1):
        row = {"timestamp": hist.timestamp.iloc[-1] + offset, "target": np.nan}
        for col in hist.columns:
            if col.startswith("aux_"):
                row[col] = hist[col].ffill().iloc[-1]
        hist = pd.concat([hist, pd.DataFrame([row])], ignore_index=True)
        x = add_time_features(hist).iloc[[-1]][columns]
        if x.isna().any().any():
            raise ValueError("Historico insuficiente para previsao futura.")
        pred = float(model.predict(x.to_numpy(dtype=float))[0])
        hist.loc[hist.index[-1], "target"] = pred
        rows.append({"timestamp": row["timestamp"], "horizonte_passos": step, "previsao_target": pred})
    return pd.DataFrame(rows)

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-file", default="milho_semanal.csv")
    parser.add_argument("--output-dir", default="outputs_milho_semanal")
    parser.add_argument("--test-steps", type=int, default=52, help="Ultimas observacoes; 52 semanas na base milho.")
    parser.add_argument("--future-steps", type=int, default=52)
    parser.add_argument("--models", default="ALL", help="ALL ou nomes separados por virgula.")
    args = parser.parse_args()
    if args.test_steps < 2 or args.future_steps < 1:
        parser.error("test-steps >= 2 e future-steps >= 1.")
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    history, freq, target_name = load_series(args.data_file)
    # Targets ausentes: somente propagacao passada, nunca interpolacao com futuro.
    history["target"] = history.target.ffill()
    features = add_time_features(history)
    # Targets preenchidos ajudam os lags, mas nao sao rotulos para treino/teste.
    original, _, _ = load_series(args.data_file)
    features.loc[original.target.isna(), "target"] = np.nan
    split = len(features) - args.test_steps
    if split <= max(LAGS) + 6:
        raise ValueError("Treino insuficiente: reduza --test-steps.")
    # Fixar fronteira nas linhas originais, antes de remover ausentes.
    test_count = len(features.iloc[split:].dropna())
    if test_count < 2:
        raise ValueError("Menos de dois targets validos no teste.")
    requested = None if args.models.upper() == "ALL" else args.models.upper().split(",")
    fitted, metrics, predictions, cols, clean, failed, times = train_and_evaluate(features, test_count, requested)
    best_name = metrics.iloc[0]["modelo"]
    best = fitted[best_name]
    metrics.to_csv(out / "metricas_modelos.csv", index=False)
    predictions.rename(columns={"target": target_name}).to_csv(out / "previsoes_teste.csv", index=False)
    times.to_csv(out / "tempos_por_modelo.csv", index=False)
    if not failed.empty:
        failed.to_csv(out / "modelos_com_erro.csv", index=False)
    best.fit(clean[cols].to_numpy(dtype=float), clean.target.to_numpy(dtype=float))
    future = recursive_forecast(best, history, cols, args.future_steps, freq)
    future.to_csv(out / "previsao_futura.csv", index=False)
    for filename, title, plot_data in [
        ("serie_historica.png", "Serie historica", [(history.timestamp, original.target, "Real")]),
        ("teste_real_vs_previsto.png", f"Teste de um passo: {best_name}", [(predictions.timestamp, predictions.target, "Real"), (predictions.timestamp, predictions[f"pred_{best_name}"], "Previsto")]),
        ("previsao_futura.png", f"Previsao recursiva: {best_name}", [(history.timestamp, original.target, "Historico"), (future.timestamp, future.previsao_target, "Futuro")])]:
        plt.figure(figsize=(14, 6))
        for dates, values, label in plot_data:
            plt.plot(dates, values, label=label)
        plt.title(title)
        plt.xlabel("Data")
        plt.ylabel(target_name)
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(out / filename, dpi=180)
        plt.close()
    print(metrics.to_string(index=False))
    print(f"Melhor RMSE no teste: {best_name}. Saidas: {out.resolve()}")
    print("Teste de um passo com historico real; futuro recursivo com auxiliares constantes.")

if __name__ == "__main__":
    main()
