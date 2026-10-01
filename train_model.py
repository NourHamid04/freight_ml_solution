from pathlib import Path
import argparse
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor


def make_features(df: pd.DataFrame, base_only: bool = False) -> pd.DataFrame:
    x = df.drop(columns=["posted_rate"], errors="ignore").copy()
    d = pd.to_datetime(x["date"])
    x["year"] = d.dt.year
    x["month"] = d.dt.month
    x["day"] = d.dt.day
    x["dayofweek"] = d.dt.dayofweek
    x["dayofyear"] = d.dt.dayofyear
    x["weekofyear"] = d.dt.isocalendar().week.astype(int)
    x["month_sin"] = np.sin(2 * np.pi * x["month"] / 12)
    x["month_cos"] = np.cos(2 * np.pi * x["month"] / 12)
    x["dow_sin"] = np.sin(2 * np.pi * x["dayofweek"] / 7)
    x["dow_cos"] = np.cos(2 * np.pi * x["dayofweek"] / 7)
    x["route"] = x["pickup"].astype(str) + "__" + x["delivery"].astype(str)
    x["distance2"] = x["distance"] ** 2
    x["weight_distance"] = x["weight"] * x["distance"]
    x = x.drop(columns=["date", "load_id"], errors="ignore")

    if base_only:
        allowed = [
            "pickup", "delivery", "distance", "equipment", "weight",
            "year", "month", "day", "dayofweek", "dayofyear", "weekofyear",
            "month_sin", "month_cos", "dow_sin", "dow_cos",
            "route", "distance2", "weight_distance",
        ]
        x = x[[c for c in allowed if c in x.columns]]
    return x


def configs():
    return [
        dict(depth=6, learning_rate=0.05, l2_leaf_reg=5, random_strength=0.5, random_seed=11, iterations=100),
        dict(depth=7, learning_rate=0.05, l2_leaf_reg=5, random_strength=0.5, random_seed=22, iterations=90),
        dict(depth=8, learning_rate=0.03, l2_leaf_reg=8, random_strength=0.2, random_seed=33, iterations=140),
        dict(depth=9, learning_rate=0.04, l2_leaf_reg=8, random_strength=0.2, random_seed=44, iterations=180),
    ]


def fit_ensemble(X, y, cat_cols):
    models = []
    for cfg in configs():
        model = CatBoostRegressor(
            loss_function="RMSE",
            verbose=False,
            allow_writing_files=False,
            **cfg,
        )
        model.fit(X, y, cat_features=cat_cols, verbose=False)
        models.append(model)
    return models


def predict_rate(models, X, distance):
    p = [m.predict(X) * distance for m in models]
    return np.maximum(np.mean(p, axis=0), 1.0)


def benchmark(train: pd.DataFrame):
    dates = pd.to_datetime(train["date"])
    fit_mask = dates < pd.Timestamp("2025-09-01")
    val_mask = ~fit_mask

    X = make_features(train)
    cat_cols = [c for c in X.columns if X[c].dtype == "object"]
    y_rpm = train["posted_rate"].to_numpy() / train["distance"].to_numpy()

    models = fit_ensemble(X.loc[fit_mask], y_rpm[fit_mask], cat_cols)
    pred = predict_rate(models, X.loc[val_mask], train.loc[val_mask, "distance"].to_numpy())
    actual = train.loc[val_mask, "posted_rate"].to_numpy()

    err = actual - pred
    rmse = float(np.sqrt(np.mean(err ** 2)))
    mae = float(np.mean(np.abs(err)))
    medae = float(np.median(np.abs(err)))
    ss_res = float(np.sum(err ** 2))
    ss_tot = float(np.sum((actual - actual.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot
    print(f"Chronological validation (Sep-Oct 2025): RMSE={rmse:.2f}, MAE={mae:.2f}, MedianAE={medae:.2f}, R2={r2:.4f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", default="data/train-test.csv")
    parser.add_argument("--validation", default="data/validation.csv")
    parser.add_argument("--output", default="validation_predictions.csv")
    parser.add_argument("--december-output", default="data/december_chart_inputs.csv")
    args = parser.parse_args()

    train = pd.read_csv(args.train)
    validation = pd.read_csv(args.validation)

    benchmark(train)

    X = make_features(train)
    Xv = make_features(validation)
    cat_cols = [c for c in X.columns if X[c].dtype == "object"]
    y_rpm = train["posted_rate"].to_numpy() / train["distance"].to_numpy()

    models = fit_ensemble(X, y_rpm, cat_cols)
    pred = predict_rate(models, Xv, validation["distance"].to_numpy())
    pd.DataFrame({
        "load_id": validation["load_id"].astype(str),
        "predicted_rate": np.round(pred, 2),
    }).to_csv(args.output, index=False)

    # The supplied score.py checks a 7-column fixed December schema. Build a
    # chart-only model from the subset of features available in that schema.
    Xb = make_features(train, base_only=True)
    catb = [c for c in Xb.columns if Xb[c].dtype == "object"]
    chart_model = CatBoostRegressor(
        iterations=180, depth=8, learning_rate=0.04, l2_leaf_reg=8,
        random_strength=0.3, random_seed=42, loss_function="RMSE",
        verbose=False, allow_writing_files=False,
    )
    chart_model.fit(Xb, y_rpm, cat_features=catb, verbose=False)

    december = pd.DataFrame({
        "pickup": ["Lexington"] * 31,
        "delivery": ["Fort Wayne"] * 31,
        "distance": [360.0] * 31,
        "equipment": ["Dry Van"] * 31,
        "weight": [32000.0] * 31,
        "date": pd.date_range("2025-12-01", "2025-12-31", freq="D").strftime("%Y-%m-%d"),
    })
    dx = make_features(december, base_only=True)
    december["predicted_rate"] = np.round(np.maximum(chart_model.predict(dx) * 360.0, 1.0), 2)
    december.to_csv(args.december_output, index=False)

    print(f"Saved {args.output}")
    print(f"Saved {args.december_output}")


if __name__ == "__main__":
    main()
