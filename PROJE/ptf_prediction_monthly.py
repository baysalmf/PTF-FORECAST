"""
Aylık PTF senaryo tahmini (ElasticNet + ölçeklendirme).
Yollar ve parametreler CLI veya ortam değişkeni ile verilir.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import ElasticNet
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None


def _project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def load_env_optional() -> None:
    if load_dotenv is None:
        return
    env_path = _project_root() / ".env"
    if env_path.is_file():
        load_dotenv(env_path)


def parse_args() -> argparse.Namespace:
    desktop = Path.home() / "Desktop"
    default_in = desktop / "2026-2027 PTF.xlsx"
    default_out = desktop / "PTF_Scenario_Output.xlsx"

    p = argparse.ArgumentParser(description="Aylık PTF senaryo tahmini (ElasticNet).")
    p.add_argument(
        "--input",
        type=Path,
        default=Path(os.environ.get("PTF_MONTHLY_INPUT", str(default_in))),
        help="Girdi Excel (Tarih, PTF, açıklayıcı kolonlar)",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=Path(os.environ.get("PTF_MONTHLY_OUTPUT", str(default_out))),
        help="Çıktı Excel yolu",
    )
    p.add_argument("--alpha", type=float, default=0.1, help="ElasticNet alpha")
    p.add_argument("--l1-ratio", type=float, default=0.5, dest="l1_ratio", help="ElasticNet l1_ratio")
    p.add_argument(
        "--no-plots",
        action="store_true",
        help="Grafik gösterme",
    )
    return p.parse_args()


def apply_scenario_high(df: pd.DataFrame) -> pd.DataFrame:
    high = df.copy()
    for col in ["TTF", "Kur", "Imported Coal", "Natural Gas"]:
        if col in high.columns:
            high[col] *= 1.10
    for col in ["Reservoir Hydro", "Run-of-River"]:
        if col in high.columns:
            high[col] *= 0.90
    return high


def apply_scenario_low(df: pd.DataFrame) -> pd.DataFrame:
    low = df.copy()
    for col in ["TTF", "Kur", "Imported Coal", "Natural Gas"]:
        if col in low.columns:
            low[col] *= 0.90
    for col in ["Reservoir Hydro", "Run-of-River"]:
        if col in low.columns:
            low[col] *= 1.10
    return low


def main() -> int:
    load_env_optional()
    args = parse_args()

    input_path = args.input.expanduser().resolve()
    if not input_path.is_file():
        raise FileNotFoundError(
            f"Girdi dosyası bulunamadı: {input_path}\n"
            "--input ile yolu verin veya masaüstüne Excel'i koyun."
        )

    df = pd.read_excel(input_path)

    train_df = df[df["PTF"].notna()].copy()
    future_df = df[df["PTF"].isna()].copy()

    X_cols = [c for c in df.columns if c not in ["Tarih", "PTF"]]

    if train_df.empty:
        raise ValueError("Eğitim verisi yok: 'PTF' dolu satır bulunamadı.")
    if future_df.empty:
        raise ValueError("Tahmin edilecek satır yok: 'PTF' boş (NaN) satır bulunamadı.")

    X_train = train_df[X_cols]
    y_train = train_df["PTF"]
    X_future = future_df[X_cols]

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("enet", ElasticNet(alpha=args.alpha, l1_ratio=args.l1_ratio, random_state=42)),
    ])

    model.fit(X_train, y_train)

    coef_df = pd.DataFrame({
        "Variable": X_cols,
        "Coefficient": model.named_steps["enet"].coef_,
    }).sort_values(by="Coefficient", key=np.abs, ascending=False)

    print("\n===== ELASTIC NET KATSAYILARI =====")
    print(coef_df)

    base = X_future.copy()
    high = apply_scenario_high(X_future)
    low = apply_scenario_low(X_future)

    future_out = future_df[["Tarih"]].copy()
    future_out["PTF_Low"] = model.predict(low)
    future_out["PTF_Base"] = model.predict(base)
    future_out["PTF_High"] = model.predict(high)

    print("\n===== SENARYO TAHMİNLERİ (İLK SATIRLAR) =====")
    print(future_out.head())

    if not args.no_plots:
        plt.figure()
        plt.plot(future_out["Tarih"], future_out["PTF_Base"], label="Base")
        plt.fill_between(
            future_out["Tarih"],
            future_out["PTF_Low"],
            future_out["PTF_High"],
            alpha=0.3,
            label="Low–High Band",
        )
        plt.xlabel("Tarih")
        plt.ylabel("PTF (TL/MWh)")
        plt.title("2026–2027 Aylık PTF Tahmini (Senaryo Bandı)")
        plt.legend()
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

    output_path = args.output.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    future_out.to_excel(output_path, index=False)

    print(f"\n✅ Senaryo Excel'i oluşturuldu:\n{output_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
