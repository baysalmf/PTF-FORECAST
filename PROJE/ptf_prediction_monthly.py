import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import ElasticNet
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

df = pd.read_excel(r"C:\Users\muhammetfb\OneDrive\Masaüstü\2026-2027 PTF.xlsx")

train_df = df[df["PTF"].notna()].copy()
future_df = df[df["PTF"].isna()].copy()

X_cols = [c for c in df.columns if c not in ["Tarih", "PTF"]]

X_train = train_df[X_cols]
y_train = train_df["PTF"]
X_future = future_df[X_cols]

model = Pipeline([
    ("scaler", StandardScaler()),
    ("enet", ElasticNet(alpha=0.1, l1_ratio=0.5, random_state=42))
])

model.fit(X_train, y_train)

coef_df = pd.DataFrame({
    "Variable": X_cols,
    "Coefficient": model.named_steps["enet"].coef_
}).sort_values(by="Coefficient", key=np.abs, ascending=False)

print("\n===== ELASTIC NET KATSAYILARI =====")
print(coef_df)

base = X_future.copy()

high = X_future.copy()
for col in ["TTF", "Kur", "Imported Coal", "Natural Gas"]:
    if col in high.columns:
        high[col] *= 1.10
for col in ["Reservoir Hydro", "Run-of-River"]:
    if col in high.columns:
        high[col] *= 0.90

low = X_future.copy()
for col in ["TTF", "Kur", "Imported Coal", "Natural Gas"]:
    if col in low.columns:
        low[col] *= 0.90
for col in ["Reservoir Hydro", "Run-of-River"]:
    if col in low.columns:
        low[col] *= 1.10

future_out = future_df[["Tarih"]].copy()
future_out["PTF_Low"] = model.predict(low)
future_out["PTF_Base"] = model.predict(base)
future_out["PTF_High"] = model.predict(high)

print("\n===== SENARYO TAHMİNLERİ (İLK SATIRLAR) =====")
print(future_out.head())

plt.figure()
plt.plot(future_out["Tarih"], future_out["PTF_Base"], label="Base")
plt.fill_between(
    future_out["Tarih"],
    future_out["PTF_Low"],
    future_out["PTF_High"],
    alpha=0.3,
    label="Low–High Band"
)
plt.xlabel("Tarih")
plt.ylabel("PTF (TL/MWh)")
plt.title("2026–2027 Aylık PTF Tahmini (Senaryo Bandı)")
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

output_path = r"C:\Users\muhammetfb\OneDrive\Masaüstü\PTF_Scenario_Output.xlsx"
future_out.to_excel(output_path, index=False)

print(f"\n✅ Senaryo Excel'i oluşturuldu:\n{output_path}")
