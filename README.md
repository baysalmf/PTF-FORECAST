# PTF FORECAST

Makine öğrenmesi yöntemi (LightGBM) ile gün öncesi elektrik piyasası saatlik PTF tahmini (EPİAŞ verisi + Excel tahmin girdisi).

## Kurulum

```bash
cd PTF-FORECAST
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Kimlik bilgisi

EPİAŞ kullanıcı adı ve şifreyi **koda yazmayın**. Ortam değişkeni veya proje kökünde `.env` kullanın:

1. `.env.example` dosyasını `.env` olarak kopyalayın.
2. `EPIAS_USERNAME` ve `EPIAS_PASSWORD` alanlarını doldurun.

`.env` dosyası git’e eklenmez (`.gitignore`).

## Çalıştırma

### Saatlik tahmin (`PROJE/ptf_prediction.py`)

Varsayılanlar: masaüstünde `forecast.xlsx` girdi, `ptf_tahmin.xlsx` çıktı.

```bash
python PROJE/ptf_prediction.py --help
python PROJE/ptf_prediction.py --no-plots
python PROJE/ptf_prediction.py --forecast-input "C:\path\to\forecast.xlsx" --output "C:\path\to\ptf_tahmin.xlsx"
```

- `--lgbm-device cpu|gpu|auto` — GPU yoksa `cpu` veya `auto` kullanın (varsayılan: `cpu`).
- `LGBM_DEVICE` ortam değişkeni ile de ayarlanabilir.

### Aylık senaryo (`PROJE/ptf_prediction_monthly.py`)

```bash
python PROJE/ptf_prediction_monthly.py --input "...\2026-2027 PTF.xlsx" --output "...\PTF_Scenario_Output.xlsx"
```

## Notlar

- Excel okuma/yazma için `openpyxl` kullanılır.
- API yanıt şeması değişirse birleştirme kolonları (`date` vb.) güncellenmelidir.
