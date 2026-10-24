GitHub Repository: [https://github.com/wilber830915/AIoT-HW2-MultipleLR](https://github.com/wilber830915/AIoT-HW2-MultipleLR)

Live Demo: [https://wilber830915.github.io/AIoT-HW2-MultipleLR/](https://wilber830915.github.io/AIoT-HW2-MultipleLR/)

![多元線性迴歸與特徵選擇探索器 網頁截圖](L5_HW2_MultipleLR_index.png)

---

# Multiple Linear Regression · Feature Selection Explorer

本專案將「多元線性迴歸 + 特徵選取」的完整 CRISP-DM 流程做成可互動的靜態網頁。頁面提供兩個 Problem：

- **Flood Prediction**：預測 `FloodProbability`
- **Automobile Dataset**：預測汽車 `price`

使用者可切換資料集、特徵選取方法與特徵數 `k`，即時查看 Validation/Test 指標、特徵清單、OLS 係數、殘差診斷、95% 預測區間與即時預測結果。

---

## 1. 專案特色

- 5 種特徵選取方法：SelectKBest、RFE、Sequential Forward Selection、Lasso、Permutation Importance
- 每個 Problem 各 4 種 `k`，共 20 組實驗
- 以 **1-SE（one-standard-error）規則**選模，避免只看最小 Validation RMSE 而永遠選最多特徵
- 使用 **split conformal prediction** 建立 95% 預測區間
- 純前端互動：Chart.js + 匯出的 JSON 模型資料，不需要後端伺服器
- 可部署到 GitHub Pages

---

## 2. 資料集

### 2.1 Flood Prediction

- 來源：[Kaggle Playground S4E5（Flood Prediction）](https://www.kaggle.com/datasets/johnwicknice/playground-series)
- 目標變數：`FloodProbability`
- 訓練資料：1,117,957 筆
- 外部測試資料：745,305 筆
- 原始特徵：20 個數值特徵
- 互動特徵：10 組領域導向的兩兩相乘特徵
- 候選特徵總數：30

互動特徵例如：

- `MonsoonIntensity × ClimateChange`
- `MonsoonIntensity × DrainageSystems`
- `ClimateChange × Deforestation`
- `WetlandLoss × CoastalVulnerability`

### 2.2 Automobile Dataset

- 來源：[Kaggle Automobile Dataset](https://www.kaggle.com/datasets/toramky/automobile-dataset)
- 目標變數：`price`（USD）
- 原始資料：205 筆
- 移除 `price` 缺失值後：201 筆
- 數值特徵：15 個
- 類別特徵：10 個
- 類別特徵以 one-hot encoding（`drop="first"`）處理
- 在 fit split 上實際產生 63 個候選特徵
- 完整資料重新配適時可能出現 64 個欄位，差異來自驗證/測試集出現訓練集沒有的類別；`handle_unknown="ignore"` 會將其編碼為全零

類別特徵包含 `make`、`fuel-type`、`body-style`、`drive-wheels`、`engine-type`、`num-of-cylinders`、`fuel-system` 等。

---

## 3. 專案結構

```text
L5_HW2_MultipleLR/
├── index.html                  # 互動式網頁
├── app.js                      # 前端邏輯：切換、圖表、預測、係數表
├── model_data.js               # Flood 實驗與模型資料（window.FLOOD_DATA）
├── automobile_model_data.js    # Automobile 實驗與模型資料（window.AUTOMOBILE_DATA）
├── main.py                     # 完整 CRISP-DM 流程與輸出
├── export_web_data.py          # Flood 實驗並匯出 model_data.js
├── export_auto_data.py         # Automobile 實驗並匯出 automobile_model_data.js
├── data/
│   ├── train.csv               # Flood 訓練資料
│   ├── test.csv                # Flood 外部測試資料
│   └── Automobile_data.csv     # Automobile 原始資料
├── outputs/                    # main.py 產出的模型、圖表、submission
└── README.md
```

---

## 4. 執行方式

### 4.1 安裝套件

本專案使用 Python 3.10，主要套件如下：

```bash
pip install numpy pandas scipy scikit-learn statsmodels matplotlib seaborn joblib
```

開發時使用的版本：

- Python 3.10.9
- scikit-learn 1.2.1
- pandas 1.5.3
- numpy 1.23.5
- scipy 1.10.0
- statsmodels 0.13.5
- matplotlib 3.7.0

### 4.2 完整 Flood 流程

```bash
python3 main.py
```

會重新產生 `outputs/` 下的模型、診斷圖、`submission.csv` 與 `best_features.json`。

### 4.3 重新匯出網頁資料

```bash
python3 export_web_data.py
python3 export_auto_data.py
```

- `export_web_data.py`  Flood 的 5 × 4 實驗，輸出 `model_data.js`
- `export_auto_data.py` Automobile 的 5 × 4 實驗，輸出 `automobile_model_data.js`

### 4.4 本機預覽

```bash
python3 -m http.server 8000
```

開啟：

- `http://localhost:8000/`
- `http://localhost:8000/?dataset=flood`
- `http://localhost:8000/?dataset=automobile`

### 4.5 部署

將 `index.html`、`app.js`、`model_data.js`、`automobile_model_data.js` 放到任何靜態網站空間即可，例如 GitHub Pages。頁面不需要後端。

---

## 5. 建模流程

### 5.1 資料切分

Flood：

- Fit：670,773
- Validation：167,694
- Calibration：111,796
- Test：167,694

Automobile：

- Fit：120
- Validation：30
- Calibration：20
- Test：31

Calibration set 只用於 conformal prediction，不參與模型訓練。

### 5.2 前處理

Flood：

- 中位數補值
- StandardScaler 標準化

Automobile：

- 數值特徵：中位數補值
- 類別特徵：眾數補值 + OneHotEncoder（`drop="first"`、`handle_unknown="ignore"`）
- 全部特徵再標準化

### 5.3 特徵選取方法

| 方法 | 說明 |
|---|---|
| SelectKBest | 以 `f_regression` 分數選出前 k 個特徵 |
| RFE | 遞迴特徵消除，每次移除最不重要特徵 |
| Sequential | Sequential Forward Selection，從空集合逐步加入特徵 |
| Lasso | 以 `LassoCV` 係數絕對值排序，選出前 k 個 |
| Permutation | 以 `permutation_importance` 的平均重要性排序，選出前 k 個 |

Flood 的 `k`：`[14, 20, 25, 30]`  
Automobile 的 `k`：`[8, 16, 24, 32]`

### 5.4 使用 1-SE 規則

1. 找到 Validation RMSE 最小的模型，計算該 RMSE 的標準誤：

```text
SE(RMSE) = std(residual²) / (2 × RMSE × sqrt(n_validation))
```

2. 門檻 = 最小 RMSE + 1 × SE。
3. 在門檻內的模型中，選擇特徵數最少者。
4. 若特徵數相同，選 Validation RMSE 較低者；再相同則依方法順序。

在誤差幾乎相同時偏好更精簡的模型，而不是無條件選擇最大 k。

### 5.5 預測區間

使用 split conformal prediction：

- 以 calibration set 的絕對殘差計算 `q_hat`
- 預測區間：`prediction ± q_hat`
- 名目覆蓋率：95%

---

## 6. 結果

### 6.1 Flood Prediction

最小 Validation RMSE：

- 方法：SelectKBest
- `k = 30`
- Validation RMSE：`0.020087766`
- Validation SE：`0.000043670`

1-SE 最佳設定：

- 方法：RFE
- `k = 20`
- 特徵：20 個原始特徵（不使用互動特徵）
- Validation RMSE：`0.020089362`
- Validation R²：`0.845568`
- Test RMSE：`0.020082935`
- Test MAE：`0.015803254`
- Test R²：`0.844850`
- 95% PI coverage：`0.950636`
- `q_hat`：`0.040165`
- 平均區間寬度：`0.080330`

各方法的 1-SE 最佳 k：

| 方法 | 最小 RMSE 的 k | 1-SE 選擇的 k |
|---|---:|---:|
| SelectKBest | 30 | 30 |
| RFE | 30 | 20 |
| Sequential | 30 | 30 |
| Lasso | 30 | 20 |
| Permutation | 30 | 20 |

### 6.2 Automobile Dataset

最小 Validation RMSE：

- 方法：RFE
- `k = 32`
- Validation RMSE：`1661.96`
- Validation SE：`217.37`

1-SE 最佳設定：

- 方法：Sequential
- `k = 16`
- 特徵：16 個（5 個數值 + 11 個 one-hot 類別特徵）
- Validation RMSE：`1742.84`
- Validation R²：`0.920729`
- Test RMSE：`4887.99`
- Test MAE：`3543.53`
- Test R²：`0.830968`
- 95% PI coverage：`0.935484`
- `q_hat`：`10583.80`
- 平均區間寬度：`21167.61`

各方法的 1-SE 最佳 k：

| 方法 | 最小 RMSE 的 k | 1-SE 選擇的 k |
|---|---:|---:|
| SelectKBest | 32 | 16 |
| RFE | 32 | 16 |
| Sequential | 32 | 16 |
| Lasso | 24 | 8 |
| Permutation | 32 | 24 |

### 6.3 選用的 Automobile 特徵

Sequential `k=16` 選出：

- `width`
- `height`
- `engine-size`
- `stroke`
- `horsepower`
- `make = isuzu`
- `make = mercedes-benz`
- `make = saab`
- `make = subaru`
- `make = toyota`
- `make = volkswagen`
- `drive-wheels = fwd`
- `engine-type = ohc`
- `engine-type = ohcv`
- `num-of-cylinders = four`
- `fuel-system = 2bbl`

---

## 7. 網頁功能

### 7.1 資料集切換

Hero 區的「選擇 Problem」下拉選單可切換：

- Flood Prediction
- Automobile Dataset

切換後會同步更新：

- 標題、資料筆數、特徵數
- 最佳方法與最佳 k
- 所有圖表
- 實驗總表
- 即時預測器
- CRISP-DM 說明

### 7.2 實驗比較

- 可切換 5 種方法或 `ALL`
- 可切換 `k`
- 綠色標記為該方法的 1-SE 最佳 k
- 圖表可點擊資料點切換 `k`

### 7.3 即時預測

Flood：

- 20 個數值輸入
- 自動計算 10 個互動特徵
- 使用目前選定模型的原始尺度 OLS 係數

Automobile：

- 15 個數值輸入
- 10 個類別下拉選單
- 類別特徵即時 one-hot
- 顯示預測價格與 95% 預測區間

### 7.4 模型診斷

- 特徵清單
- OLS 迴歸方程式
- 係數、標準誤、t 值、p 值、95% CI、VIF
- Predicted vs Actual
- Residual plot
- 預測區間圖
- 誤差最大的測試樣本
- 20 組實驗總表

---

## 8. CRISP-DM 對應

| 階段 | 專案內容 |
|---|---|
| Business Understanding | 洪水機率與汽車價格預測 |
| Data Understanding | 資料載入、缺失值檢查、目標分布 |
| Data Preparation | 補值、標準化、互動特徵、one-hot |
| Modeling | 5 種特徵選取 × 4 種 k，OLS 配適 |
| Evaluation | Validation RMSE、1-SE 選模、Test RMSE/MAE/R²、conformal PI |
| Deployment | 匯出靜態 JSON、互動網頁、即時預測 |

---

## 9. 主要輸出

`main.py` 會在 `outputs/` 產生：

- `best_features.json`
- `best_model.joblib`
- `experiment_config.json`
- `feature_selection_comparison.csv`
- `feature_selection_comparison.png`
- `prediction_interval.png`
- `prediction_interval_metrics.csv`
- `regression_diagnostics.png`
- `submission.csv`
- `submission_model.joblib`
- `target_distribution.png`
- `test_metrics.csv`

`export_web_data.py` 與 `export_auto_data.py` 會產生：

- `model_data.js`
- `automobile_model_data.js`

---

## 10. 限制與注意事項

- Automobile 只有 201 筆有效資料，Test set 只有 31 筆，指標容易受單一資料切分影響。
- Automobile 的 95% PI 很寬（`q_hat ≈ 10,584`），反映小樣本與價格變異大。
- one-hot 類別中有些類別樣本數很少；驗證/測試集若出現訓練集沒有的類別，會被編碼為全零。
- 1-SE 規則是「在誤差接近最小時偏好精簡模型」的啟發式規則，不保證 Test 一定更好，但能降低過度依賴單一 Validation 切分的風險。
- Flood 的資料接近線性，完整 30 特徵模型的 Validation RMSE 仍然略低；1-SE 規則是為了在幾乎相同誤差下選擇更精簡的 20 特徵模型。

---

## 11. 參考資料

- [Kaggle Playground S4E5（Flood Prediction）](https://www.kaggle.com/datasets/johnwicknice/playground-series)
- [Kaggle Automobile Dataset](https://www.kaggle.com/datasets/toramky/automobile-dataset)
- [Chart.js](https://www.chartjs.org/)
- [scikit-learn Feature Selection](https://scikit-learn.org/stable/modules/feature_selection.html)
- [scikit-learn Conformal Prediction](https://scikit-learn.org/stable/auto_examples/model_selection/plot_conformal_prediction.html)
