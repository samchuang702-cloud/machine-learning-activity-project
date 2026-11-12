# Spotify 音樂情緒分類｜機器學習模型比較與集成學習


以 Spotify 2023 音樂特徵資料建立歌曲正向情緒（Valence）分類模型，完整比較基準模型、降維、超參數調校與 Soft Voting 集成學習的表現。

## 專案簡介

音樂平台擁有大量歌曲，但僅依靠曲名或曲風，難以快速判斷一首歌帶給聽眾的情緒傾向。本專案利用歌曲的節奏、能量、舞動性與聲學特徵，預測歌曲屬於 **High Valence** 或 **Low Valence**，展示從資料清理、特徵工程、模型訓練到評估比較的完整機器學習流程。

Valence 代表歌曲所傳達的正向、愉悅程度。本專案以資料集中 `valence_%` 的中位數作為分類門檻：

- **High Valence**：較正向、明亮或愉悅
- **Low Valence**：較低沉、平靜或負向

## 想解決的問題

本專案聚焦以下問題：

- 能否只使用音訊特徵辨識歌曲的情緒傾向？
- PCA 與 LDA 降維是否能提升 Logistic Regression 的分類能力？
- 超參數調校能否改善基準模型？
- 結合不同演算法的 Soft Voting，是否能得到更穩定的預測結果？

此類方法可延伸應用於情境歌單推薦、音樂內容標記、推薦系統特徵建構與曲庫分析。

## 使用特徵

模型使用以下 9 項歌曲特徵：

| 類型 | 特徵 |
| --- | --- |
| 數值特徵 | BPM、Danceability、Energy、Acousticness、Instrumentalness、Liveness、Speechiness |
| 類別特徵 | Key、Mode |

資料前處理包含：

- 數值缺失值以中位數補值
- 數值特徵標準化
- 類別缺失值補值
- 類別特徵 One-Hot Encoding
- 分層切分訓練集與測試集
- 透過 Scikit-learn Pipeline 將前處理與模型封裝，降低資料洩漏風險

## 實驗設計

| 實驗 | 方法 | 目的 |
| --- | --- | --- |
| 01 | Logistic Regression | 建立可解釋的基準模型 |
| 02 | PCA + Logistic Regression | 評估非監督式降維效果 |
| 03 | LDA + Logistic Regression | 評估監督式降維效果 |
| 04 | GridSearchCV + Logistic Regression | 使用五折交叉驗證搜尋超參數 |
| 05 | Soft Voting Ensemble | 整合 LR、SVM、KNN 與 Decision Tree |

所有實驗採用相同設定：

- `test_size=0.2`
- `random_state=42`
- Stratified train/test split
- Precision、Recall 與 F1 採 Macro Average
- AUC 以 High Valence 為正類

## 模型成果

| 模型 | Accuracy | Precision | Recall | F1 Score | AUC |
| --- | ---: | ---: | ---: | ---: | ---: |
| Baseline Logistic Regression | 0.7696 | 0.7740 | 0.7678 | 0.7678 | 0.8308 |
| PCA (13) + Logistic Regression | 0.7539 | 0.7586 | 0.7520 | 0.7517 | 0.8202 |
| LDA (1) + Logistic Regression | 0.7696 | 0.7740 | 0.7678 | 0.7678 | **0.8341** |
| Logistic Regression + GridSearchCV | 0.7644 | 0.7679 | 0.7627 | 0.7627 | 0.8305 |
| Soft Voting Ensemble | **0.7801** | **0.7884** | **0.7778** | **0.7774** | 0.8286 |

### 結果觀察

- **Soft Voting Ensemble** 取得最高 Accuracy、Precision、Recall 與 F1 Score，顯示整合不同模型能改善整體分類表現。
- **LDA + Logistic Regression** 取得最高 AUC，代表監督式降維對類別排序能力有所幫助。
- PCA 並未優於基準模型，說明降低維度不一定能保留最具判別力的資訊。
- 超參數調校的結果未超越基準模型，反映交叉驗證最佳參數不保證在單一測試集上取得最高分。

## 技術亮點

- 建立可重複使用的資料載入、前處理及評估模組
- 使用 `ColumnTransformer` 同時處理數值與類別欄位
- 使用 `Pipeline` 確保訓練與推論採用一致的資料處理流程
- 以 `GridSearchCV` 進行五折交叉驗證與超參數搜尋
- 比較 PCA 與 LDA 兩種不同性質的降維方法
- 使用 Soft Voting 結合線性模型、核方法、鄰近法與決策樹
- 同時使用 Accuracy、Precision、Recall、F1 與 AUC，避免只以單一指標判斷模型

## 使用技術

- Python
- Pandas
- NumPy
- Scikit-learn
- Logistic Regression
- PCA / LDA
- GridSearchCV
- SVM / KNN / Decision Tree
- Soft Voting Ensemble

## 專案結構

| 檔案 | 說明 |
| --- | --- |
| `common_valence_utils.py` | 資料載入、前處理、切分、AUC 與評估工具 |
| `01_baseline_logistic_regression.py` | Logistic Regression 基準模型 |
| `02_pca_logistic_regression.py` | PCA 降維實驗 |
| `03_lda_logistic_regression.py` | LDA 降維實驗 |
| `04_logistic_regression_hyperparameter_tuning.py` | GridSearchCV 超參數調校 |
| `05_soft_voting_ensemble.py` | Soft Voting 集成模型 |
| `spotify-2023.csv` | Spotify 2023 音樂特徵資料 |

## 執行方式

### 1. 下載專案

~~~bash
git clone https://github.com/samchuang702-cloud/machine-learning-activity-project.git
cd machine-learning-activity-project
~~~

### 2. 安裝套件

~~~bash
pip install pandas numpy scikit-learn
~~~

### 3. 執行實驗

~~~bash
python 01_baseline_logistic_regression.py
python 02_pca_logistic_regression.py
python 03_lda_logistic_regression.py
python 04_logistic_regression_hyperparameter_tuning.py
python 05_soft_voting_ensemble.py
~~~

每支程式都會自動讀取專案目錄中的 `spotify-2023.csv`，並在終端輸出模型評估結果。

## 未來改進方向

- 加入混淆矩陣、ROC Curve 與特徵重要性視覺化
- 使用 Nested Cross-Validation 提高模型比較的可靠度
- 嘗試 Random Forest、XGBoost 或 LightGBM
- 處理類別閾值選擇與機率校準
- 將最佳模型封裝成 API 或互動式音樂情緒預測工具

## 專案價值

本專案不只呈現單一模型結果，也展示如何在一致的資料切分與評估標準下，系統性比較基準模型、特徵降維、超參數調校及集成學習。這套流程可延伸至其他二元分類問題，並作為後續部署與產品化的基礎。
