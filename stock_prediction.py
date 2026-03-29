import yfinance as yf #Yahoo Finance 股價資料庫
import pandas as pd #資料處理
import numpy as np #數值計算
import matplotlib.pyplot as plt #資料視覺化
from sklearn.ensemble import RandomForestRegressor #隨機森林
from xgboost import XGBRegressor #XGBoost
from sklearn.metrics import mean_squared_error #評估指標    

# ==========================================
# 1. 資料抓取 (Data Fetching)
# ==========================================
print("Fetching S&P 500 data...") #顯示正在抓取資料
ticker = "^GSPC" #S&P 500 指數
start_date = "2021-01-01" #開始日期
end_date = "2026-01-01"  # yfinance end date 是 exclusive 的，所以我們設定為 2026-01-01 以包含 2025-12-31

# 抓取 S&P 500 資料
df = yf.download(ticker, start=start_date, end=end_date)

# yfinance 最新版本可能會回傳 MultiIndex columns，如果是的話則移除第二層 (Ticker層)
if isinstance(df.columns, pd.MultiIndex):
    df.columns = df.columns.droplevel(1)

# 印出剛抓下來的這張 pandas 資料表，看看它前 5 筆長什麼樣子
print("\n--- 剛下載好的 S&P 500 原始資料 (前 5 筆) ---")
print(df.head())
print("-" * 50 + "\n")

# ==========================================
# 2. 資料前處理與特徵工程 (Data Preprocessing)
# ==========================================
# 為了避免未來資料外洩 (look-ahead bias)，我們的目標 (Target) y 是「隔天的收盤價」
df['Target'] = df['Close'].shift(-1)

# 我們的特徵 X 使用當天的開盤價、最高價、最低價、收盤價、以及交易量
features = ['Open', 'High', 'Low', 'Close', 'Volume']

# 刪除因為 shift 產生的 NaN 值 (即最後一筆資料)
df.dropna(inplace=True)

# ==========================================
# 3. 資料切分 (Data Splitting) - 根據時間順序(嚴禁隨機打亂)
# ==========================================
# 訓練集 (Training Set): 2021-01-01 to 2024-12-31
# 測試集 (Testing Set): 2025-01-01 to 2025-12-31
train_mask = (df.index >= '2021-01-01') & (df.index <= '2024-12-31')
test_mask = (df.index >= '2025-01-01') & (df.index <= '2025-12-31')

train_data = df.loc[train_mask] #訓練集
test_data = df.loc[test_mask] #測試集

X_train = train_data[features] #訓練集特徵
y_train = train_data['Target'] #訓練集目標

X_test = test_data[features] #測試集特徵
y_test = test_data['Target'] #測試集目標

print(f"Training data size (Days): {len(X_train)}") #顯示訓練集大小
print(f"Testing data size (Days): {len(X_test)}") #顯示測試集大小

# ==========================================
# 4. 模型訓練 (Model Training)
# ==========================================
print("Training Random Forest...") #顯示正在訓練隨機森林
# 加大樹的數量到300、限制樹的深度最多長到8層、要求葉子至少要包含5筆資料
rf_model = RandomForestRegressor(n_estimators=100, max_depth=8, min_samples_leaf=5, random_state=42) #建立隨機森林模型
rf_model.fit(X_train, y_train) #訓練隨機森林模型

print("Training XGBoost...") #顯示正在訓練XGBoost
# 降低學習率、增加接棒的樹數量來互補、並保持樹不要太深
xgb_model = XGBRegressor(n_estimators=1000, learning_rate=0.01, max_depth=2, random_state=42)
xgb_model.fit(X_train, y_train) #訓練XGBoost模型

# ==========================================
# 5. 預測與評估 (Prediction and Evaluation)
# ==========================================
rf_preds = rf_model.predict(X_test) #隨機森林預測
xgb_preds = xgb_model.predict(X_test) #XGBoost預測

# 計算 MSE (Mean Squared Error)
rf_mse = mean_squared_error(y_test, rf_preds) #隨機森林MSE
xgb_mse = mean_squared_error(y_test, xgb_preds) #XGBoost MSE

print("\n" + "="*40) #顯示評估結果
print("--- 評估結果 (Evaluation Results) ---")
print(f"Random Forest MSE: {rf_mse:.2f}") #顯示隨機森林MSE
print(f"XGBoost MSE:       {xgb_mse:.2f}") #顯示XGBoost MSE
print("="*40 + "\n")

# ==========================================
# 6. 視覺化結果 (Visualization)
# ==========================================
plt.figure(figsize=(14, 7)) #建立圖表

# 畫出實際的價格
plt.plot(test_data.index, y_test, label='Actual Price (Target)', color='black', linewidth=2)

# 畫出兩種模型的預測結果
plt.plot(test_data.index, rf_preds, label='Random Forest Prediction', color='blue', alpha=0.7) #畫出隨機森林預測結果
plt.plot(test_data.index, xgb_preds, label='XGBoost Prediction', color='green', alpha=0.7) #畫出XGBoost預測結果

plt.title('S&P 500 Price Prediction: Actual vs. Random Forest vs. XGBoost (Testing Set: 2025)')
plt.xlabel('Date')
plt.ylabel('Closing Price (USD)')
plt.legend()
plt.grid(True)
plt.tight_layout()

# 儲存圖表並顯示
png_filename = 'prediction_comparison.png'
plt.savefig(png_filename)
print(f"Plot correctly saved as {png_filename}")

plt.show()

# ==========================================
# 7. 視覺化 MSE 比較結果 (MSE Bar Chart)
# ==========================================
# 建立一個新的圖表 (較小一點，因為是單純的比較圖)
plt.figure(figsize=(8, 6))
models = ['Random Forest', 'XGBoost']
mses = [rf_mse, xgb_mse]

# 畫出長條圖，顏色設定和折線圖一致
bars = plt.bar(models, mses, color=['blue', 'green'], alpha=0.7, width=0.5)

# 在長條上方自動標上分數數字
for bar in bars:
    yval = bar.get_height()
    # 標籤顯示在正中央上方，並用逗號隔開千位數
    plt.text(bar.get_x() + bar.get_width()/2, yval, f'{yval:,.0f}', ha='center', va='bottom', fontsize=12, fontweight='bold')

# 設定圖表標題與軸線名稱
plt.title('MSE Score Comparison (Lower is Better)', fontsize=14)
plt.ylabel('Mean Squared Error', fontsize=12)
plt.grid(axis='y', linestyle='--', alpha=0.7)

# 儲存第二張圖表
mse_filename = 'mse_comparison.png'
plt.savefig(mse_filename)
print(f"MSE bar chart correctly saved as {mse_filename}")

plt.show()
