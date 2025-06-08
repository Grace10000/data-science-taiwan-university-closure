import pandas as pd
import numpy as np
from scipy.stats import ttest_ind
from sklearn.linear_model import LinearRegression

# 讀取資料
df = pd.read_excel(r"D:\Users\桌面\資料科學程式設計\d_資料科學期末報告(自己整理)\01_自行分析\倒閉大學數據 6.0.xlsx", sheet_name="整合大學")

# 要分析的年度
years = [106, 107, 108, 109, 110, 111, 112]
year_cols = [f"{y}註冊率" for y in years]
year_nums = np.array(range(len(years))).reshape(-1, 1)

# 儲存結果
slopes = []
valid_rows = []

# 計算註冊率變化斜率
for idx, row in df.iterrows():
    try:
        rates = row[year_cols].values.astype(float)
    except:
        slopes.append(np.nan)
        valid_rows.append(False)
        continue

    if np.any(pd.isnull(rates)) or len(set(rates)) <= 1:
        slopes.append(np.nan)
        valid_rows.append(False)
        continue

    # 檢查年度間最大變動，排除異常值
    max_diff = np.max(np.abs(np.diff(rates)))
    if max_diff > 25:  # 可調整閾值
        slopes.append(np.nan)
        valid_rows.append(False)
        continue

    model = LinearRegression()
    model.fit(year_nums, rates.reshape(-1, 1))
    slopes.append(model.coef_[0][0])  # 斜率
    valid_rows.append(True)

# 新增結果欄位
df["註冊率斜率"] = slopes
df["有效數據"] = valid_rows

# 拆成倒閉與未倒閉組
slopes_closed = df[(df["是否倒閉"] == 1) & (df["有效數據"])]["註冊率斜率"].dropna()
slopes_alive = df[(df["是否倒閉"] == 0) & (df["有效數據"])]["註冊率斜率"].dropna()

# T 檢定
t_stat, p_value = ttest_ind(slopes_closed, slopes_alive, equal_var=False)

# 輸出
print("有效倒閉學校數量:", len(slopes_closed))
print("有效現存學校數量:", len(slopes_alive))
print("倒閉學校 平均註冊率變化:", slopes_closed.mean())
print("現存學校 平均註冊率變化:", slopes_alive.mean())
print("T統計值:", t_stat)
print("P值:", p_value)

if p_value < 0.05:
    print("✅ 註冊率變化有顯著差異，可能與倒閉有關")
else:
    print("❌ 註冊率變化沒有顯著差異")