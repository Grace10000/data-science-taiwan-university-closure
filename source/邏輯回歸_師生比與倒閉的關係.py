import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, LogisticRegression
import matplotlib.pyplot as plt


# 加這段讓中文字能正常顯示（微軟正黑體適用於 Windows）
plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei']
plt.rcParams['axes.unicode_minus'] = False


# 讀取資料
file_path = r"D:\Users\桌面\資料科學程式設計\d_資料科學期末報告(自己整理)\01_自行分析\倒閉大學數據 6.0.xlsx"
sheet_name = "整合大學"
df = pd.read_excel(file_path, sheet_name=sheet_name)

reg_rate_cols = ['106師生比', '107師生比', '108師生比', '109師生比',
                 '110師生比', '111師生比', '112師生比', '113師生比']
df["平均師生比"] = df[reg_rate_cols].mean(axis=1, numeric_only=True)

# 取出 X 與 y，並清除缺失值
df_model = df[["平均師生比", "是否倒閉"]].dropna()
X = df_model[["平均師生比"]]
y = df_model["是否倒閉"]

# 建立與訓練邏輯回歸模型
log_model = LogisticRegression()
log_model.fit(X, y)

# 顯示結果
a_log = log_model.coef_[0][0]
b_log = log_model.intercept_[0]
print(f"每增加 1% 師生比，倒閉對數機率改變：{a_log:.4f}")
print(f"截距（師生比 0%）對數機率為：{b_log:.4f}")



# 建立註冊率範圍與倒閉機率
x_range = np.linspace(30, 100, 300)
prob = 1 / (1 + np.exp(-(b_log + a_log * x_range)))

# 畫圖
plt.figure(figsize=(10, 6))
plt.plot(x_range, prob, label="邏輯回歸：倒閉機率", linewidth=2)
plt.scatter(X, y, alpha=0.5, label="實際數據", color="gray")
plt.xlabel("106~113年歷年平均師生比 (%)")
plt.ylabel("倒閉機率")
plt.title("歷年平均師生比與倒閉機率的關係")
plt.legend()
plt.grid(True)
plt.show()
