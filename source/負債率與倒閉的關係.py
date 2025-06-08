import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


# 中文顯示設定
plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei']  
plt.rcParams['axes.unicode_minus'] = False

# 讀取檔案
file_path = r"D:\Users\桌面\資料科學程式設計\d_資料科學期末報告(自己整理)\01_自行分析\倒閉大學數據 6.0.csv"
df = pd.read_csv(file_path, encoding="big5")  
print(df.head())

# 建立「平均負債率」欄位
debt_rate_cols = ['106負債率', '107負債率', '108負債率', '109負債率',
                  '110負債率', '111負債率', '112負債率']
df["平均負債率"] = df[debt_rate_cols].mean(axis=1, numeric_only=True)

# 篩選資料
df_debt_model = df[["平均負債率", "是否倒閉"]].dropna()
X = df_debt_model[["平均負債率"]]
y = df_debt_model["是否倒閉"]

# 建立模型
log_model = LogisticRegression()
log_model.fit(X, y)

# 取得模型參數
a = log_model.coef_[0][0]
b = log_model.intercept_[0]

print(f"[邏輯回歸] 平均負債率每增加 1%，倒閉對數機率改變：{a:.4f}")
print(f"[邏輯回歸] 截距（負債率為 0%）對數機率：{b:.4f}")

# x 軸範圍（負債率 %）
x_range = np.linspace(0, 200, 300)
# 使用 sigmoid 函數計算機率
prob = 1 / (1 + np.exp(-(b + a * x_range)))



# 模型預測
y_pred = log_model.predict(X)

# 模型評估
print("\n2. 模型評估指標 =====")
acc = accuracy_score(y, y_pred)
print(f"(1) 準確率 (Accuracy): {acc:.4f}")  # 模型正確預測的比例

print("(2) 混淆矩陣 (Confusion Matrix):")
print(confusion_matrix(y, y_pred))  # 顯示 TP, FP, FN, TN 數值

print("(3) 分類報告 (Classification Report):")
print(classification_report(y, y_pred))  # precision, recall, f1-score


# 畫圖
plt.figure(figsize=(10, 6))
plt.plot(x_range, prob, label="倒閉機率", color='orange', linewidth=2)
plt.scatter(X, y, alpha=0.5, label="實際數據", color="gray")
plt.xlabel("平均負債率 (%)")
plt.ylabel("倒閉機率")
plt.title("平均負債率與倒閉機率的關係")
plt.legend()
plt.grid(True)
plt.show()
