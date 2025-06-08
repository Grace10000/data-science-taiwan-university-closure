import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# 中文顯示設定
plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei']
plt.rcParams['axes.unicode_minus'] = False

# 讀取資料
file_path = r"D:\Users\桌面\資料科學程式設計\d_資料科學期末報告(自己整理)\01_自行分析\倒閉大學數據 6.0.csv"
df = pd.read_csv(file_path, encoding="big5")
print(df.head())

# 建立「平均資金增減」欄位
capital_cols = ['106資金增減', '107資金增減', '108資金增減',
                '109資金增減', '110資金增減', '111資金增減', '112資金增減']
df["平均資金增減"] = df[capital_cols].mean(axis=1, numeric_only=True)

# 篩選有效資料
df_model = df[["平均資金增減", "是否倒閉"]].dropna()
X = df_model[["平均資金增減"]]
y = df_model["是否倒閉"]

# 建立並訓練模型
log_model = LogisticRegression()
log_model.fit(X, y)

# 模型係數與截距
a = log_model.coef_[0][0]
b = log_model.intercept_[0]
print(f"[邏輯回歸] 每增加一單位資金變動，倒閉對數機率改變：{a:.6f}")
print(f"[邏輯回歸] 資金變動為 0 時的對數機率截距：{b:.6f}")

# 模型預測與評估
y_pred = log_model.predict(X)

print("\n模型評估指標 =====")
acc = accuracy_score(y, y_pred)
print(f"(1) 準確率 (Accuracy): {acc:.4f}")

print("(2) 混淆矩陣 (Confusion Matrix):")
conf_mat = confusion_matrix(y, y_pred)
print(conf_mat)

print("(3) 分類報告 (Classification Report):")
print(classification_report(y, y_pred))

# 畫出混淆矩陣圖
labels = ["未倒閉", "已倒閉"]
plt.figure(figsize=(6, 5))
sns.heatmap(conf_mat, annot=True, fmt="d", cmap="Oranges", xticklabels=labels, yticklabels=labels)
plt.xlabel("預測值")
plt.ylabel("實際值")
plt.title("混淆矩陣：平均資金增減 vs 是否倒閉")
plt.tight_layout()
plt.show()

# 畫出邏輯回歸機率曲線
x_range = np.linspace(df["平均資金增減"].min(), df["平均資金增減"].max(), 300)
prob = 1 / (1 + np.exp(-(b + a * x_range)))

plt.figure(figsize=(10, 6))
plt.plot(x_range, prob, label="倒閉機率", color='orange', linewidth=2)
plt.scatter(X, y, alpha=0.5, label="實際數據", color="gray")
plt.xlabel("平均資金增減")
plt.ylabel("倒閉機率")
plt.title("平均資金增減與倒閉機率的關係")
plt.legend()
plt.grid(True)
plt.show()
