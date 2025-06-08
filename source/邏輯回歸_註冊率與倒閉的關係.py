import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# 中文顯示設定（適用 Windows）
plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei']
plt.rcParams['axes.unicode_minus'] = False

# 讀取 Excel 資料
file_path = r"D:\Users\桌面\資料科學程式設計\d_資料科學期末報告(自己整理)\01_自行分析\倒閉大學數據 6.0.xlsx"
sheet_name = "整合大學"
df = pd.read_excel(file_path, sheet_name=sheet_name)

# 建立平均註冊率欄位
reg_rate_cols = ['106註冊率', '107註冊率', '108註冊率', '109註冊率',
                 '110註冊率', '111註冊率', '112註冊率', '113註冊率']
df["平均註冊率"] = df[reg_rate_cols].mean(axis=1, numeric_only=True)

# 準備模型資料
df_model = df[["平均註冊率", "是否倒閉"]].dropna()
X = df_model[["平均註冊率"]]
y = df_model["是否倒閉"]

# 建立與訓練邏輯回歸模型
log_model = LogisticRegression()
log_model.fit(X, y)

# 顯示邏輯回歸參數
a_log = log_model.coef_[0][0]
b_log = log_model.intercept_[0]
print(f"每增加 1% 註冊率，倒閉對數機率改變：{a_log:.4f}")
print(f"截距（註冊率 0%）對數機率為：{b_log:.4f}")

# 建立預測結果
y_pred = log_model.predict(X)

# 模型評估指標
print("\n模型評估指標 =====")
acc = accuracy_score(y, y_pred)
print(f"(1) 準確率 (Accuracy): {acc:.4f}")

print("(2) 混淆矩陣 (Confusion Matrix):")
conf_mat = confusion_matrix(y, y_pred)
print(conf_mat)

print("(3) 分類報告 (Classification Report):")
print(classification_report(y, y_pred))

# 混淆矩陣圖
labels = ["未倒閉", "已倒閉"]
plt.figure(figsize=(6, 5))
sns.heatmap(conf_mat, annot=True, fmt="d", cmap="Oranges", xticklabels=labels, yticklabels=labels)
plt.xlabel("預測值")
plt.ylabel("實際值")
plt.title("混淆矩陣：平均註冊率 vs 是否倒閉")
plt.tight_layout()
plt.show()

# 畫出邏輯回歸曲線
x_range = np.linspace(30, 100, 300)
prob = 1 / (1 + np.exp(-(b_log + a_log * x_range)))

plt.figure(figsize=(10, 6))
plt.plot(x_range, prob, label="倒閉機率", linewidth=2)
plt.scatter(X, y, alpha=0.5, label="實際數據", color="gray")
plt.xlabel("平均註冊率 (%)")
plt.ylabel("倒閉機率")
plt.title("平均註冊率與倒閉機率的關係")
plt.legend()
plt.grid(True)
plt.show()
