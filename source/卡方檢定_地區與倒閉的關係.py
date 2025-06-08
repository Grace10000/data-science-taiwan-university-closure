import pandas as pd
from scipy.stats import chi2_contingency

file_path = r'D:\Users\桌面\資料科學程式設計\d_資料科學期末報告(自己整理)\01_自行分析\倒閉大學數據 5.0.xlsx'

# 讀取 Excel（假設標題在第一列或第2列，先嘗試不帶 header）
df = pd.read_excel(file_path, header=1)  # 你可調整header行數，例如header=1或header=0看哪個正常

# 篩選你需要的欄位，使用欄位位置索引
# pandas的iloc是左閉右開，所以用0、2、4、12
df_selected = df.iloc[:, [0, 2, 4, 12]]

# 重設欄位名稱為方便讀取的名稱
df_selected.columns = ['代碼', '學校名稱', '是否倒閉', '地區碼']

print(df_selected.head())   # 先看前五筆資料

# 做交叉表
contingency_table = pd.crosstab(df_selected['地區碼'], df_selected['是否倒閉'])
print("\n交叉表：")
print(contingency_table)

# 卡方檢定
chi2, p, dof, expected = chi2_contingency(contingency_table)

print(f"卡方值 = {chi2:.4f}")  # 數字越大，代表理論與實際的差異越大
print(f"自由度 = {dof}")       # 表示結果的值變動幅度= 4，自由度 = (地區類別數 - 1) × (倒閉狀態數 - 1) = (5 -1)×(2 -1) = 4
print(f"p值 = {p:.4f}")        # 假設分析的值之間沒有關係，那遇到目前資料表的情形的機率  # < 0.05 表示其實有顯著關係(其實真的有關係)

if p < 0.05:
    print("結論：地區碼與是否倒閉之間有顯著關係（拒絕虛無假設）")
else:
    print("結論：地區碼與是否倒閉之間沒有顯著關係（無法拒絕虛無假設）")