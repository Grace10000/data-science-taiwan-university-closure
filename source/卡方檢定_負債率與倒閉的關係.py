import pandas as pd

file_path = r'D:\Users\桌面\資料科學程式設計\d_資料科學期末報告(自己整理)\01_自行分析\倒閉大學數據 5.0.xlsx'

df = pd.read_excel(file_path, header=None)
print(df.head(10))
