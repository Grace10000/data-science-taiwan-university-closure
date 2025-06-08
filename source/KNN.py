import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE

# ======== 1. 載入資料 ========
file_path = r'D:\Users\桌面\資料科學程式設計\d_資料科學期末報告(自己整理)\01_自行分析\倒閉大學數據 6.0.csv'
encoding = 'big5'  # 根據實際檔案編碼調整

try:
    df = pd.read_csv(file_path, encoding=encoding)
except Exception as e:
    raise Exception(f"讀取 CSV 發生錯誤: {e}")

# ======== 2. 特徵與目標欄位 ========
feature_columns = [
    '地區碼', '公私立碼',
    '106註冊率', '107註冊率', '108註冊率', '109註冊率', '110註冊率', '111註冊率', '112註冊率', '113註冊率',
    '106師生比', '107師生比', '108師生比', '109師生比', '110師生比', '111師生比', '112師生比', '113師生比',
    '106負債率', '107負債率', '108負債率', '109負債率', '110負債率', '111負債率', '112負債率',
    '106學生數', '107學生數', '108學生數', '109學生數', '110學生數', '111學生數', '112學生數', '113學生數',
    '106資金增減', '107資金增減', '108資金增減', '109資金增減', '110資金增減', '111資金增減', '112資金增減'
]
target_column = '是否倒閉'

# ======== 3. 清洗資料與補值 ========
df_model = df[feature_columns + [target_column]].copy()
df_model[target_column] = df_model[target_column].fillna(df_model[target_column].mode()[0])

# 補特徵欄位缺值（中位數）
imputer = SimpleImputer(strategy='median')
X = imputer.fit_transform(df_model[feature_columns])
y = df_model[target_column].astype(int)

# ======== 4. 特徵標準化 ========
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ======== 5. 分割資料（分層） ========
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.3, random_state=42, stratify=y
)

# ======== 6. SMOTE 類別平衡處理 ========
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

# ======== 7. 建立與訓練 KNN 模型 ========
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train_smote, y_train_smote)

# ======== 8. 預測與評估 ========
y_pred = knn.predict(X_test)

print("混淆矩陣:")
print(confusion_matrix(y_test, y_pred))
print("\n分類報告:")
print(classification_report(y_test, y_pred))

print(f"\n原始資料總筆數: {len(y)}（倒閉數: {(y==1).sum()}，未倒閉數: {(y==0).sum()}）")
print(f"SMOTE後訓練資料筆數: {len(y_train_smote)}（倒閉數: {(y_train_smote==1).sum()}，未倒閉數: {(y_train_smote==0).sum()}）")
