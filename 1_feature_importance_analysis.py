import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor

# ===================== 超参数配置 =====================
TEST_SIZE = 0.2  # 测试集比例 [0.1, 0.3]
RANDOM_STATE = 42  # 随机种子

# 神经网络结构
HIDDEN_DIM_1 = 128
HIDDEN_DIM_2 = 256
HIDDEN_DIM_3 = 128
HIDDEN_DIM_4 = 64
DROPOUT_RATE = 0.3
LEARNING_RATE = 0.001
NUM_EPOCHS = 100
# =====================================================

class NeuralNetwork(nn.Module):
    def __init__(self, input_size):
        super(NeuralNetwork, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, HIDDEN_DIM_1),
            nn.ReLU(),
            nn.BatchNorm1d(HIDDEN_DIM_1),
            nn.Dropout(DROPOUT_RATE),
            nn.Linear(HIDDEN_DIM_1, HIDDEN_DIM_2),
            nn.ReLU(),
            nn.BatchNorm1d(HIDDEN_DIM_2),
            nn.Dropout(DROPOUT_RATE),
            nn.Linear(HIDDEN_DIM_2, HIDDEN_DIM_3),
            nn.ReLU(),
            nn.BatchNorm1d(HIDDEN_DIM_3),
            nn.Dropout(DROPOUT_RATE),
            nn.Linear(HIDDEN_DIM_3, HIDDEN_DIM_4),
            nn.ReLU(),
            nn.BatchNorm1d(HIDDEN_DIM_4),
            nn.Linear(HIDDEN_DIM_4, 1)
        )
    
    def forward(self, x):
        return self.network(x)

def load_and_preprocess_data(filepath):
    df = pd.read_csv(filepath)
    
    categorical_columns = [
        'Parental_Involvement', 'Access_to_Resources', 'Extracurricular_Activities',
        'Motivation_Level', 'Internet_Access', 'Family_Income', 'Teacher_Quality',
        'School_Type', 'Peer_Influence', 'Learning_Disabilities',
        'Parental_Education_Level', 'Distance_from_Home', 'Gender'
    ]
    
    for col in categorical_columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
    
    X = df.drop('Exam_Score', axis=1)
    y = df['Exam_Score'].values
    feature_names = X.columns.tolist()
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    return X_scaled, y, feature_names, scaler

def calculate_feature_importance(X, y, feature_names):
    print("\n正在使用随机森林计算特征重要性...")
    rf = RandomForestRegressor(n_estimators=100, random_state=RANDOM_STATE)
    rf.fit(X, y)
    
    feature_importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': rf.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    return feature_importance_df

def main():
    print("=" * 80)
    print("文件 1: 特征重要性分析")
    print("=" * 80)
    
    data_path = "/workspace/方案/StudentPerformanceFactors .csv"
    X, y, feature_names, scaler = load_and_preprocess_data(data_path)
    
    print(f"\n数据加载完成: 样本数 = {X.shape[0]}, 特征数 = {X.shape[1]}")
    
    feature_importance_df = calculate_feature_importance(X, y, feature_names)
    
    print("\n" + "=" * 80)
    print("所有 19 个特征的重要性排序:")
    print("=" * 80)
    print(f"{'排名':<6} {'特征名称':<30} {'重要性':<15}")
    print("-" * 80)
    for idx, row in feature_importance_df.iterrows():
        rank = idx + 1
        print(f"{rank:<6} {row.Feature:<30} {row.Importance:.6f}")
    
    top_10_features = feature_importance_df.head(10)
    print("\n" + "=" * 80)
    print("Top 10 最具影响力特征:")
    print("=" * 80)
    print(f"{'排名':<6} {'特征名称':<30} {'重要性':<15}")
    print("-" * 80)
    for rank, (_, row) in enumerate(top_10_features.iterrows(), 1):
        print(f"{rank:<6} {row.Feature:<30} {row.Importance:.6f}")
    
    np.save('top_10_features.npy', top_10_features['Feature'].tolist())
    print(f"\nTop 10 特征已保存至: top_10_features.npy")
    print(f"Top 10 特征列表: {', '.join(top_10_features['Feature'].tolist())}")
    
    print("\n" + "=" * 80)
    print("特征重要性分析完成!")
    print("=" * 80)

if __name__ == "__main__":
    main()
