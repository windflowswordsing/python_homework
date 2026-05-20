import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

class NeuralNetwork(nn.Module):
    def __init__(self, input_size):
        super(NeuralNetwork, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(0.2),
            
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.2),
            
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(0.2),
            
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.BatchNorm1d(32),
            
            nn.Linear(32, 1)
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
    
    label_encoders = {}
    for col in categorical_columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le
    
    X = df.drop('Exam_Score', axis=1)
    y = df['Exam_Score'].values
    
    feature_names = X.columns.tolist()
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    return X_scaled, y, feature_names, scaler

def calculate_feature_importance(X, y, feature_names):
    print("\n正在计算特征重要性 (使用随机森林)...")
    
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X, y)
    
    importances = rf.feature_importances_
    
    feature_importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances
    })
    feature_importance_df = feature_importance_df.sort_values('Importance', ascending=False)
    
    return feature_importance_df

def train_model(model, train_loader, criterion, optimizer, num_epochs=100):
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs.squeeze(), batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        if (epoch + 1) % 10 == 0:
            avg_loss = total_loss / len(train_loader)
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}')

def evaluate_model(model, test_loader):
    model.eval()
    predictions = []
    actuals = []
    
    with torch.no_grad():
        for batch_X, batch_y in test_loader:
            outputs = model(batch_X)
            predictions.extend(outputs.squeeze().numpy())
            actuals.extend(batch_y.numpy())
    
    predictions = np.array(predictions)
    actuals = np.array(actuals)
    
    mse = mean_squared_error(actuals, predictions)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(actuals, predictions)
    r2 = r2_score(actuals, predictions)
    
    return mse, rmse, mae, r2

def main():
    print("=" * 60)
    print("特征重要性分析与优化神经网络 (PyTorch)")
    print("=" * 60)
    
    data_path = "/workspace/方案/StudentPerformanceFactors .csv"
    print(f"\n正在加载数据: {data_path}")
    
    X, y, feature_names, scaler = load_and_preprocess_data(data_path)
    print(f"数据加载完成! 样本数: {X.shape[0]}, 特征数: {X.shape[1]}")
    
    feature_importance_df = calculate_feature_importance(X, y, feature_names)
    
    print("\n" + "=" * 60)
    print("特征重要性排序:")
    print("=" * 60)
    for i, row in enumerate(feature_importance_df.itertuples(), 1):
        print(f"  {i:2d}. {row.Feature:25s}  重要性: {row.Importance:.6f}")
    
    top_10_features = feature_importance_df.head(10)['Feature'].tolist()
    top_10_indices = [feature_names.index(f) for f in top_10_features]
    
    print("\n" + "=" * 60)
    print("TOP 10 最具影响力特征:")
    print("=" * 60)
    for i, (feature, importance) in enumerate(zip(top_10_features, feature_importance_df.head(10)['Importance']), 1):
        print(f"  {i:2d}. {feature:25s}  重要性: {importance:.6f}")
    
    X_top10 = X[:, top_10_indices]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X_top10, y, test_size=0.2, random_state=42
    )
    
    print(f"\n数据划分:")
    print(f"  训练集: {len(X_train)} 样本")
    print(f"  测试集: {len(X_test)} 样本")
    
    X_train_tensor = torch.FloatTensor(X_train)
    y_train_tensor = torch.FloatTensor(y_train)
    X_test_tensor = torch.FloatTensor(X_test)
    y_test_tensor = torch.FloatTensor(y_test)
    
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    input_size = 10
    model = NeuralNetwork(input_size)
    
    print(f"\n新神经网络结构 (Top 10 特征):")
    print(model)
    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    
    print("\n开始训练新模型 (Top 10 特征)...")
    print("-" * 60)
    train_model(model, train_loader, criterion, optimizer, num_epochs=100)
    
    print("\n新模型评估...")
    print("-" * 60)
    mse, rmse, mae, r2 = evaluate_model(model, test_loader)
    
    print("\n" + "=" * 60)
    print("新模型 (Top 10 特征) 性能指标:")
    print("=" * 60)
    print(f"  均方误差 (MSE):      {mse:.4f}")
    print(f"  均方根误差 (RMSE):   {rmse:.4f}")
    print(f"  平均绝对误差 (MAE):  {mae:.4f}")
    print(f"  R² 分数:             {r2:.4f}")
    
    torch.save({
        'model_state_dict': model.state_dict(),
        'feature_names': top_10_features,
        'all_feature_names': feature_names,
        'top_10_indices': top_10_indices
    }, 'student_performance_top10_model.pth')
    
    print(f"\n模型已保存至: student_performance_top10_model.pth")
    
    print("\n" + "=" * 60)
    print("完成!")
    print("=" * 60)

if __name__ == "__main__":
    main()
