import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

class NeuralNetwork(nn.Module):
    def __init__(self, input_size):
        super(NeuralNetwork, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.3),
            
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.3),
            
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.2),
            
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            
            nn.Linear(64, 1)
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
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    return X_scaled, y, X.columns.tolist(), scaler

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
    print("学生成绩预测神经网络 (PyTorch)")
    print("=" * 60)
    
    data_path = "/workspace/方案/StudentPerformanceFactors .csv"
    print(f"\n正在加载数据: {data_path}")
    
    X, y, feature_names, scaler = load_and_preprocess_data(data_path)
    print(f"数据加载完成!")
    print(f"特征数量: {X.shape[1]}")
    print(f"样本数量: {X.shape[0]}")
    print(f"\n特征列表:")
    for i, name in enumerate(feature_names):
        print(f"  {i+1}. {name}")
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
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
    
    input_size = X_train.shape[1]
    model = NeuralNetwork(input_size)
    
    print(f"\n神经网络结构:")
    print(model)
    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=10, factor=0.5)
    
    print("\n开始训练...")
    print("-" * 60)
    train_model(model, train_loader, criterion, optimizer, num_epochs=100)
    
    print("\n模型评估...")
    print("-" * 60)
    mse, rmse, mae, r2 = evaluate_model(model, test_loader)
    
    print(f"\n模型性能指标:")
    print(f"  均方误差 (MSE):      {mse:.4f}")
    print(f"  均方根误差 (RMSE):   {rmse:.4f}")
    print(f"  平均绝对误差 (MAE):  {mae:.4f}")
    print(f"  R² 分数:             {r2:.4f}")
    
    torch.save({
        'model_state_dict': model.state_dict(),
        'feature_names': feature_names,
        'scaler_mean': scaler.mean_,
        'scaler_scale': scaler.scale_
    }, 'student_performance_model.pth')
    print(f"\n模型已保存至: student_performance_model.pth")
    
    print("\n" + "=" * 60)
    print("训练完成!")
    print("=" * 60)

if __name__ == "__main__":
    main()
