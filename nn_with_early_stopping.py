import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# ===================== 超参数配置 =====================
# 数据相关超参数
TEST_SIZE = 0.2  # 测试集比例 [0.1, 0.3]
VAL_SIZE = 0.1  # 验证集比例 (从训练集中划分) [0.1, 0.25]
RANDOM_STATE = 42  # 随机种子 [固定值, 用于复现]

# 数据预处理相关
BATCH_SIZE = 32  # 批次大小 [8, 128]
USE_TOP10_FEATURES = True  # 是否使用Top 10特征 [True, False]

# 神经网络结构相关
HIDDEN_DIM_1 = 64  # 第一个隐藏层维度 [32, 256]
HIDDEN_DIM_2 = 128  # 第二个隐藏层维度 [32, 256]
HIDDEN_DIM_3 = 64  # 第三个隐藏层维度 [32, 256]
HIDDEN_DIM_4 = 32  # 第四个隐藏层维度 [16, 64]
DROPOUT_RATE = 0.2  # Dropout比率 [0.1, 0.5]

# 训练相关超参数
LEARNING_RATE = 0.001  # 学习率 [1e-5, 1e-2]
WEIGHT_DECAY = 1e-5  # 权重衰减 [1e-6, 1e-3]
NUM_EPOCHS = 200  # 最大训练轮数 [50, 500]

# 早停机制相关
EARLY_STOPPING_PATIENCE = 15  # 早停耐心值 (验证集无改进的轮数) [5, 30]
EARLY_STOPPING_MIN_DELTA = 1e-4  # 早停最小改进阈值 [1e-5, 1e-3]
# =====================================================

class EarlyStopping:
    """早停机制类"""
    def __init__(self, patience=10, min_delta=0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_model_state = None

    def __call__(self, val_loss, model):
        score = -val_loss
        
        if self.best_score is None:
            self.best_score = score
            self.best_model_state = model.state_dict().copy()
        elif score < self.best_score + self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.best_model_state = model.state_dict().copy()
            self.counter = 0

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
    
    label_encoders = {}
    for col in categorical_columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le
    
    X = df.drop('Exam_Score', axis=1)
    y = df['Exam_Score'].values
    feature_names = X.columns.tolist()
    
    if USE_TOP10_FEATURES:
        top_10_features = [
            'Attendance', 'Hours_Studied', 'Previous_Scores', 'Tutoring_Sessions',
            'Access_to_Resources', 'Parental_Involvement', 'Physical_Activity',
            'Sleep_Hours', 'Family_Income', 'Peer_Influence'
        ]
        top_10_indices = [feature_names.index(f) for f in top_10_features]
        X = X.values[:, top_10_indices]
        selected_features = top_10_features
    else:
        X = X.values
        selected_features = feature_names
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    return X_scaled, y, selected_features, scaler

def train_one_epoch(model, train_loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    for batch_X, batch_y in train_loader:
        batch_X = batch_X.to(device)
        batch_y = batch_y.to(device)
        
        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs.squeeze(), batch_y)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item() * batch_X.size(0)
    
    return total_loss / len(train_loader.dataset)

def validate(model, val_loader, criterion, device):
    model.eval()
    total_loss = 0.0
    predictions = []
    actuals = []
    
    with torch.no_grad():
        for batch_X, batch_y in val_loader:
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)
            
            outputs = model(batch_X)
            loss = criterion(outputs.squeeze(), batch_y)
            
            total_loss += loss.item() * batch_X.size(0)
            predictions.extend(outputs.squeeze().cpu().numpy())
            actuals.extend(batch_y.cpu().numpy())
    
    avg_loss = total_loss / len(val_loader.dataset)
    return avg_loss, np.array(predictions), np.array(actuals)

def main():
    print("=" * 80)
    print("学生成绩预测 - PyTorch 神经网络 (含早停机制)")
    print("=" * 80)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n使用设备: {device}")
    
    print("\n" + "=" * 80)
    print("超参数配置:")
    print("=" * 80)
    print(f"  数据集划分: 测试集={TEST_SIZE}, 验证集={VAL_SIZE}")
    print(f"  批次大小: {BATCH_SIZE}")
    print(f"  使用特征: {'Top 10' if USE_TOP10_FEATURES else '全部 19 个'}")
    print(f"  网络结构: {HIDDEN_DIM_1} → {HIDDEN_DIM_2} → {HIDDEN_DIM_3} → {HIDDEN_DIM_4}")
    print(f"  Dropout率: {DROPOUT_RATE}")
    print(f"  学习率: {LEARNING_RATE}")
    print(f"  权重衰减: {WEIGHT_DECAY}")
    print(f"  最大训练轮数: {NUM_EPOCHS}")
    print(f"  早停耐心值: {EARLY_STOPPING_PATIENCE}")
    print(f"  早停最小改进: {EARLY_STOPPING_MIN_DELTA}")
    
    data_path = "/workspace/方案/StudentPerformanceFactors .csv"
    X, y, feature_names, scaler = load_and_preprocess_data(data_path)
    
    print(f"\n数据加载完成: 样本数={X.shape[0]}, 特征数={X.shape[1]}")
    print(f"使用的特征: {', '.join(feature_names)}")
    
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=VAL_SIZE / (1 - TEST_SIZE), random_state=RANDOM_STATE
    )
    
    print(f"\n数据集划分:")
    print(f"  训练集: {len(X_train)} 样本")
    print(f"  验证集: {len(X_val)} 样本")
    print(f"  测试集: {len(X_test)} 样本")
    
    X_train_tensor = torch.FloatTensor(X_train)
    y_train_tensor = torch.FloatTensor(y_train)
    X_val_tensor = torch.FloatTensor(X_val)
    y_val_tensor = torch.FloatTensor(y_val)
    X_test_tensor = torch.FloatTensor(X_test)
    y_test_tensor = torch.FloatTensor(y_test)
    
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    input_size = X_train.shape[1]
    model = NeuralNetwork(input_size).to(device)
    
    print(f"\n神经网络结构:")
    print(model)
    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    early_stopping = EarlyStopping(patience=EARLY_STOPPING_PATIENCE, min_delta=EARLY_STOPPING_MIN_DELTA)
    
    print("\n" + "=" * 80)
    print("开始训练...")
    print("=" * 80)
    print(f"{'Epoch':<8} {'Train Loss':<15} {'Val Loss':<15} {'Status':<10}")
    print("-" * 80)
    
    train_losses = []
    val_losses = []
    
    for epoch in range(NUM_EPOCHS):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, _, _ = validate(model, val_loader, criterion, device)
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        
        status = ""
        early_stopping(val_loss, model)
        
        if early_stopping.counter == 0:
            status = "✓ Best"
        elif early_stopping.early_stop:
            status = "✗ Early Stop"
        
        print(f"{epoch+1:<8} {train_loss:<15.4f} {val_loss:<15.4f} {status:<10}")
        
        if early_stopping.early_stop:
            print("\n早停触发! 恢复最佳模型...")
            model.load_state_dict(early_stopping.best_model_state)
            break
    
    print("\n" + "=" * 80)
    print("最终模型评估:")
    print("=" * 80)
    
    print("\n--- 验证集 ---")
    val_loss, val_preds, val_actuals = validate(model, val_loader, criterion, device)
    val_rmse = np.sqrt(val_loss)
    val_mae = mean_absolute_error(val_actuals, val_preds)
    val_r2 = r2_score(val_actuals, val_preds)
    print(f"  验证 MSE: {val_loss:.4f}")
    print(f"  验证 RMSE: {val_rmse:.4f}")
    print(f"  验证 MAE: {val_mae:.4f}")
    print(f"  验证 R²: {val_r2:.4f}")
    
    print("\n--- 测试集 ---")
    test_loss, test_preds, test_actuals = validate(model, test_loader, criterion, device)
    test_rmse = np.sqrt(test_loss)
    test_mae = mean_absolute_error(test_actuals, test_preds)
    test_r2 = r2_score(test_actuals, test_preds)
    print(f"  测试 MSE: {test_loss:.4f}")
    print(f"  测试 RMSE: {test_rmse:.4f}")
    print(f"  测试 MAE: {test_mae:.4f}")
    print(f"  测试 R²: {test_r2:.4f}")
    
    torch.save({
        'model_state_dict': model.state_dict(),
        'feature_names': feature_names,
        'scaler_mean': scaler.mean_,
        'scaler_scale': scaler.scale_,
        'hyperparameters': {
            'TEST_SIZE': TEST_SIZE,
            'VAL_SIZE': VAL_SIZE,
            'BATCH_SIZE': BATCH_SIZE,
            'HIDDEN_DIM_1': HIDDEN_DIM_1,
            'HIDDEN_DIM_2': HIDDEN_DIM_2,
            'HIDDEN_DIM_3': HIDDEN_DIM_3,
            'HIDDEN_DIM_4': HIDDEN_DIM_4,
            'DROPOUT_RATE': DROPOUT_RATE,
            'LEARNING_RATE': LEARNING_RATE,
            'WEIGHT_DECAY': WEIGHT_DECAY,
            'NUM_EPOCHS': NUM_EPOCHS,
            'EARLY_STOPPING_PATIENCE': EARLY_STOPPING_PATIENCE,
            'EARLY_STOPPING_MIN_DELTA': EARLY_STOPPING_MIN_DELTA,
        }
    }, 'student_performance_early_stopped_model.pth')
    
    print("\n" + "=" * 80)
    print("模型已保存至: student_performance_early_stopped_model.pth")
    print("=" * 80)

if __name__ == "__main__":
    main()
