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
TEST_SIZE = 0.2       # 测试集比例 [0.1, 0.3]
VAL_SIZE = 0.1        # 验证集比例 [0.1, 0.25]
RANDOM_STATE = 42     # 随机种子
BATCH_SIZE = 32       # 批次大小 [8, 128]
HIDDEN_DIM_1 = 64     # 隐藏层1维度 [32, 256]
HIDDEN_DIM_2 = 128    # 隐藏层2维度 [32, 256]
HIDDEN_DIM_3 = 64     # 隐藏层3维度 [32, 256]
HIDDEN_DIM_4 = 32     # 隐藏层4维度 [16, 64]
DROPOUT_RATE = 0.2    # Dropout率 [0.1, 0.5]
LEARNING_RATE = 0.001 # 学习率 [1e-5, 1e-2]
WEIGHT_DECAY = 1e-5   # 权重衰减 [1e-6, 1e-3]
NUM_EPOCHS = 200      # 最大训练轮数 [50, 500]
EARLY_STOP_PATIENCE = 15  # 早停耐心值 [5, 30]
EARLY_STOP_MIN_DELTA = 1e-4  # 早停最小改进 [1e-5, 1e-3]
# =====================================================

class EarlyStopping:
    def __init__(self, patience=10, min_delta=0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_state = None

    def __call__(self, val_loss, model):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.best_state = model.state_dict().copy()
        elif score < self.best_score + self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.best_state = model.state_dict().copy()
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

def load_and_preprocess_data(filepath, selected_features):
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
    
    all_feature_names = [col for col in df.columns if col != 'Exam_Score']
    feature_indices = [all_feature_names.index(f) for f in selected_features]
    
    X = df[selected_features].values
    y = df['Exam_Score'].values
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    return X_scaled, y, scaler

def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    for X, y in loader:
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()
        pred = model(X)
        loss = criterion(pred.squeeze(), y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * X.size(0)
    return total_loss / len(loader.dataset)

def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    preds, actuals = [], []
    with torch.no_grad():
        for X, y in loader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            loss = criterion(pred.squeeze(), y)
            total_loss += loss.item() * X.size(0)
            preds.extend(pred.squeeze().cpu().numpy())
            actuals.extend(y.cpu().numpy())
    return total_loss/len(loader.dataset), np.array(preds), np.array(actuals)

def main():
    print("=" * 80)
    print("文件 2: 使用 Top 10 特征预测 Exam_Score")
    print("=" * 80)
    
    try:
        top_10_features = np.load('top_10_features.npy', allow_pickle=True).tolist()
    except FileNotFoundError:
        print("\n未找到 top_10_features.npy，使用预设的 Top 10 特征")
        top_10_features = [
            'Attendance', 'Hours_Studied', 'Previous_Scores', 'Tutoring_Sessions',
            'Access_to_Resources', 'Parental_Involvement', 'Physical_Activity',
            'Sleep_Hours', 'Family_Income', 'Peer_Influence'
        ]
    
    print(f"\n使用的 Top 10 特征: {', '.join(top_10_features)}")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data_path = "/workspace/方案/StudentPerformanceFactors .csv"
    X, y, scaler = load_and_preprocess_data(data_path, top_10_features)
    
    print(f"\n数据加载完成: 样本数 = {X.shape[0]}, 使用特征数 = {X.shape[1]}")
    
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=VAL_SIZE/(1-TEST_SIZE), random_state=RANDOM_STATE
    )
    
    print(f"\n数据集划分:")
    print(f"  训练集: {len(X_train)} 样本")
    print(f"  验证集: {len(X_val)} 样本")
    print(f"  测试集: {len(X_test)} 样本")
    
    X_train_t = torch.FloatTensor(X_train)
    y_train_t = torch.FloatTensor(y_train)
    X_val_t = torch.FloatTensor(X_val)
    y_val_t = torch.FloatTensor(y_val)
    X_test_t = torch.FloatTensor(X_test)
    y_test_t = torch.FloatTensor(y_test)
    
    train_loader = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val_t, y_val_t), batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(TensorDataset(X_test_t, y_test_t), batch_size=BATCH_SIZE, shuffle=False)
    
    model = NeuralNetwork(input_size=10).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    early_stopping = EarlyStopping(patience=EARLY_STOP_PATIENCE, min_delta=EARLY_STOP_MIN_DELTA)
    
    print(f"\n神经网络结构:")
    print(model)
    
    print("\n" + "=" * 80)
    print("开始训练...")
    print("=" * 80)
    print(f"{'Epoch':<8} {'Train Loss':<15} {'Val Loss':<15} {'Status':<10}")
    print("-" * 80)
    
    for epoch in range(NUM_EPOCHS):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, _, _ = evaluate(model, val_loader, criterion, device)
        
        early_stopping(val_loss, model)
        status = "✓ Best" if early_stopping.counter == 0 else ("✗ Early Stop" if early_stopping.early_stop else "")
        
        print(f"{epoch+1:<8} {train_loss:<15.4f} {val_loss:<15.4f} {status:<10}")
        
        if early_stopping.early_stop:
            print("\n早停触发！恢复最佳模型...")
            model.load_state_dict(early_stopping.best_state)
            break
    
    print("\n" + "=" * 80)
    print("最终模型评估:")
    print("=" * 80)
    
    _, val_preds, val_actuals = evaluate(model, val_loader, criterion, device)
    val_rmse = np.sqrt(mean_squared_error(val_actuals, val_preds))
    val_mae = mean_absolute_error(val_actuals, val_preds)
    val_r2 = r2_score(val_actuals, val_preds)
    
    print("\n--- 验证集 ---")
    print(f"  RMSE: {val_rmse:.4f}")
    print(f"  MAE:  {val_mae:.4f}")
    print(f"  R²:   {val_r2:.4f}")
    
    _, test_preds, test_actuals = evaluate(model, test_loader, criterion, device)
    test_rmse = np.sqrt(mean_squared_error(test_actuals, test_preds))
    test_mae = mean_absolute_error(test_actuals, test_preds)
    test_r2 = r2_score(test_actuals, test_preds)
    
    print("\n--- 测试集 ---")
    print(f"  RMSE: {test_rmse:.4f}")
    print(f"  MAE:  {test_mae:.4f}")
    print(f"  R²:   {test_r2:.4f}")
    
    torch.save({
        'model_state': model.state_dict(),
        'features': top_10_features,
        'scaler_mean': scaler.mean_,
        'scaler_scale': scaler.scale_
    }, 'exam_score_predictor.pth')
    
    print(f"\n模型已保存至: exam_score_predictor.pth")
    print("\n" + "=" * 80)
    print("任务完成!")
    print("=" * 80)

if __name__ == "__main__":
    main()
