import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

try:
    import matplotlib.pyplot as plt
    PLOTTING_AVAILABLE = True
except ImportError:
    PLOTTING_AVAILABLE = False
    print("⚠️  matplotlib未安装，将跳过绘图功能")

# ===================== 超参数配置 =====================
TEST_SIZE = 0.2
VAL_SIZE = 0.1
RANDOM_STATE = 42
BATCH_SIZE = 32
USE_TOP10_FEATURES = True
USE_KFOLD = True
NUM_FOLDS = 5

# 网络结构
HIDDEN_DIM_1 = 96
HIDDEN_DIM_2 = 192
HIDDEN_DIM_3 = 96
HIDDEN_DIM_4 = 48
DROPOUT_RATE_1 = 0.25
DROPOUT_RATE_2 = 0.35
DROPOUT_RATE_3 = 0.25
DROPOUT_RATE_4 = 0.15

# 训练参数
LEARNING_RATE = 0.001
WEIGHT_DECAY = 5e-5
L1_LAMBDA = 1e-6
NUM_EPOCHS = 300
GRADIENT_CLIP = 1.0

# 早停机制
EARLY_STOPPING_PATIENCE = 20
EARLY_STOPPING_MIN_DELTA = 5e-5

# 学习率调度
SCHEDULER_PATIENCE = 10
SCHEDULER_FACTOR = 0.5
# =====================================================


class EarlyStopping:
    """增强版早停机制"""
    def __init__(self, patience=15, min_delta=1e-4, verbose=False):
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_model_state = None
        self.best_loss = np.inf

    def __call__(self, val_loss, model):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.best_loss = val_loss
            self.best_model_state = model.state_dict().copy()
            if self.verbose:
                print(f"✓ Best loss: {val_loss:.6f}")
        elif score < self.best_score + self.min_delta:
            self.counter += 1
            if self.verbose:
                print(f"  EarlyStopping counter: {self.counter}/{self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.best_loss = val_loss
            self.best_model_state = model.state_dict().copy()
            self.counter = 0
            if self.verbose:
                print(f"✓ Best loss: {val_loss:.6f}")


class RegularizedNeuralNetwork(nn.Module):
    """带多种正则化技术的神经网络"""
    def __init__(self, input_size):
        super(RegularizedNeuralNetwork, self).__init__()
        
        # 输入层 - 加入BatchNorm在激活前
        self.network = nn.Sequential(
            nn.Linear(input_size, HIDDEN_DIM_1),
            nn.BatchNorm1d(HIDDEN_DIM_1),
            nn.ReLU(),
            nn.Dropout(DROPOUT_RATE_1),
            
            nn.Linear(HIDDEN_DIM_1, HIDDEN_DIM_2),
            nn.BatchNorm1d(HIDDEN_DIM_2),
            nn.ReLU(),
            nn.Dropout(DROPOUT_RATE_2),
            
            nn.Linear(HIDDEN_DIM_2, HIDDEN_DIM_3),
            nn.BatchNorm1d(HIDDEN_DIM_3),
            nn.ReLU(),
            nn.Dropout(DROPOUT_RATE_3),
            
            nn.Linear(HIDDEN_DIM_3, HIDDEN_DIM_4),
            nn.BatchNorm1d(HIDDEN_DIM_4),
            nn.ReLU(),
            nn.Dropout(DROPOUT_RATE_4),
            
            nn.Linear(HIDDEN_DIM_4, 1)
        )
        
        # 初始化权重
        self._init_weights()
    
    def _init_weights(self):
        """Kaiming初始化用于ReLU激活"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        return self.network(x)
    
    def get_l1_penalty(self):
        """计算L1正则化项"""
        l1_loss = 0
        for param in self.parameters():
            l1_loss += torch.sum(torch.abs(param))
        return l1_loss


def add_noise(X, noise_level=0.01):
    """添加高斯噪声进行数据增强"""
    noise = np.random.normal(0, noise_level, X.shape)
    return X + noise


def load_and_preprocess_data(filepath, augment=False):
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
    
    if augment:
        X_scaled = add_noise(X_scaled)
    
    return X_scaled, y, selected_features, scaler


def train_one_epoch(model, train_loader, criterion, optimizer, device, use_l1=True):
    model.train()
    total_loss = 0.0
    
    for batch_X, batch_y in train_loader:
        batch_X = batch_X.to(device)
        batch_y = batch_y.to(device)
        
        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs.squeeze(), batch_y)
        
        if use_l1:
            l1_penalty = L1_LAMBDA * model.get_l1_penalty()
            loss += l1_penalty
        
        loss.backward()
        
        if GRADIENT_CLIP > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=GRADIENT_CLIP)
        
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


def plot_training_curves(train_losses, val_losses, fold=None):
    if not PLOTTING_AVAILABLE:
        return
    try:
        plt.figure(figsize=(10, 6))
        plt.plot(train_losses, label='Train Loss', linewidth=2)
        plt.plot(val_losses, label='Val Loss', linewidth=2)
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('Loss', fontsize=12)
        plt.title(f'Training Curves{" - Fold " + str(fold) if fold else ""}', fontsize=14)
        plt.legend(fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.savefig(f'training_curves{"_fold" + str(fold) if fold else ""}.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  ✓ 训练曲线已保存至 training_curves{'_fold' + str(fold) if fold else ''}.png")
    except Exception as e:
        print(f"  ⚠️  绘图失败: {e}")


def train_fold(X_train, y_train, X_val, y_val, fold=None):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    X_train_tensor = torch.FloatTensor(X_train)
    y_train_tensor = torch.FloatTensor(y_train)
    X_val_tensor = torch.FloatTensor(X_val)
    y_val_tensor = torch.FloatTensor(y_val)
    
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    input_size = X_train.shape[1]
    model = RegularizedNeuralNetwork(input_size).to(device)
    
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=SCHEDULER_FACTOR, 
        patience=SCHEDULER_PATIENCE, min_lr=1e-7
    )
    
    early_stopping = EarlyStopping(patience=EARLY_STOPPING_PATIENCE, min_delta=EARLY_STOPPING_MIN_DELTA, verbose=True)
    
    train_losses = []
    val_losses = []
    
    for epoch in range(NUM_EPOCHS):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_preds, val_actuals = validate(model, val_loader, criterion, device)
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        
        scheduler.step(val_loss)
        early_stopping(val_loss, model)
        
        if (epoch + 1) % 10 == 0:
            val_rmse = np.sqrt(val_loss)
            val_mae = mean_absolute_error(val_actuals, val_preds)
            val_r2 = r2_score(val_actuals, val_preds)
            print(f"Epoch [{epoch+1:3d}/{NUM_EPOCHS}] Train: {train_loss:.4f} Val: {val_loss:.4f} RMSE: {val_rmse:.4f} MAE: {val_mae:.4f} R²: {val_r2:.4f}")
        
        if early_stopping.early_stop:
            print(f"\n早停触发于 Epoch {epoch+1}")
            model.load_state_dict(early_stopping.best_model_state)
            break
    
    plot_training_curves(train_losses, val_losses, fold)
    
    return model, val_losses[-1] if val_losses else float('inf')


def main():
    print("=" * 100)
    print("学生成绩预测 - 高级正则化神经网络 (PyTorch)".center(100))
    print("=" * 100)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n使用设备: {device}")
    
    print("\n" + "=" * 100)
    print("高级正则化技术配置:".center(100))
    print("=" * 100)
    print(f"  ✓ L2正则化 (Weight Decay): {WEIGHT_DECAY}")
    print(f"  ✓ L1正则化 (Lasso): λ={L1_LAMBDA}")
    print(f"  ✓ Dropout: 层1={DROPOUT_RATE_1}, 层2={DROPOUT_RATE_2}, 层3={DROPOUT_RATE_3}, 层4={DROPOUT_RATE_4}")
    print(f"  ✓ 批量归一化 (BatchNorm)")
    print(f"  ✓ 梯度裁剪 (Max Norm: {GRADIENT_CLIP})")
    print(f"  ✓ 早停机制 (Patience: {EARLY_STOPPING_PATIENCE})")
    print(f"  ✓ 学习率调度 (ReduceLROnPlateau)")
    print(f"  ✓ Kaiming权重初始化")
    if USE_KFOLD:
        print(f"  ✓ K折交叉验证 (Folds: {NUM_FOLDS})")
    
    data_path = "/workspace/方案/StudentPerformanceFactors .csv"
    X, y, feature_names, scaler = load_and_preprocess_data(data_path)
    
    print(f"\n数据加载完成: 样本数={X.shape[0]}, 特征数={X.shape[1]}")
    print(f"使用的特征: {', '.join(feature_names)}")
    
    if USE_KFOLD:
        print("\n" + "=" * 100)
        print("K折交叉验证".center(100))
        print("=" * 100)
        
        kf = KFold(n_splits=NUM_FOLDS, shuffle=True, random_state=RANDOM_STATE)
        fold_scores = []
        models = []
        
        for fold, (train_idx, val_idx) in enumerate(kf.split(X), 1):
            print(f"\n{'=' * 100}")
            print(f"Fold {fold}/{NUM_FOLDS}".center(100))
            print('=' * 100)
            
            X_train_fold, X_val_fold = X[train_idx], X[val_idx]
            y_train_fold, y_val_fold = y[train_idx], y[val_idx]
            
            print(f"训练集: {len(X_train_fold)}, 验证集: {len(X_val_fold)}")
            
            model, best_val_loss = train_fold(X_train_fold, y_train_fold, X_val_fold, y_val_fold, fold)
            fold_scores.append(best_val_loss)
            models.append(model)
        
        print("\n" + "=" * 100)
        print("交叉验证结果汇总".center(100))
        print("=" * 100)
        print(f"  各Fold验证损失: {', '.join([f'{s:.4f}' for s in fold_scores])}")
        print(f"  平均验证损失: {np.mean(fold_scores):.4f} ± {np.std(fold_scores):.4f}")
        
        best_fold_idx = np.argmin(fold_scores)
        best_model = models[best_fold_idx]
        print(f"  最佳模型来自 Fold {best_fold_idx + 1}")
    else:
        X_train_val, X_test, y_train_val, y_test = train_test_split(
            X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
        )
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_val, y_train_val, test_size=VAL_SIZE / (1 - TEST_SIZE), random_state=RANDOM_STATE
        )
        
        print(f"\n数据集划分:")
        print(f"  训练集: {len(X_train)}, 验证集: {len(X_val)}, 测试集: {len(X_test)}")
        
        best_model, _ = train_fold(X_train, y_train, X_val, y_val)
        X_test_tensor = torch.FloatTensor(X_test)
        y_test_tensor = torch.FloatTensor(y_test)
        test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
        test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    print("\n" + "=" * 100)
    print("最终模型评估".center(100))
    print("=" * 100)
    
    # 在完整的测试集上评估
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    
    X_test_tensor = torch.FloatTensor(X_test)
    y_test_tensor = torch.FloatTensor(y_test)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    criterion = nn.MSELoss()
    test_loss, test_preds, test_actuals = validate(best_model, test_loader, criterion, device)
    
    test_rmse = np.sqrt(test_loss)
    test_mae = mean_absolute_error(test_actuals, test_preds)
    test_r2 = r2_score(test_actuals, test_preds)
    
    print("\n测试集性能:")
    print(f"  均方误差 (MSE):   {test_loss:.6f}")
    print(f"  均方根误差 (RMSE): {test_rmse:.6f}")
    print(f"  平均绝对误差 (MAE):{test_mae:.6f}")
    print(f"  R² 分数:           {test_r2:.6f}")
    
    # 保存最佳模型
    torch.save({
        'model_state_dict': best_model.state_dict(),
        'feature_names': feature_names,
        'scaler_mean': scaler.mean_,
        'scaler_scale': scaler.scale_,
        'hyperparameters': {
            'HIDDEN_DIM_1': HIDDEN_DIM_1,
            'HIDDEN_DIM_2': HIDDEN_DIM_2,
            'HIDDEN_DIM_3': HIDDEN_DIM_3,
            'HIDDEN_DIM_4': HIDDEN_DIM_4,
            'DROPOUT_RATE_1': DROPOUT_RATE_1,
            'DROPOUT_RATE_2': DROPOUT_RATE_2,
            'DROPOUT_RATE_3': DROPOUT_RATE_3,
            'DROPOUT_RATE_4': DROPOUT_RATE_4,
            'LEARNING_RATE': LEARNING_RATE,
            'WEIGHT_DECAY': WEIGHT_DECAY,
            'L1_LAMBDA': L1_LAMBDA,
            'GRADIENT_CLIP': GRADIENT_CLIP,
            'EARLY_STOPPING_PATIENCE': EARLY_STOPPING_PATIENCE,
            'SCHEDULER_PATIENCE': SCHEDULER_PATIENCE,
            'SCHEDULER_FACTOR': SCHEDULER_FACTOR,
        },
        'test_metrics': {
            'test_loss': test_loss,
            'test_rmse': test_rmse,
            'test_mae': test_mae,
            'test_r2': test_r2
        }
    }, 'advanced_regularized_model.pth')
    
    print(f"\n模型已保存至: advanced_regularized_model.pth")
    print("\n" + "=" * 100)
    print("任务完成!".center(100))
    print("=" * 100)


if __name__ == "__main__":
    main()
