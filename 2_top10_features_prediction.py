import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings
warnings.filterwarnings('ignore')

try:
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use('Agg')
    PLOTTING_AVAILABLE = True
except ImportError:
    PLOTTING_AVAILABLE = False
    print("⚠️  matplotlib未安装，将跳过绘图功能")


class EarlyStopping:
    """早停机制"""
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
                print(f"  ✓ 最佳损失: {val_loss:.6f}")
        elif score < self.best_score + self.min_delta:
            self.counter += 1
            if self.verbose:
                print(f"  早停计数: {self.counter}/{self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.best_loss = val_loss
            self.best_model_state = model.state_dict().copy()
            self.counter = 0
            if self.verbose:
                print(f"  ✓ 最佳损失: {val_loss:.6f}")


class Top10PredictorNN(nn.Module):
    """用于Top 10特征预测的神经网络"""
    def __init__(self, input_size):
        super(Top10PredictorNN, self).__init__()
        
        self.network = nn.Sequential(
            nn.Linear(input_size, 96),
            nn.BatchNorm1d(96),
            nn.ReLU(),
            nn.Dropout(0.25),
            
            nn.Linear(96, 192),
            nn.BatchNorm1d(192),
            nn.ReLU(),
            nn.Dropout(0.35),
            
            nn.Linear(192, 96),
            nn.BatchNorm1d(96),
            nn.ReLU(),
            nn.Dropout(0.25),
            
            nn.Linear(96, 48),
            nn.BatchNorm1d(48),
            nn.ReLU(),
            nn.Dropout(0.15),
            
            nn.Linear(48, 1)
        )
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        return self.network(x)


def load_and_preprocess_data(filepath, top_10_features):
    """加载和预处理数据，仅使用Top 10特征"""
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
    all_feature_names = X.columns.tolist()
    
    top_10_indices = [all_feature_names.index(f) for f in top_10_features]
    X_selected = X.values[:, top_10_indices]
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_selected)
    
    return X_scaled, y, top_10_features, scaler


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
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
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


def plot_training_curves(train_losses, val_losses, save_path='training_curves.png'):
    if not PLOTTING_AVAILABLE:
        return
    try:
        plt.figure(figsize=(10, 6))
        plt.plot(train_losses, label='Train Loss', linewidth=2)
        plt.plot(val_losses, label='Val Loss', linewidth=2)
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('Loss', fontsize=12)
        plt.title('Training Curves', fontsize=14, fontweight='bold')
        plt.legend(fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  ✓ 训练曲线已保存至: {save_path}")
    except Exception as e:
        print(f"  ⚠️  绘图失败: {e}")


def plot_predictions(y_true, y_pred, save_path='predictions_plot.png'):
    if not PLOTTING_AVAILABLE:
        return
    try:
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        axes[0].scatter(y_true, y_pred, alpha=0.5, s=20)
        min_val = min(y_true.min(), y_pred.min())
        max_val = max(y_true.max(), y_pred.max())
        axes[0].plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2)
        axes[0].set_xlabel('True Exam Score', fontsize=12)
        axes[0].set_ylabel('Predicted Exam Score', fontsize=12)
        axes[0].set_title('True vs Predicted Exam Scores', fontsize=14, fontweight='bold')
        axes[0].grid(True, alpha=0.3)
        
        residuals = y_true - y_pred
        axes[1].hist(residuals, bins=50, alpha=0.7, edgecolor='black')
        axes[1].axvline(x=0, color='r', linestyle='--', linewidth=2)
        axes[1].set_xlabel('Residual (True - Predicted)', fontsize=12)
        axes[1].set_ylabel('Frequency', fontsize=12)
        axes[1].set_title('Residual Distribution', fontsize=14, fontweight='bold')
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  ✓ 预测结果图表已保存至: {save_path}")
    except Exception as e:
        print(f"  ⚠️  绘图失败: {e}")


def train_model(X, y, input_size, device='cpu'):
    """训练模型"""
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=0.125, random_state=42
    )
    
    X_train_tensor = torch.FloatTensor(X_train)
    y_train_tensor = torch.FloatTensor(y_train)
    X_val_tensor = torch.FloatTensor(X_val)
    y_val_tensor = torch.FloatTensor(y_val)
    X_test_tensor = torch.FloatTensor(X_test)
    y_test_tensor = torch.FloatTensor(y_test)
    
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    model = Top10PredictorNN(input_size).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=5e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10, min_lr=1e-7)
    
    early_stopping = EarlyStopping(patience=20, min_delta=5e-5, verbose=True)
    
    train_losses = []
    val_losses = []
    
    for epoch in range(300):
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
            print(f"Epoch [{epoch+1:3d}/300] Train Loss: {train_loss:.4f} Val Loss: {val_loss:.4f} "
                  f"RMSE: {val_rmse:.4f} MAE: {val_mae:.4f} R²: {val_r2:.4f}")
        
        if early_stopping.early_stop:
            print(f"\n早停触发于 Epoch {epoch+1}")
            model.load_state_dict(early_stopping.best_model_state)
            break
    
    plot_training_curves(train_losses, val_losses)
    
    test_loss, test_preds, test_actuals = validate(model, test_loader, criterion, device)
    
    return model, test_loss, test_preds, test_actuals


def main():
    print("="*100)
    print("学生成绩预测 - Top 10特征模型".center(100))
    print("="*100)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n使用设备: {device}")
    
    try:
        top_10_features = np.load('top_10_features.npy', allow_pickle=True).tolist()
        print(f"\n从文件加载 Top 10 特征: {', '.join(top_10_features)}")
    except FileNotFoundError:
        print("\n未找到 top_10_features.npy 文件，使用默认的 Top 10 特征")
        top_10_features = [
            'Attendance', 'Hours_Studied', 'Previous_Scores', 'Tutoring_Sessions',
            'Access_to_Resources', 'Parental_Involvement', 'Sleep_Hours',
            'Physical_Activity', 'Family_Income', 'Peer_Influence'
        ]
        print(f"使用的特征: {', '.join(top_10_features)}")
    
    data_path = "./StudentPerformanceFactors .csv"
    X, y, feature_names, scaler = load_and_preprocess_data(data_path, top_10_features)
    
    print(f"\n数据加载完成:")
    print(f"  样本数: {X.shape[0]}")
    print(f"  使用的特征数: {X.shape[1]}")
    print(f"  特征列表: {', '.join(feature_names)}")
    
    print("\n" + "="*100)
    print("开始训练模型".center(100))
    print("="*100)
    
    model, test_loss, test_preds, test_actuals = train_model(X, y, X.shape[1], device)
    
    print("\n" + "="*100)
    print("模型评估结果".center(100))
    print("="*100)
    
    test_rmse = np.sqrt(test_loss)
    test_mae = mean_absolute_error(test_actuals, test_preds)
    test_r2 = r2_score(test_actuals, test_preds)
    
    print(f"\n测试集性能指标:")
    print(f"  均方误差 (MSE):   {test_loss:.6f}")
    print(f"  均方根误差 (RMSE): {test_rmse:.6f}")
    print(f"  平均绝对误差 (MAE):{test_mae:.6f}")
    print(f"  R² 分数:           {test_r2:.6f}")
    
    plot_predictions(test_actuals, test_preds)
    
    print("\n" + "="*100)
    print("K折交叉验证".center(100))
    print("="*100)
    
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    fold_mse_scores = []
    fold_rmse_scores = []
    fold_mae_scores = []
    fold_r2_scores = []
    
    for fold, (train_idx, test_idx) in enumerate(kf.split(X), 1):
        print(f"\nFold {fold}/5")
        print("-"*60)
        
        X_train_fold, X_test_fold = X[train_idx], X[test_idx]
        y_train_fold, y_test_fold = y[train_idx], y[test_idx]
        
        X_train_fold, X_val_fold, y_train_fold, y_val_fold = train_test_split(
            X_train_fold, y_train_fold, test_size=0.125, random_state=42
        )
        
        X_train_tensor = torch.FloatTensor(X_train_fold)
        y_train_tensor = torch.FloatTensor(y_train_fold)
        X_val_tensor = torch.FloatTensor(X_val_fold)
        y_val_tensor = torch.FloatTensor(y_val_fold)
        X_test_tensor = torch.FloatTensor(X_test_fold)
        y_test_tensor = torch.FloatTensor(y_test_fold)
        
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
        test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
        
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
        
        fold_model = Top10PredictorNN(X.shape[1]).to(device)
        criterion = nn.MSELoss()
        optimizer = optim.AdamW(fold_model.parameters(), lr=0.001, weight_decay=5e-5)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10, min_lr=1e-7)
        early_stopping = EarlyStopping(patience=20, min_delta=5e-5, verbose=False)
        
        for epoch in range(300):
            train_loss = train_one_epoch(fold_model, train_loader, criterion, optimizer, device)
            val_loss, _, _ = validate(fold_model, val_loader, criterion, device)
            scheduler.step(val_loss)
            early_stopping(val_loss, fold_model)
            if early_stopping.early_stop:
                fold_model.load_state_dict(early_stopping.best_model_state)
                break
        
        test_loss, test_preds, test_actuals = validate(fold_model, test_loader, criterion, device)
        fold_rmse = np.sqrt(test_loss)
        fold_mae = mean_absolute_error(test_actuals, test_preds)
        fold_r2 = r2_score(test_actuals, test_preds)
        
        fold_mse_scores.append(test_loss)
        fold_rmse_scores.append(fold_rmse)
        fold_mae_scores.append(fold_mae)
        fold_r2_scores.append(fold_r2)
        
        print(f"  Fold {fold} - MSE: {test_loss:.6f}, RMSE: {fold_rmse:.6f}, MAE: {fold_mae:.6f}, R²: {fold_r2:.6f}")
    
    print(f"\n5折交叉验证结果汇总:")
    print(f"  MSE:  {np.mean(fold_mse_scores):.6f} ± {np.std(fold_mse_scores):.6f}")
    print(f"  RMSE: {np.mean(fold_rmse_scores):.6f} ± {np.std(fold_rmse_scores):.6f}")
    print(f"  MAE:  {np.mean(fold_mae_scores):.6f} ± {np.std(fold_mae_scores):.6f}")
    print(f"  R²:   {np.mean(fold_r2_scores):.6f} ± {np.std(fold_r2_scores):.6f}")
    
    torch.save({
        'model_state_dict': model.state_dict(),
        'feature_names': feature_names,
        'scaler_mean': scaler.mean_,
        'scaler_scale': scaler.scale_,
        'test_metrics': {
            'test_mse': test_loss,
            'test_rmse': test_rmse,
            'test_mae': test_mae,
            'test_r2': test_r2
        }
    }, 'top10_predictor_model.pth')
    
    print(f"\n✓ 模型已保存至: top10_predictor_model.pth")
    
    print("\n" + "="*100)
    print("任务完成!".center(100))
    print("="*100)


if __name__ == "__main__":
    main()
