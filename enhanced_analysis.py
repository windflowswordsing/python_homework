import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split, KFold, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor
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

def load_and_preprocess_data(filepath, use_all_features=False):
    """加载和预处理数据"""
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
    
    if not use_all_features:
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
    
    return X_scaled, y, selected_features, feature_names, scaler

def analyze_feature_importance(X, y, all_feature_names, save_path='feature_importance.png'):
    """分析并可视化特征重要性"""
    print("\n" + "="*80)
    print("特征重要性分析".center(80))
    print("="*80)
    
    rf = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
    rf.fit(X, y)
    
    importance_df = pd.DataFrame({
        'Feature': all_feature_names,
        'Importance': rf.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    print("\n所有19个特征的重要性排序：")
    print("-"*80)
    for idx, row in importance_df.iterrows():
        rank = len(importance_df) - idx
        print(f"{rank:2d}. {row.Feature:30s} {row.Importance:.6f}")
    
    top_10 = importance_df.head(10)
    print("\nTop 10 重要特征：")
    print("-"*80)
    for idx, row in enumerate(top_10.itertuples(), 1):
        print(f"{idx:2d}. {row.Feature:30s} {row.Importance:.6f}")
    
    if PLOTTING_AVAILABLE:
        try:
            fig, axes = plt.subplots(1, 2, figsize=(20, 8))
            
            colors = plt.cm.viridis(np.linspace(0, 1, len(importance_df)))
            axes[0].barh(range(len(importance_df)), importance_df['Importance'][::-1], 
                        color=colors[::-1], alpha=0.8)
            axes[0].set_yticks(range(len(importance_df)))
            axes[0].set_yticklabels(importance_df['Feature'][::-1])
            axes[0].set_xlabel('Importance Score', fontsize=12)
            axes[0].set_title('All 19 Features - Feature Importance', fontsize=14, fontweight='bold')
            axes[0].grid(axis='x', alpha=0.3)
            
            colors_top = plt.cm.viridis(np.linspace(0, 1, 10))
            axes[1].barh(range(10), top_10['Importance'][::-1], 
                        color=colors_top[::-1], alpha=0.8)
            axes[1].set_yticks(range(10))
            axes[1].set_yticklabels(top_10['Feature'][::-1])
            axes[1].set_xlabel('Importance Score', fontsize=12)
            axes[1].set_title('Top 10 Important Features', fontsize=14, fontweight='bold')
            axes[1].grid(axis='x', alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"\n✓ 特征重要性图表已保存至: {save_path}")
        except Exception as e:
            print(f"\n⚠️  绘图失败: {e}")
    
    return importance_df

class RegularizedNeuralNetwork(nn.Module):
    def __init__(self, input_size, hidden_dims=[96, 192, 96, 48], dropout_rates=[0.25, 0.35, 0.25, 0.15]):
        super(RegularizedNeuralNetwork, self).__init__()
        
        layers = []
        prev_dim = input_size
        
        for hidden_dim, dropout_rate in zip(hidden_dims, dropout_rates):
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, 1))
        
        self.network = nn.Sequential(*layers)
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        return self.network(x)

def train_evaluate_model(X_train, y_train, X_val, y_val, X_test, y_test, 
                       hidden_dims, dropout_rates, learning_rate, weight_decay, 
                       batch_size=32, num_epochs=200, device='cpu'):
    """训练并评估模型"""
    
    X_train_tensor = torch.FloatTensor(X_train)
    y_train_tensor = torch.FloatTensor(y_train)
    X_val_tensor = torch.FloatTensor(X_val)
    y_val_tensor = torch.FloatTensor(y_val)
    X_test_tensor = torch.FloatTensor(X_test)
    y_test_tensor = torch.FloatTensor(y_test)
    
    train_loader = DataLoader(TensorDataset(X_train_tensor, y_train_tensor), 
                             batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val_tensor, y_val_tensor), 
                           batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(TensorDataset(X_test_tensor, y_test_tensor), 
                            batch_size=batch_size, shuffle=False)
    
    model = RegularizedNeuralNetwork(input_size=X_train.shape[1], 
                                    hidden_dims=hidden_dims, 
                                    dropout_rates=dropout_rates).to(device)
    
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', 
                                                     factor=0.5, patience=10, min_lr=1e-7)
    
    best_val_loss = float('inf')
    best_model_state = None
    patience_counter = 0
    patience = 20
    
    train_losses = []
    val_losses = []
    
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs.squeeze(), batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_loss += loss.item() * batch_X.size(0)
        
        train_loss /= len(train_loader.dataset)
        train_losses.append(train_loss)
        
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                outputs = model(batch_X)
                loss = criterion(outputs.squeeze(), batch_y)
                val_loss += loss.item() * batch_X.size(0)
        
        val_loss /= len(val_loader.dataset)
        val_losses.append(val_loss)
        
        scheduler.step(val_loss)
        
        if val_loss < best_val_loss - 1e-5:
            best_val_loss = val_loss
            best_model_state = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break
    
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    model.eval()
    test_preds = []
    with torch.no_grad():
        for batch_X, _ in test_loader:
            batch_X = batch_X.to(device)
            outputs = model(batch_X)
            test_preds.extend(outputs.squeeze().cpu().numpy())
    
    test_preds = np.array(test_preds)
    test_mse = mean_squared_error(y_test, test_preds)
    test_rmse = np.sqrt(test_mse)
    test_mae = mean_absolute_error(y_test, test_preds)
    test_r2 = r2_score(y_test, test_preds)
    
    return {
        'test_mse': test_mse,
        'test_rmse': test_rmse,
        'test_mae': test_mae,
        'test_r2': test_r2,
        'best_val_loss': best_val_loss,
        'model': model
    }

def hyperparameter_grid_search(X, y, param_grid, n_splits=3):
    """超参数网格搜索"""
    print("\n" + "="*80)
    print("超参数网格搜索".center(80))
    print("="*80)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    best_score = -float('inf')
    best_params = None
    all_results = []
    
    param_combinations = []
    for hd in param_grid['hidden_dims']:
        for dr in param_grid['dropout_rates']:
            for lr in param_grid['learning_rate']:
                for wd in param_grid['weight_decay']:
                    for bs in param_grid['batch_size']:
                        param_combinations.append({
                            'hidden_dims': hd,
                            'dropout_rates': dr,
                            'learning_rate': lr,
                            'weight_decay': wd,
                            'batch_size': bs
                        })
    
    print(f"\n共 {len(param_combinations)} 组参数组合待搜索...")
    print("-"*80)
    
    for idx, params in enumerate(param_combinations, 1):
        print(f"\n[{idx}/{len(param_combinations)}] 测试参数组合:")
        print(f"  Hidden Dims: {params['hidden_dims']}")
        print(f"  Dropout Rates: {params['dropout_rates']}")
        print(f"  Learning Rate: {params['learning_rate']}")
        print(f"  Weight Decay: {params['weight_decay']}")
        print(f"  Batch Size: {params['batch_size']}")
        
        fold_scores = []
        for fold, (train_idx, val_idx) in enumerate(kf.split(X_train_val), 1):
            X_train, X_val = X_train_val[train_idx], X_train_val[val_idx]
            y_train, y_val = y_train_val[train_idx], y_train_val[val_idx]
            
            result = train_evaluate_model(
                X_train, y_train, X_val, y_val, X_test, y_test,
                hidden_dims=params['hidden_dims'],
                dropout_rates=params['dropout_rates'],
                learning_rate=params['learning_rate'],
                weight_decay=params['weight_decay'],
                batch_size=params['batch_size'],
                num_epochs=150,
                device=device
            )
            
            fold_scores.append(result['test_r2'])
        
        avg_r2 = np.mean(fold_scores)
        std_r2 = np.std(fold_scores)
        
        all_results.append({
            'params': params,
            'avg_r2': avg_r2,
            'std_r2': std_r2
        })
        
        print(f"  平均 R²: {avg_r2:.6f} (±{std_r2:.6f})")
        
        if avg_r2 > best_score:
            best_score = avg_r2
            best_params = params
            print(f"  ✓ 发现新的最佳参数!")
    
    print("\n" + "="*80)
    print("网格搜索结果汇总".center(80))
    print("="*80)
    
    all_results_sorted = sorted(all_results, key=lambda x: x['avg_r2'], reverse=True)
    print("\nTop 5 最佳参数组合：")
    print("-"*80)
    for i, result in enumerate(all_results_sorted[:5], 1):
        print(f"\n{i}. R² = {result['avg_r2']:.6f} (±{result['std_r2']:.6f})")
        print(f"   Params: {result['params']}")
    
    print("\n" + "="*80)
    print("最佳参数".center(80))
    print("="*80)
    print(f"\n{best_params}")
    print(f"\n最佳 R²: {best_score:.6f}")
    
    return best_params, all_results_sorted

def main():
    print("="*100)
    print("学生成绩预测 - 增强分析版本".center(100))
    print("="*100)
    
    data_path = "/workspace/方案/StudentPerformanceFactors .csv"
    
    X_all, y, all_feature_names, _, _ = load_and_preprocess_data(data_path, use_all_features=True)
    
    importance_df = analyze_feature_importance(X_all, y, all_feature_names, 
                                              save_path='feature_importance_analysis.png')
    
    X_top10, y, selected_features, _, scaler = load_and_preprocess_data(data_path, use_all_features=False)
    
    print("\n" + "="*100)
    print("使用Top 10特征训练基线模型".center(100))
    print("="*100)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X_top10, y, test_size=0.2, random_state=42
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=0.1/0.8, random_state=42
    )
    
    baseline_result = train_evaluate_model(
        X_train, y_train, X_val, y_val, X_test, y_test,
        hidden_dims=[96, 192, 96, 48],
        dropout_rates=[0.25, 0.35, 0.25, 0.15],
        learning_rate=0.001,
        weight_decay=5e-5,
        batch_size=32,
        num_epochs=200,
        device=device
    )
    
    print("\n基线模型性能：")
    print(f"  MSE: {baseline_result['test_mse']:.6f}")
    print(f"  RMSE: {baseline_result['test_rmse']:.6f}")
    print(f"  MAE: {baseline_result['test_mae']:.6f}")
    print(f"  R²: {baseline_result['test_r2']:.6f}")
    
    param_grid = {
        'hidden_dims': [
            [64, 128, 64, 32],
            [96, 192, 96, 48],
            [128, 256, 128, 64]
        ],
        'dropout_rates': [
            [0.2, 0.3, 0.2, 0.1],
            [0.25, 0.35, 0.25, 0.15],
            [0.3, 0.4, 0.3, 0.2]
        ],
        'learning_rate': [0.001, 0.0005],
        'weight_decay': [1e-5, 5e-5, 1e-4],
        'batch_size': [32, 64]
    }
    
    best_params, all_results = hyperparameter_grid_search(X_top10, y, param_grid, n_splits=3)
    
    print("\n" + "="*100)
    print("使用最佳参数重新训练完整模型".center(100))
    print("="*100)
    
    final_result = train_evaluate_model(
        X_train, y_train, X_val, y_val, X_test, y_test,
        hidden_dims=best_params['hidden_dims'],
        dropout_rates=best_params['dropout_rates'],
        learning_rate=best_params['learning_rate'],
        weight_decay=best_params['weight_decay'],
        batch_size=best_params['batch_size'],
        num_epochs=250,
        device=device
    )
    
    print("\n最终模型性能（最佳参数）：")
    print("-"*80)
    print(f"  MSE: {final_result['test_mse']:.6f}")
    print(f"  RMSE: {final_result['test_rmse']:.6f}")
    print(f"  MAE: {final_result['test_mae']:.6f}")
    print(f"  R²: {final_result['test_r2']:.6f}")
    
    print("\n" + "="*100)
    print("改进对比".center(100))
    print("="*100)
    print(f"  基线模型 R²: {baseline_result['test_r2']:.6f}")
    print(f"  优化后 R²: {final_result['test_r2']:.6f}")
    improvement = (final_result['test_r2'] - baseline_result['test_r2']) * 100
    print(f"  改进: {improvement:+.2f}%")
    
    torch.save({
        'model_state_dict': final_result['model'].state_dict(),
        'feature_names': selected_features,
        'scaler_mean': scaler.mean_,
        'scaler_scale': scaler.scale_,
        'best_params': best_params,
        'test_metrics': {
            'test_mse': final_result['test_mse'],
            'test_rmse': final_result['test_rmse'],
            'test_mae': final_result['test_mae'],
            'test_r2': final_result['test_r2']
        }
    }, 'optimized_model.pth')
    
    print("\n✓ 优化后的模型已保存至: optimized_model.pth")
    print("\n" + "="*100)
    print("任务完成!".center(100))
    print("="*100)

if __name__ == "__main__":
    main()

