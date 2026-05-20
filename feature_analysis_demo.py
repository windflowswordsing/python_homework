import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
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

def analyze_feature_importance(X, y, all_feature_names, save_path='feature_importance_analysis.png'):
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
            fig, axes = plt.subplots(2, 1, figsize=(16, 18))
            
            colors = plt.cm.viridis(np.linspace(0, 1, len(importance_df)))
            bars1 = axes[0].barh(range(len(importance_df)), importance_df['Importance'][::-1], 
                               color=colors[::-1], alpha=0.8)
            axes[0].set_yticks(range(len(importance_df)))
            axes[0].set_yticklabels(importance_df['Feature'][::-1], fontsize=11)
            axes[0].set_xlabel('Importance Score', fontsize=13)
            axes[0].set_title('All 19 Features - Feature Importance', fontsize=15, fontweight='bold', pad=20)
            axes[0].grid(axis='x', alpha=0.3)
            
            for i, bar in enumerate(bars1):
                width = bar.get_width()
                axes[0].text(width, bar.get_y() + bar.get_height()/2, 
                           f'{width:.4f}', va='center', fontsize=9)
            
            colors_top = plt.cm.viridis(np.linspace(0, 1, 10))
            bars2 = axes[1].barh(range(10), top_10['Importance'][::-1], 
                               color=colors_top[::-1], alpha=0.8)
            axes[1].set_yticks(range(10))
            axes[1].set_yticklabels(top_10['Feature'][::-1], fontsize=11)
            axes[1].set_xlabel('Importance Score', fontsize=13)
            axes[1].set_title('Top 10 Important Features', fontsize=15, fontweight='bold', pad=20)
            axes[1].grid(axis='x', alpha=0.3)
            
            for i, bar in enumerate(bars2):
                width = bar.get_width()
                axes[1].text(width, bar.get_y() + bar.get_height()/2, 
                           f'{width:.4f}', va='center', fontsize=9)
            
            plt.tight_layout()
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"\n✓ 特征重要性图表已保存至: {save_path}")
        except Exception as e:
            print(f"\n⚠️  绘图失败: {e}")
    
    return importance_df

def create_comparison_plot(X, y, importance_df, save_path='feature_comparison.png'):
    """创建特征重要性对比图"""
    if not PLOTTING_AVAILABLE:
        return
    
    try:
        fig = plt.figure(figsize=(18, 12))
        
        gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
        
        ax1 = fig.add_subplot(gs[0, :])
        importance_sorted = importance_df.sort_values('Importance', ascending=True)
        colors = plt.cm.plasma(np.linspace(0, 1, len(importance_sorted)))
        bars = ax1.barh(range(len(importance_sorted)), importance_sorted['Importance'], 
                       color=colors, alpha=0.8)
        ax1.set_yticks(range(len(importance_sorted)))
        ax1.set_yticklabels(importance_sorted['Feature'], fontsize=10)
        ax1.set_xlabel('Importance Score', fontsize=12)
        ax1.set_title('Feature Importance - Complete Ranking', fontsize=14, fontweight='bold')
        ax1.grid(axis='x', alpha=0.3)
        
        ax2 = fig.add_subplot(gs[1, 0])
        top_10 = importance_df.head(10)
        ax2.pie(top_10['Importance'], labels=top_10['Feature'], 
               autopct='%1.1f%%', startangle=90, colors=plt.cm.Set3(np.linspace(0, 1, 10)))
        ax2.set_title('Top 10 Features - Contribution Distribution', fontsize=12, fontweight='bold')
        
        ax3 = fig.add_subplot(gs[1, 1])
        cumulative = np.cumsum(importance_df['Importance'].values)
        ax3.plot(range(1, len(cumulative)+1), cumulative, 'o-', linewidth=2, markersize=8, color='darkblue')
        ax3.set_xlabel('Number of Features', fontsize=12)
        ax3.set_ylabel('Cumulative Importance', fontsize=12)
        ax3.set_title('Cumulative Feature Importance', fontsize=12, fontweight='bold')
        ax3.grid(True, alpha=0.3)
        ax3.axhline(y=0.8, color='red', linestyle='--', label='80% Importance')
        ax3.legend()
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✓ 综合对比图已保存至: {save_path}")
    except Exception as e:
        print(f"⚠️  绘图失败: {e}")

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

def train_model(X, y, input_size, hidden_dims, dropout_rates, learning_rate, weight_decay, 
               batch_size=32, num_epochs=200, device='cpu'):
    """训练模型"""
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.125, random_state=42
    )
    
    X_train_tensor = torch.FloatTensor(X_train)
    y_train_tensor = torch.FloatTensor(y_train)
    X_val_tensor = torch.FloatTensor(X_val)
    y_val_tensor = torch.FloatTensor(y_val)
    X_test_tensor = torch.FloatTensor(X_test)
    y_test_tensor = torch.FloatTensor(y_test)
    
    from torch.utils.data import DataLoader, TensorDataset
    train_loader = DataLoader(TensorDataset(X_train_tensor, y_train_tensor), 
                             batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val_tensor, y_val_tensor), 
                           batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(TensorDataset(X_test_tensor, y_test_tensor), 
                            batch_size=batch_size, shuffle=False)
    
    model = RegularizedNeuralNetwork(input_size=input_size, 
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
        'train_losses': train_losses,
        'val_losses': val_losses
    }

def main():
    print("="*100)
    print("学生成绩预测 - 特征重要性分析与可视化".center(100))
    print("="*100)
    
    data_path = "/workspace/方案/StudentPerformanceFactors .csv"
    
    X_all, y, all_feature_names, _, _ = load_and_preprocess_data(data_path, use_all_features=True)
    
    importance_df = analyze_feature_importance(X_all, y, all_feature_names, 
                                              save_path='feature_importance_detailed.png')
    
    create_comparison_plot(X_all, y, importance_df, save_path='feature_comprehensive_analysis.png')
    
    X_top10, y, selected_features, _, scaler = load_and_preprocess_data(data_path, use_all_features=False)
    
    print("\n" + "="*100)
    print("模型训练与性能评估".center(100))
    print("="*100)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    configs = [
        {
            'name': 'Small Network',
            'hidden_dims': [64, 128, 64, 32],
            'dropout_rates': [0.2, 0.3, 0.2, 0.1],
            'learning_rate': 0.001,
            'weight_decay': 5e-5,
            'batch_size': 32
        },
        {
            'name': 'Medium Network',
            'hidden_dims': [96, 192, 96, 48],
            'dropout_rates': [0.25, 0.35, 0.25, 0.15],
            'learning_rate': 0.001,
            'weight_decay': 5e-5,
            'batch_size': 32
        },
        {
            'name': 'Large Network',
            'hidden_dims': [128, 256, 128, 64],
            'dropout_rates': [0.3, 0.4, 0.3, 0.2],
            'learning_rate': 0.0005,
            'weight_decay': 1e-4,
            'batch_size': 64
        }
    ]
    
    results = []
    for config in configs:
        print(f"\n{'='*60}")
        print(f"训练: {config['name']}".center(60))
        print(f"{'='*60}")
        
        result = train_model(
            X_top10, y, 
            input_size=X_top10.shape[1],
            hidden_dims=config['hidden_dims'],
            dropout_rates=config['dropout_rates'],
            learning_rate=config['learning_rate'],
            weight_decay=config['weight_decay'],
            batch_size=config['batch_size'],
            num_epochs=200,
            device=device
        )
        
        result['config'] = config
        results.append(result)
        
        print(f"\n性能指标:")
        print(f"  MSE: {result['test_mse']:.6f}")
        print(f"  RMSE: {result['test_rmse']:.6f}")
        print(f"  MAE: {result['test_mae']:.6f}")
        print(f"  R²: {result['test_r2']:.6f}")
    
    print("\n" + "="*100)
    print("性能对比总结".center(100))
    print("="*100)
    print(f"\n{'模型名称':<20} {'MSE':<12} {'RMSE':<12} {'MAE':<12} {'R²':<12}")
    print("-"*72)
    
    best_result = max(results, key=lambda x: x['test_r2'])
    for result in results:
        is_best = " ✓" if result == best_result else ""
        print(f"{result['config']['name']:<20} {result['test_mse']:<12.4f} {result['test_rmse']:<12.4f} "
              f"{result['test_mae']:<12.4f} {result['test_r2']:<12.4f}{is_best}")
    
    print("\n" + "="*100)
    print("最佳模型".center(100))
    print("="*100)
    print(f"\n{best_result['config']['name']}")
    print(f"  MSE: {best_result['test_mse']:.6f}")
    print(f"  RMSE: {best_result['test_rmse']:.6f}")
    print(f"  MAE: {best_result['test_mae']:.6f}")
    print(f"  R²: {best_result['test_r2']:.6f}")
    
    if PLOTTING_AVAILABLE:
        try:
            fig, ax = plt.subplots(figsize=(12, 8))
            model_names = [r['config']['name'] for r in results]
            r2_scores = [r['test_r2'] for r in results]
            
            colors = plt.cm.RdYlGn(np.linspace(0.3, 1, len(results)))
            bars = ax.bar(model_names, r2_scores, color=colors, alpha=0.8)
            
            ax.set_ylabel('R² Score', fontsize=12)
            ax.set_title('Model Performance Comparison', fontsize=14, fontweight='bold', pad=20)
            ax.grid(axis='y', alpha=0.3)
            
            for i, (bar, score) in enumerate(zip(bars, r2_scores)):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{score:.4f}',
                       ha='center', va='bottom', fontsize=11)
            
            plt.tight_layout()
            plt.savefig('model_comparison.png', dpi=300, bbox_inches='tight')
            plt.close()
            print(f"\n✓ 模型对比图已保存至: model_comparison.png")
        except Exception as e:
            print(f"\n⚠️  绘图失败: {e}")
    
    print("\n" + "="*100)
    print("任务完成!".center(100))
    print("="*100)

if __name__ == "__main__":
    main()

