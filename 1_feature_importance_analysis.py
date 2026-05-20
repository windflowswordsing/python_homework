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


def load_and_preprocess_data(filepath):
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
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X.values)
    
    return X_scaled, y, feature_names


class FeatureImportanceNN(nn.Module):
    """用于特征重要性分析的神经网络"""
    def __init__(self, input_size):
        super(FeatureImportanceNN, self).__init__()
        
        self.network = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(64, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(64, 1)
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


def train_nn_for_importance(X, y, input_size, device='cpu'):
    """训练神经网络用于特征重要性分析"""
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    X_train_tensor = torch.FloatTensor(X_train)
    y_train_tensor = torch.FloatTensor(y_train)
    X_val_tensor = torch.FloatTensor(X_val)
    y_val_tensor = torch.FloatTensor(y_val)
    
    model = FeatureImportanceNN(input_size).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=5e-5)
    
    best_val_loss = float('inf')
    patience = 15
    patience_counter = 0
    
    for epoch in range(100):
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train_tensor)
        loss = criterion(outputs.squeeze(), y_train_tensor)
        loss.backward()
        optimizer.step()
        
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val_tensor)
            val_loss = criterion(val_outputs.squeeze(), y_val_tensor)
        
        if val_loss < best_val_loss - 1e-5:
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break
    
    return model


def permutation_importance(model, X, y, feature_names, device='cpu', n_repeats=5):
    """计算排列特征重要性"""
    model.eval()
    X_tensor = torch.FloatTensor(X)
    y_tensor = torch.FloatTensor(y)
    
    with torch.no_grad():
        baseline_preds = model(X_tensor).squeeze().cpu().numpy()
        baseline_mse = mean_squared_error(y, baseline_preds)
    
    importance_scores = []
    
    for i in range(X.shape[1]):
        X_permuted = X.copy()
        scores = []
        
        for _ in range(n_repeats):
            np.random.shuffle(X_permuted[:, i])
            X_permuted_tensor = torch.FloatTensor(X_permuted)
            
            with torch.no_grad():
                permuted_preds = model(X_permuted_tensor).squeeze().cpu().numpy()
                permuted_mse = mean_squared_error(y, permuted_preds)
            
            score = permuted_mse - baseline_mse
            scores.append(score)
        
        importance_scores.append(np.mean(scores))
    
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importance_scores
    }).sort_values('Importance', ascending=False)
    
    return importance_df


def analyze_feature_importance(X, y, feature_names):
    """综合分析特征重要性"""
    print("\n" + "="*80)
    print("特征重要性分析".center(80))
    print("="*80)
    
    rf = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
    rf.fit(X, y)
    rf_importance = pd.DataFrame({
        'Feature': feature_names,
        'RF_Importance': rf.feature_importances_
    })
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = train_nn_for_importance(X, y, X.shape[1], device)
    nn_importance = permutation_importance(model, X, y, feature_names, device)
    
    combined_importance = rf_importance.merge(nn_importance, on='Feature')
    combined_importance['Normalized_RF'] = combined_importance['RF_Importance'] / combined_importance['RF_Importance'].max()
    combined_importance['Normalized_NN'] = combined_importance['Importance'] / combined_importance['Importance'].max()
    combined_importance['Combined_Score'] = (combined_importance['Normalized_RF'] + combined_importance['Normalized_NN']) / 2
    combined_importance = combined_importance.sort_values('Combined_Score', ascending=False).reset_index(drop=True)
    
    print("\n所有19个特征的重要性排序：")
    print("-"*80)
    print(f"{'排名':<5}{'特征名称':<35}{'综合得分':<15}{'随机森林':<15}{'神经网络':<15}")
    print("-"*80)
    for idx, row in combined_importance.iterrows():
        print(f"{idx+1:<5}{row.Feature:<35}{row.Combined_Score:<15.6f}{row.RF_Importance:<15.6f}{row.Importance:<15.6f}")
    
    top_10 = combined_importance.head(10)
    print("\nTop 10 重要特征：")
    print("-"*80)
    for idx, row in enumerate(top_10.itertuples(), 1):
        print(f"{idx:2d}. {row.Feature:30s} 综合得分: {row.Combined_Score:.6f}")
    
    if PLOTTING_AVAILABLE:
        try:
            fig, axes = plt.subplots(2, 1, figsize=(16, 16))
            
            colors = plt.cm.viridis(np.linspace(0, 1, len(combined_importance)))
            bars1 = axes[0].barh(range(len(combined_importance)), combined_importance['Combined_Score'][::-1], 
                               color=colors[::-1], alpha=0.8)
            axes[0].set_yticks(range(len(combined_importance)))
            axes[0].set_yticklabels(combined_importance['Feature'][::-1], fontsize=10)
            axes[0].set_xlabel('Combined Importance Score', fontsize=12)
            axes[0].set_title('All 19 Features - Combined Feature Importance (Random Forest + Neural Network)', 
                            fontsize=14, fontweight='bold', pad=20)
            axes[0].grid(axis='x', alpha=0.3)
            
            for i, bar in enumerate(bars1):
                width = bar.get_width()
                axes[0].text(width, bar.get_y() + bar.get_height()/2, 
                           f'{width:.4f}', va='center', fontsize=8)
            
            colors_top = plt.cm.viridis(np.linspace(0, 1, 10))
            bars2 = axes[1].barh(range(10), top_10['Combined_Score'][::-1], 
                               color=colors_top[::-1], alpha=0.8)
            axes[1].set_yticks(range(10))
            axes[1].set_yticklabels(top_10['Feature'][::-1], fontsize=11)
            axes[1].set_xlabel('Combined Importance Score', fontsize=12)
            axes[1].set_title('Top 10 Important Features', fontsize=14, fontweight='bold', pad=20)
            axes[1].grid(axis='x', alpha=0.3)
            
            for i, bar in enumerate(bars2):
                width = bar.get_width()
                axes[1].text(width, bar.get_y() + bar.get_height()/2, 
                           f'{width:.4f}', va='center', fontsize=10)
            
            plt.tight_layout()
            plt.savefig('feature_importance_analysis.png', dpi=300, bbox_inches='tight')
            plt.close()
            print(f"\n✓ 特征重要性图表已保存至: feature_importance_analysis.png")
        except Exception as e:
            print(f"\n⚠️  绘图失败: {e}")
    
    return combined_importance


def main():
    print("="*100)
    print("学生成绩预测 - 特征重要性分析".center(100))
    print("="*100)
    
    data_path = "/workspace/方案/StudentPerformanceFactors .csv"
    
    X, y, feature_names = load_and_preprocess_data(data_path)
    
    print(f"\n数据加载完成:")
    print(f"  样本数: {X.shape[0]}")
    print(f"  特征数: {X.shape[1]}")
    print(f"  特征列表: {', '.join(feature_names)}")
    
    importance_df = analyze_feature_importance(X, y, feature_names)
    
    top_10_features = importance_df.head(10)['Feature'].tolist()
    
    print("\n" + "="*100)
    print("Top 10 重要特征总结".center(100))
    print("="*100)
    print("\n影响最大的10个特征（按重要性排序）：")
    for i, feature in enumerate(top_10_features, 1):
        print(f"  {i}. {feature}")
    
    np.save('top_10_features.npy', top_10_features)
    print(f"\n✓ Top 10 特征已保存至: top_10_features.npy")
    
    print("\n" + "="*100)
    print("任务完成!".center(100))
    print("="*100)


if __name__ == "__main__":
    main()
