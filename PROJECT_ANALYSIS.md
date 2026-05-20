# 学生成绩预测项目分析文档

## 项目概述
本项目旨在通过机器学习和深度学习方法，分析影响学生考试成绩的关键因素，并构建预测模型。

## 1. 数据分析

### 1.1 数据集信息
- **数据集**: StudentPerformanceFactors.csv
- **样本数量**: 6,607条
- **特征数量**: 19个
- **目标变量**: Exam_Score（考试成绩）

### 1.2 特征列表
1.  Attendance（出勤率）
2.  Hours_Studied（学习时长）
3.  Previous_Scores（过往成绩）
4.  Tutoring_Sessions（辅导课程数）
5.  Access_to_Resources（资源获取情况）
6.  Parental_Involvement（家长参与度）
7.  Sleep_Hours（睡眠时长）
8.  Physical_Activity（体育活动）
9.  Family_Income（家庭收入）
10. Peer_Influence（同伴影响）
11. Distance_from_Home（家到学校距离）
12. Parental_Education_Level（家长教育水平）
13. Teacher_Quality（教师质量）
14. Motivation_Level（学习动机）
15. Learning_Disabilities（学习障碍）
16. School_Type（学校类型）
17. Extracurricular_Activities（课外活动）
18. Gender（性别）
19. Internet_Access（网络访问）

### 1.3 特征重要性分析
使用随机森林算法对19个特征进行重要性评估，结果如下：

#### Top 10 关键特征
| 排名 | 特征名称 | 重要性评分 |
|------|----------|------------|
| 1 | **Attendance** | 0.3852 |
| 2 | **Hours_Studied** | 0.2474 |
| 3 | Previous_Scores | 0.0880 |
| 4 | Tutoring_Sessions | 0.0355 |
| 5 | Access_to_Resources | 0.0301 |
| 6 | Parental_Involvement | 0.0290 |
| 7 | Sleep_Hours | 0.0269 |
| 8 | Physical_Activity | 0.0263 |
| 9 | Family_Income | 0.0177 |
| 10 | Peer_Influence | 0.0164 |

#### 关键发现
- **出勤率**是影响学生成绩最重要的因素（占比38.5%）
- **学习时长**位居第二（占比24.7%）
- 前2个特征合计贡献了超过60%的重要性
- 前10个特征合计贡献了约90%的重要性
- 性别、网络访问等因素对成绩影响较小

---

## 2. feature_analysis_demo.py 代码思路

### 2.1 功能概述
此脚本专注于**特征重要性分析和可视化**，以及**多种模型架构对比**。

### 2.2 核心模块

#### 2.2.1 数据加载与预处理 (`load_and_preprocess_data`)
```python
- 读取CSV数据
- 对分类变量进行LabelEncoder编码
- 支持使用全部19个特征或Top 10特征
- 使用StandardScaler进行特征标准化
```

#### 2.2.2 特征重要性分析 (`analyze_feature_importance`)
```python
- 使用随机森林回归器计算特征重要性
- 输出所有特征的重要性排序
- 生成详细的水平条形图可视化
```

#### 2.2.3 综合可视化 (`create_comparison_plot`)
```python
创建三图合一的综合分析图：
1. 完整特征重要性排序（水平条形图）
2. Top 10特征贡献分布（饼图）
3. 累积重要性曲线（标注80%重要性线）
```

#### 2.2.4 神经网络模型 (`RegularizedNeuralNetwork`)
```python
- 灵活的网络架构（支持自定义隐藏层维度）
- 每层包含：Linear -> BatchNorm -> ReLU -> Dropout
- 使用Kaiming初始化确保训练稳定
```

#### 2.2.5 模型训练与评估 (`train_model`)
```python
- 数据划分：训练集(70%)、验证集(10%)、测试集(20%)
- 使用AdamW优化器（带权重衰减）
- 学习率调度（ReduceLROnPlateau）
- 早停机制防止过拟合
- 梯度裁剪提升训练稳定性
```

#### 2.2.6 多种架构对比
测试三种不同规模的网络：
1. **Small Network**: [64, 128, 64, 32]
2. **Medium Network**: [96, 192, 96, 48] ⭐（最佳）
3. **Large Network**: [128, 256, 128, 64]

### 2.3 输出文件
- `feature_importance_detailed.png` - 特征重要性详细图表
- `feature_comprehensive_analysis.png` - 综合分析图
- `model_comparison.png` - 模型性能对比图

---

## 3. advanced_nn_regularization.py 代码思路

### 3.1 功能概述
此脚本专注于**高级正则化技术**和**K折交叉验证**，实现更稳健的模型训练。

### 3.2 高级正则化技术

#### 3.2.1 L1正则化 (Lasso)
```python
- 在损失函数中添加L1惩罚项
- λ系数：1e-6
- 作用：促使权重稀疏化，自动特征选择
```

#### 3.2.2 L2正则化 (Weight Decay)
```python
- 通过AdamW优化器实现
- 权重衰减系数：5e-5
- 作用：防止权重过大，增强泛化能力
```

#### 3.2.3 Dropout
```python
- 多层渐进式Dropout：[0.25, 0.35, 0.25, 0.15]
- 在激活函数之后应用
- 作用：随机失活神经元，防止过拟合
```

#### 3.2.4 批量归一化 (BatchNorm)
```python
- 每层激活前应用BatchNorm
- 作用：加速收敛，减少内部协变量偏移
```

#### 3.2.5 梯度裁剪
```python
- max_norm = 1.0
- 作用：防止梯度爆炸
```

#### 3.2.6 学习率调度
```python
- ReduceLROnPlateau策略
- factor = 0.5（验证损失无改善时学习率减半）
- patience = 10
- min_lr = 1e-7
```

#### 3.2.7 早停机制
```python
- patience = 20
- min_delta = 5e-5
- 保存最佳模型权重
```

#### 3.2.8 K折交叉验证
```python
- k = 5折
- 确保模型评估的稳健性
- 选择平均性能最佳的模型
```

### 3.3 超参数网格搜索框架
虽然完整网格搜索需要较长时间，但代码提供了完整框架：
- 隐藏层维度组合
- Dropout率组合
- 学习率搜索空间
- 权重衰减搜索空间
- 批次大小搜索空间

### 3.4 输出文件
- `advanced_regularized_model.pth` - 最佳模型文件
- `training_curves_fold_X.png` - 各折训练曲线

---

## 4. 模型性能总结

### 4.1 最佳模型性能（Medium Network）
| 指标 | 数值 |
|------|------|
| MSE | 4.6578 |
| RMSE | 2.1582 |
| MAE | 1.1107 |
| R² | 0.6705 |

### 4.2 模型对比
| 模型 | R² Score | 评价 |
|------|----------|------|
| Small Network | 0.5409 | 表现中等 |
| Medium Network | 0.6705 | ⭐ 最佳 |
| Large Network | 0.2413 | 过拟合 |

---

## 5. 项目文件说明

| 文件名 | 描述 |
|--------|------|
| `feature_analysis_demo.py` | 特征重要性分析与多模型对比 |
| `advanced_nn_regularization.py` | 高级正则化与K折交叉验证 |
| `neural_network.py` | 基础神经网络实现 |
| `nn_with_early_stopping.py` | 带早停的神经网络 |
| `1_feature_importance_analysis.py` | 特征重要性分析基础版 |
| `2_top10_features_prediction.py` | Top 10特征预测 |
| `feature_importance_nn.py` | 特征重要性结合神经网络 |
| `*.pth` | PyTorch模型权重文件 |
| `*.png` | 分析可视化图表 |

---

## 6. 使用说明

### 快速开始
```bash
# 运行特征重要性分析（推荐）
python feature_analysis_demo.py

# 运行高级正则化版本
python advanced_nn_regularization.py
```

### 依赖环境
```
- Python 3.7+
- PyTorch
- scikit-learn
- pandas
- numpy
- matplotlib (可选，用于可视化)
```

---

## 7. 结论与建议

### 7.1 教育启示
1. **提高出勤率**是提升学生成绩的最有效途径
2. **保证充足学习时间**是第二重要因素
3. 关注学生**身心健康**（睡眠、体育活动）
4. 加强**家校合作**（家长参与）

### 7.2 模型改进建议
1. 尝试集成学习（XGBoost、LightGBM）
2. 进行更精细的超参数调优
3. 收集更多时间序列数据
4. 尝试多任务学习（同时预测多个科目）
