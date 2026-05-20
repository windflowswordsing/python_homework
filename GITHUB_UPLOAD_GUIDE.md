# GitHub Upload Guide
==================

## 方法一：使用 Git 命令行（推荐）

### 1. 在 GitHub 上创建新仓库
1. 访问 https://github.com/new
2. 填写仓库名称（例如：student-score-prediction）
3. 选择 Public 或 Private
4. **不要**勾选 "Initialize this repository with a README"
5. 点击 "Create repository"

### 2. 初始化 Git 仓库并上传
在当前工作目录执行以下命令：

```bash
cd /workspace

# 初始化 Git 仓库
git init

# 配置用户信息（如果是首次使用）
git config user.name "Your Name"
git config user.email "your.email@example.com"

# 添加文件
git add README.md
git add PROJECT_ANALYSIS.md
git add feature_analysis_demo.py
git add advanced_nn_regularization.py
git add neural_network.py
git add nn_with_early_stopping.py
git add 1_feature_importance_analysis.py
git add 2_top10_features_prediction.py
git add feature_importance_nn.py
git add .gitignore

# 如果想上传图片和模型文件，取消下面的注释
# git add *.png
# git add *.pth
# git add *.npy

# 提交
git commit -m "Initial commit: Student exam score prediction project"

# 添加远程仓库（替换为您的仓库地址）
git remote add origin https://github.com/your-username/student-score-prediction.git

# 推送到 GitHub
git branch -M main
git push -u origin main
```

### 3. 如果使用 SSH 密钥
```bash
# 使用 SSH URL 而不是 HTTPS
git remote add origin git@github.com:your-username/student-score-prediction.git
```

---

## 方法二：使用 GitHub 网页上传

1. 访问 https://github.com/new 创建仓库
2. 在仓库页面中，点击 "uploading an existing file"
3. 拖拽或选择要上传的文件
4. 填写提交信息
5. 点击 "Commit changes"

---

## 推荐的上传文件清单

### 必要文件
- README.md
- PROJECT_ANALYSIS.md
- feature_analysis_demo.py
- advanced_nn_regularization.py
- neural_network.py
- nn_with_early_stopping.py
- 1_feature_importance_analysis.py
- 2_top10_features_prediction.py
- feature_importance_nn.py
- .gitignore

### 可选文件（较大文件，谨慎上传）
- *.png（可视化图片）
- *.pth（模型文件）
- *.npy（Top 10特征数据）

---

## 更新仓库
后续修改后，可以使用以下命令更新：

```bash
git add <modified-files>
git commit -m "Describe your changes"
git push
```

---

## 获取帮助
如果遇到问题，请查看 GitHub 官方文档：https://docs.github.com
