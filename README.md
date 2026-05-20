# Student Exam Score Prediction Project
=============================

## 📖 Overview
This project uses machine learning and deep learning techniques to analyze factors influencing student exam performance and build predictive models.

## 🔍 Key Features
- Comprehensive feature importance analysis using Random Forest
- Multiple neural network architectures with advanced regularization
- Model performance comparison and visualization
- Advanced regularization techniques: L1/L2, Dropout, BatchNorm, etc.

## 📊 Analysis Highlights
### Top 10 Important Features
1.  **Attendance** (38.5%) - Most important
2.  **Hours_Studied** (24.7%)
3.  Previous_Scores
4.  Tutoring_Sessions
5.  Access_to_Resources
6.  Parental_Involvement
7.  Sleep_Hours
8.  Physical_Activity
9.  Family_Income
10. Peer_Influence

### Model Performance
The best model achieved:
- **R² Score**: 0.6705
- **RMSE**: 2.1582
- **MAE**: 1.1107

## 📁 Project Structure
```
├── feature_analysis_demo.py      # Feature importance analysis & visualization
├── advanced_nn_regularization.py   # Advanced regularization with K-fold CV
├── neural_network.py          # Basic neural network
├── nn_with_early_stopping.py  # Neural network with early stopping
├── 1_feature_importance_analysis.py    # Feature importance basics
├── 2_top10_features_prediction.py  # Prediction using Top 10 features
├── feature_importance_nn.py          # Feature importance + NN
├── PROJECT_ANALYSIS.md       # Detailed analysis in Chinese
├── README.md                 # This file
├── *.pth                     # Trained model files
└── *.png                     # Visualization charts
```

## 🚀 Quick Start
```bash
# Install dependencies
pip install torch pandas numpy scikit-learn matplotlib

# Run feature importance analysis (recommended)
python feature_analysis_demo.py

# Run advanced regularization version
python advanced_nn_regularization.py
```

## 📊 Visualizations Generated
- `feature_importance_detailed.png` - Detailed feature importance visualization
- `feature_comprehensive_analysis.png` - Comprehensive analysis charts
- `model_comparison.png` - Model performance comparison
- `training_curves_fold_X.png` - Training curves for K-fold CV

## 📈 Algorithms Used
- Random Forest (feature importance)
- PyTorch Neural Networks
- L1/L2 Regularization
- Dropout
- Batch Normalization
- K-Fold Cross Validation

## 📋 Requirements
- Python 3.7+
- PyTorch
- scikit-learn
- pandas
- numpy
- matplotlib (optional)

## 📄 License
MIT License

## 📞 Contact
For questions or collaboration, please contact project maintainer.
