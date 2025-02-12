# Lung Cancer Prediction

## Overview
This project aims to predict lung cancer using machine learning techniques. It utilizes a dataset from Kaggle that contains various attributes related to lung cancer and applies data preprocessing, feature engineering, and classification techniques to build an accurate predictive model.

## Dataset
The dataset used in this project is sourced from Kaggle:
- **Dataset Name:** `nancyalaswad90/lung-cancer`
- **File Used:** `survey lung cancer.csv`
- **Target Variable:** `LUNG_CANCER`
- **Binary Encoding:** YES = 1, NO = 0
- **Gender Encoding:** Male = 1, Female = 0

## Installation
### Required Libraries
Ensure you have the following Python libraries installed before running the notebook:
```bash
pip install pandas numpy matplotlib seaborn sklearn imbalanced-learn dtreeviz kagglehub
```

### Importing Dataset
```python
import kagglehub
nancyalaswad90_lung_cancer_path = kagglehub.dataset_download('nancyalaswad90/lung-cancer')
print('Data source import complete.')
```

## Data Preprocessing
1. **Handling Duplicates**: Removed duplicate entries.
2. **Handling Missing Values**: Checked for and handled any missing values.
3. **Feature Encoding**:
   - Converted categorical variables to numerical using `LabelEncoder`.
   - YES/NO attributes were mapped to 1/0.
4. **Feature Selection**: Dropped less relevant features (`GENDER`, `AGE`, `SMOKING`, `SHORTNESS OF BREATH`).
5. **Feature Engineering**: Created a new feature `ANXYELFIN` (ANXIETY × YELLOW_FINGERS).
6. **Target Imbalance Handling**: Used **ADASYN** oversampling technique to balance the target variable distribution.

## Data Visualization
- Used `Seaborn` and `Matplotlib` for correlation heatmaps and feature-target relationships.
- Heatmap analysis showed strong correlation among certain features.

## Model Implementation
### Decision Tree Classifier
1. **Model Training**:
   ```python
   from sklearn.tree import DecisionTreeClassifier
   dt_model = DecisionTreeClassifier(criterion='entropy', random_state=0)
   dt_model.fit(X_train, y_train)
   ```
2. **Visualization**:
   - Decision tree plot using `plot_tree`.
   - Text-based tree representation using `export_text`.
   - Interactive tree visualization using `dtreeviz`.
3. **Evaluation**:
   - Model achieved **94% accuracy**.
   - Performance metrics calculated using `classification_report` and `accuracy_score`.

## Results
- **Target variable was imbalanced, requiring oversampling.**
- **Decision Tree achieved high accuracy (94%).**
- **Feature selection improved model interpretability.**

## Future Improvements
- Experiment with other classifiers like Random Forest or XGBoost.
- Perform hyperparameter tuning for better optimization.
- Deploy the model using Flask or Streamlit.

## Author
- **Kaggle Dataset:** [nancyalaswad90/lung-cancer](https://www.kaggle.com/datasets/nancyalaswad90/lung-cancer)

## License
This project is for educational purposes and follows open-source guidelines.

---
Feel free to contribute or improve upon this project!

