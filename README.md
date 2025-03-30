# Financial-Risk-Prediction-Using-Data-Engineering
Financial institutions face significant risks in lending, investments, and fraud. The ability to predict financial risks using advanced data engineering pipelines is critical. This hackathon challenges participants to design a scalable data pipeline that ingests, processes, and analyzes financial data to predict risk levels.

# Credit Card Fraud Detection

## Overview

This project aims to build a machine learning model that can detect fraudulent transactions in a credit card transaction dataset. The dataset used for this project contains real-world credit card transaction data, with a mix of fraudulent and non-fraudulent transactions. The goal is to train a model capable of accurately identifying fraudulent transactions and reducing the impact of fraud.

This project leverages various machine learning algorithms, data preprocessing techniques, and data visualization methods to achieve high accuracy in fraud detection.

---

## Table of Contents

1. [Dataset Overview](#dataset-overview)
2. [Dependencies](#dependencies)
3. [Project Steps](#project-steps)
4. [Results and Evaluation](#results-and-evaluation)
5. [Conclusion](#conclusion)

---

## Dataset Overview

The dataset used for this project is the **Credit Card Fraud Detection dataset** from Kaggle. It consists of 31 features:

- **Time**: The time of the transaction.
- **Amount**: The amount for the transaction.
- **V1 to V28**: Anonymized features that represent various characteristics of the transaction.
- **Class**: Target variable where `1` indicates a fraudulent transaction and `0` indicates a non-fraudulent transaction.

---

## Dependencies

To run this project, you'll need the following Python libraries:

- numpy
- pandas
- matplotlib
- seaborn
- scikit-learn
- tensorflow
- imbalanced-learn

You can install the required dependencies using the following command:

```bash
pip install -r requirements.txt
```

---

## Project Steps

### 1. **Data Loading and Inspection**
   - The dataset is loaded using `pandas` and initial insights like the dataset head, description, class distribution, and missing values are displayed.
   
### 2. **Data Preprocessing**
   - Missing values are checked and handled.
   - A class distribution analysis is performed to evaluate the imbalance between fraudulent and non-fraudulent transactions.
   - Feature correlation is visualized using a heatmap.
   - Principal Component Analysis (PCA) is used for dimensionality reduction and visualizing the data in 2D.

### 3. **Data Resampling**
   - Since the dataset is highly imbalanced (fraudulent transactions are much fewer than non-fraudulent ones), **SMOTE** (Synthetic Minority Over-sampling Technique) is used to balance the classes by generating synthetic data for the minority class (fraudulent transactions).

### 4. **Model Evaluation**
   - Various machine learning algorithms are evaluated, including:
     - Logistic Regression
     - K-Nearest Neighbors (KNN)
     - Decision Tree Classifier
     - Random Forest Classifier
   - The model is evaluated using accuracy, precision, recall, F1-score, and ROC AUC score.

### 5. **Confusion Matrix**
   - The final model's performance is assessed using a confusion matrix to show how well the model predicts both fraudulent and non-fraudulent transactions.

### 6. **Feature Importance**
   - The importance of each feature in predicting fraud is visualized using bar charts.

### 7. **Model Prediction**
   - The model is used to predict the fraud status of a sample transaction based on its features.

---

## Results and Evaluation

The evaluation metrics for the final model (Random Forest Classifier) include:

- **Accuracy**: 99.94%
- **Precision**: 0.81 for fraudulent transactions
- **Recall**: 0.83 for fraudulent transactions
- **F1-Score**: 0.82 for fraudulent transactions

The confusion matrix and classification report further validate that the model has performed well in distinguishing between fraudulent and non-fraudulent transactions.

---

## Conclusion

This project demonstrates the process of handling imbalanced datasets and training a machine learning model to detect credit card fraud. The final Random Forest model showed promising results, with high accuracy and reasonable recall for fraud detection. The use of SMOTE to balance the data and the feature importance analysis provides valuable insights into the model’s decision-making process.

Future work could involve trying more advanced models such as ensemble methods, deep learning, or fine-tuning hyperparameters for better accuracy and recall.

---

## How to Run

1. Clone this repository:
   ```bash
   git clone https://github.com/username/credit-card-fraud-detection.git
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the script:
   ```bash
   python fraud_detection.py
   ```

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
