# Machine Learning Assignment 2: Classification Models & Deployment

**Course:** BITS Pilani WILP - M.Tech (AIML)  
**Subject:** Machine Learning  
**Assignment:** Assignment - 2  

---

## a. Problem Statement

Direct telemarketing campaigns remain a vital channel for retail banking to promote term deposit subscriptions. However, cold-calling uninterested clients leads to unnecessary operational costs and client dissatisfaction, while missing prospective depositors causes lost revenue. 

The objective of this project is to develop, evaluate, and deploy an end-to-end Machine Learning classification pipeline to predict whether a client will subscribe to a bank term deposit (target variable `y`: `yes` or `no`). We implement five distinct classification algorithms, evaluate them across six key metrics under class imbalance, and deploy an interactive Streamlit web application on Streamlit Community Cloud for real-time inference on unseen test data without model retraining.

---

## b. Dataset Description

- **Dataset Name:** Bank Marketing Dataset
- **Source Repository:** UCI Machine Learning Repository (https://archive.ics.uci.edu/dataset/222/bank+marketing)
- **Number of Instances:** 4,521 samples (meets the criterion of >= 500 instances)
- **Number of Features:** 16 input features (meets the criterion of >= 12 features) + 1 target column (`y`)
- **Target Variable:** `y` (Binary: `yes` = 1, `no` = 0)
- **Target Distribution:** 4,000 "no" (~88.5%), 521 "yes" (~11.5%) - demonstrating realistic class imbalance.

### Key Input Features:
1. `age` (Numeric): Age of customer.
2. `job` (Categorical): Occupation (admin., technician, services, management, blue-collar, etc.).
3. `marital` (Categorical): Marital status (married, single, divorced).
4. `education` (Categorical): Education level (primary, secondary, tertiary, unknown).
5. `default` (Binary): Credit in default (yes, no).
6. `balance` (Numeric): Average yearly balance in euros.
7. `housing` (Binary): Housing loan (yes, no).
8. `loan` (Binary): Personal loan (yes, no).
9. `contact` (Categorical): Contact communication type (cellular, telephone, unknown).
10. `day` (Numeric): Last contact day of the month.
11. `month` (Categorical): Last contact month of year (jan to dec).
12. `duration` (Numeric): Last contact duration in seconds.
13. `campaign` (Numeric): Number of contacts performed during this campaign.
14. `pdays` (Numeric): Days passed after client was last contacted (-1 means not previously contacted).
15. `previous` (Numeric): Number of contacts performed before this campaign.
16. `poutcome` (Categorical): Outcome of previous marketing campaign (failure, other, success, unknown).

---

## c. Github Repository Link

- **GitHub Repository Link:** https://github.com/Saksham2026AJ05199/ml-assignment-bank-marketing
- **Live Streamlit App Link:** https://ml-assignment-bank-marketing-m4wmbhiumrdh6dx4vxl3zb.streamlit.app/

### Repository Structure:
project-folder/
|-- app.py                     # Streamlit interactive application
|-- requirements.txt           # Deployment dependency specifications
|-- README.md                  # Project documentation & benchmark report
|-- test_data.csv              # Unseen hold-out test dataset (features + target 'y')
|-- model/                     # Trained models & preprocessing pipelines
|   |-- logistic_regression.joblib
|   |-- decision_tree.joblib
|   |-- knn.joblib
|   |-- naive_bayes.joblib
|   |-- svm.joblib
|   |-- test.csv
|   |-- model_comparison.csv
|   |-- Assignment_ML2.ipynb         # Source code of training & evaluation

---

## d. Models Used

All five classifiers were trained on the same 80% stratified training split (N = 3,616) and evaluated on the exact same 20% holdout test set (N = 905).

### Comparison Table:

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** | 0.8232 | 0.8908 | 0.3716 | 0.7788 | 0.5031 | 0.4533 |
| **Decision Tree** | 0.7602 | 0.8194 | 0.3058 | 0.8558 | 0.4506 | 0.4121 |
| **KNN** | 0.8884 | 0.7997 | 0.5517 | 0.1538 | 0.2406 | 0.2492 |
| **Naive Bayes** | 0.8210 | 0.7881 | 0.3092 | 0.4519 | 0.3672 | 0.2737 |
| **Support Vector Machine** | **0.8453** | **0.9019** | **0.4126** | **0.8173** | **0.5484** | **0.5068** |

---

## Observations on Model Performance

| ML Model Name | Observation about model performance |
| :--- | :--- |
| **Logistic Regression** | Achieves a strong linear baseline with balanced weighting, reaching an AUC of **0.8908** and capturing **77.88%** of potential depositors (Recall). It demonstrates steady discriminative performance with an MCC of **0.4533**. |
| **Decision Tree** | Yields the highest Recall (**85.58%**) by aggressively isolating positive leaves. However, it incurs higher false positives, reducing Precision (**0.3058**) and overall Accuracy (**76.02%**). |
| **KNN** | Exhibits the classic pitfall of majority-voting on imbalanced data. While nominal Accuracy appears high (**88.84%**), Recall collapses to **15.38%** (missing ~85% of actual subscribers). Its low F1 (**0.2406**) and MCC (**0.2492**) reflect this severe majority-class bias. |
| **Naive Bayes** | Produces moderate results (AUC: **0.7881**, Recall: **45.19%**, MCC: **0.2737**). Its core limitation is the conditional feature independence assumption, which is violated by interrelated banking features such as call `duration`, `pdays`, and `campaign`. |
| **Support Vector Machine** | Delivers the best overall performance with the RBF kernel. It achieves the **highest AUC (0.9019)**, **highest F1-score (0.5484)**, and **highest MCC (0.5068)**, balancing high Recall (**81.73%**) with superior Precision (**0.4126**). |
| **Overall, Winner for your dataset?** | **Support Vector Machine (SVM) is the overall winner.** On an imbalanced dataset, Accuracy is misleading. SVM delivers the best possible combination: highest discriminative separation (AUC = 0.9019), highest harmonic mean of precision and recall (F1 = 0.5484), and highest correlation across both classes (MCC = 0.5068), successfully capturing over 81% of true subscribers while keeping false alarms controlled. |
