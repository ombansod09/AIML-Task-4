# AIML-Task-3
# Titanic Survival Prediction: A Model Comparison

This project analyzes the Kaggle Titanic dataset to predict passenger survival. The primary goal is to demonstrate a complete machine learning workflow, from data cleaning and feature engineering to building, comparing, and optimizing three different classification models.

## Project Workflow

The project followed these key steps:

1.  **Data Cleaning:** Loaded the raw data, handled missing values by imputing `Embarked` and dropping the `Cabin` column (due to excessive missing data).
2.  **Feature Engineering:** This was a critical step. We created new features that were more predictive than the originals:
    * **`Title`**: Extracted from the `Name` column (e.g., "Mr", "Miss", "Mrs", "Master").
    * **`FamilySize`**: Combined `SibSp` and `Parch` into a single, more useful feature.
    * **`IsAlone`**: A binary feature based on `FamilySize`.
3.  **Exploratory Data Analysis (EDA):** Visualized the data to find patterns. This confirmed that `Sex`, `Pclass`, and our new `Title` feature were all strong predictors of survival.
4.  **Modeling & Evaluation:** Built and evaluated three models of increasing complexity to find the best performer.

## Model Deep Dive: Relation and Differences

This project's core is the comparison of three models. Here's how they relate and differ:

### 1. Model 1: Logistic Regression (The Baseline)

* **What it is:** A fundamental and highly interpretable classification algorithm. It's a *linear model*, meaning it tries to find a single straight "line" (or hyperplane) to separate the two classes (Survived vs. Did Not Survive).
* **Role in Project:** This was our **baseline model**. It gives us a benchmark accuracy to beat. If a more complex model can't outperform this, it's not worth the extra complexity.
* **Performance:** 79.89% Accuracy. A solid, but not spectacular, start.

### 2. Model 2: Random Forest (The Challenger)

* **What it is:** A powerful "ensemble" model. It's *non-linear*. Instead of one "line," it builds hundreds of individual "decision trees" and lets them "vote" on the outcome. This "wisdom of the crowd" approach makes it very effective.
* **The Difference (from Logistic Regression):**
    * **Linear vs. Non-linear:** Logistic Regression can only find simple, linear relationships. A Random Forest can find very complex, "branching" rules (e.g., "IF Sex is female AND Pclass is 3 AND FamilySize > 4, THEN...").
    * **Performance:** Because the Titanic problem is not a simple linear one, the Random Forest immediately performed better by capturing these complex interactions.
* **Performance:** 82.68% Accuracy. A significant improvement over our baseline.

### 3. Model 3: Tuned Random Forest (The Champion)

* **What it is:** This is **not a new type of model**. It is the *same* Random Forest algorithm from Model 2, but it has been **optimized** through hyperparameter tuning.
* **The Difference (from Default Random Forest):**
    * **Default vs. Tuned:** The default Random Forest uses "sensible guess" settings (e.g., `n_estimators=100`). A *tuned* model uses a process like `GridSearchCV` to systematically test thousands of combinations of settings to find the *absolute best* ones (e.g., `n_estimators=200`, `max_depth=10`, etc.).
    * **Optimization:** This tuning process makes the model better at finding patterns without "overfitting" (memorizing the training data).
* **Performance:** **83.80% Accuracy.** Our best-performing and most robust model.

## Model Comparison Summary

| Model | Accuracy | Type | Key Characteristic |
| :--- | :--- | :--- | :--- |
| **Logistic Regression** | 79.89% | Linear | Simple, interpretable baseline. |
| **Random Forest (Default)** | 82.68% | Ensemble (Non-linear) | Powerful; captures complex patterns. |
| **Random Forest (Tuned)** | **83.80%** | Optimized Ensemble | Best performance after fine-tuning. |

## Key Findings from Visuals

The model comparison visuals (saved as `.png` files) confirm our findings:

* **`roc_curve_comparison.png`**: This plot shows the **Tuned Random Forest** has the highest **AUC (Area Under the Curve) score (0.880)**. This proves it is the best model at discriminating between passengers who would survive and those who would not.
* **`feature_importance.png`**: This plot, generated from our best model, shows *why* it's so effective. It confirms our EDA hypotheses:
    * **`Sex`** and **`Title_Mr`** were by far the most important predictors.
    * Our engineered features, **`FamilySize`** and **`Title_Miss`**, were also highly important—even more so than `Age` or `Fare`. This validates our feature engineering efforts.

## How to Run This Project

This project is set up to be modular. You can either re-train the models or (more quickly) just load the pre-trained models to generate the visuals.

### Requirements

Make sure you have the required libraries:
```bash
pip install pandas scikit-learn matplotlib seaborn joblib
