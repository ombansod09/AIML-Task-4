# AIML-Task-4
# 🚢 Titanic Survival Project 🚢

This project predicts who survived the Titanic. We started with raw data, cleaned it up, and then built three different models to see which one was the best at predicting survival.

## What We Did

* **Cleaned the Data** 🧼: Handled missing values and dropped columns we didn't need.
* **Created New Features** 👷: Made new features like `Title` (from "Name") and `FamilySize`. This was super important!
* **Found Patterns** 📊: Looked at graphs to see who was most likely to survive (spoiler: it helped to be female or in 1st class).
* **Built 3 Models** 🎯: We built, compared, and optimized three different models.

## Our Models: From Good to Best

We used three models, each one "smarter" than the last.

### 1. Logistic Regression (The Baseline 👍)
This was our simple, starting model. It's a good benchmark to see if we can do better.
* **Accuracy: 79.89%**

### 2. Random Forest (The Challenger 🌳)
This is a much more powerful model. It works by building hundreds of small "decision trees" and letting them vote. It's great at finding complex patterns.
* **Accuracy: 82.68%** (A nice jump!)

### 3. Tuned Random Forest (The Champion 🏆)
This is our **best model**. We took the Random Forest and "tuned" it (like tuning a car engine 🏎️) to find its absolute best settings.
* **Accuracy: 83.80%**

---

## What We Learned 💡

* **Features are everything!** Our new `Title` and `FamilySize` features were *super* important. The `feature_importance.png` plot shows that `Sex` and `Title_Mr` were the biggest clues for the model.
* **Tuning matters.** Our tuned model (83.8%) is clearly better than the default one (82.7%).
* **See the proof:** The `roc_curve_comparison.png` 📈 chart shows our Tuned RF (the red line) is the best at telling survivors from non-survivors.

## How to Run This Project

### Requirements
You'll need: `pandas`, `scikit-learn`, `matplotlib`, `seaborn`, `joblib`

### Option 1: Re-Train Everything (Slow 🐢)
Run the notebooks in order. This will create the `.joblib` model files.
1.  `log_reg.ipynb`
2.  `rand_forest.ipynb`
3.  `tuned_rand_forest.ipynb`

### Option 2: Just Make Plots (Fast 🚀)
If you already have the `.joblib` files, just run this script to generate all the comparison visuals.
```bash
python visualize_models.py
```bash
pip install pandas scikit-learn matplotlib seaborn joblib
