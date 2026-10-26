# Car Price Prediction: Decision Tree, Ensembles and Neural Network

## Goal
Predict the **price of a car** from its characteristics (engine size, weight, fuel type, etc.) using the `CarPrice.csv` dataset (205 cars, 24 features), and compare several models.

## How to run
```bash
pip install scikit-learn pandas matplotlib
python main.py
```
Output: a results table in the terminal, plus two images: `decision_tree.png` and `comparison.png`.

## What the code does
1. **Data preparation**
   - Remove `car_ID` and `CarName` (not useful for prediction).
   - Text columns (fuel type, body type, ...) are turned into 0/1 columns (one-hot encoding), because models need numbers.
   - The price is expressed in thousands of dollars (helps the neural network train).
   - Split: 80% training, 20% testing. Models are evaluated only on cars they have never seen.

2. **Decision Tree** (`decision_tree.png`)
   - A tree asks a sequence of yes/no questions (e.g. *engine size <= 182?*) and each leaf gives a price (the average price of the training cars that reach it).
   - Depth is limited to 3 so the plot stays readable. In the plot, darker color = higher price; `samples` = how many cars reached that node; `value` = predicted price.
   - Weakness: a single tree is unstable and can overfit (small changes in data produce a very different tree).

3. **Ensemble methods** (combine many trees; this is the idea from the course "Ensemble methods")
   - **Bagging**: train 100 trees, each on a random sample (with replacement) of the training data, then **average** their predictions. Reduces variance.
   - **Random Forest**: like bagging, but each split also looks at only a random subset of the features, so the trees are less similar to each other. Averaging different trees works better.
   - **Boosting** (Gradient Boosting): trees are built **one after another**; each new small tree tries to correct the errors of the previous ones. Reduces bias.

4. **Neural Network (black box)**
   - A multi-layer perceptron (2 hidden layers: 32 and 16 neurons) with standardized inputs.
   - It can predict well, but it is a "black box": we cannot easily read *why* it predicts a price, unlike the decision tree, which can be plotted and read.

5. **Evaluation**
   - **R²**: how much of the price variation the model explains (1.0 = perfect, higher is better).
   - **RMSE**: typical prediction error in thousands of dollars (lower is better).

## Results (test set)
| Model          | R²    | RMSE (k$) |
|----------------|-------|-----------|
| Decision Tree  | 0.853 | 3.40      |
| Bagging        | 0.958 | 1.82      |
| Random Forest  | 0.958 | 1.83      |
| Boosting       | 0.927 | 2.40      |
| Neural Network | 0.943 | 2.12      |

(Exact numbers may vary slightly between library versions.)

## Conclusions
- Every ensemble clearly beats the single decision tree: combining many trees reduces errors.
- Bagging and Random Forest are best here; Boosting is a bit lower on this small dataset (default settings, no tuning).
- The neural network is competitive but not interpretable; the decision tree is less accurate but fully explainable. Accuracy vs. interpretability is the main trade-off.
