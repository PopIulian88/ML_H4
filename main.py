import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.ensemble import BaggingRegressor, RandomForestRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error

# 1. Load data: predict car price from the other columns
df = pd.read_csv("CarPrice.csv").drop(columns=["car_ID", "CarName"])
X = pd.get_dummies(df.drop(columns="price"), dtype=float)  # text columns -> 0/1 columns
y = df["price"] / 1000  # price in thousands of $ (helps the neural network converge)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 2. Single decision tree (small depth so the plot stays readable)
tree = DecisionTreeRegressor(max_depth=3, random_state=42).fit(X_train, y_train)
plt.figure(figsize=(20, 8))
plot_tree(tree, feature_names=X.columns, filled=True, rounded=True, fontsize=9)
plt.savefig("decision_tree.png", dpi=150, bbox_inches="tight")

# 3. Models to compare
models = {
    "Decision Tree": tree,
    "Bagging": BaggingRegressor(DecisionTreeRegressor(), n_estimators=100, random_state=42),
    "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
    "Boosting": GradientBoostingRegressor(n_estimators=100, random_state=42),
    "Neural Network": make_pipeline(
        StandardScaler(), MLPRegressor(hidden_layer_sizes=(32, 16), max_iter=3000, random_state=42)
    ),
}

# 4. Train and evaluate
results = []
for name, model in models.items():
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    results.append((name, r2_score(y_test, pred), np.sqrt(mean_squared_error(y_test, pred))))

results = pd.DataFrame(results, columns=["Model", "R2", "RMSE"]).round(3)
print(results.to_string(index=False))

# 5. Comparison plot
results.plot.bar(x="Model", y="R2", legend=False, rot=20, title="R2 on test set (higher = better)")
plt.tight_layout()
plt.savefig("comparison.png", dpi=150)
