import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import RandomizedSearchCV, cross_val_score, train_test_split
from skopt import BayesSearchCV
from skopt.space import Integer

# TODO fix this monkey patch
np.int = int


# Custom classifier that extends RandomForestClassifier
class CustomRandomForestClassifier(RandomForestClassifier):
    def __init__(self, n_estimators=100, criterion="gini", max_depth=None,
                 min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0,
                 max_features="sqrt", max_leaf_nodes=None, min_impurity_decrease=0.0,
                 bootstrap=True, oob_score=False, n_jobs=None,
                 random_state=None, verbose=0, warm_start=False, class_weight=None,
                 ccp_alpha=0.0, max_samples=None):
        super().__init__(
            n_estimators=n_estimators, criterion=criterion, max_depth=max_depth,
            min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf,
            min_weight_fraction_leaf=min_weight_fraction_leaf, max_features=max_features,
            max_leaf_nodes=max_leaf_nodes, min_impurity_decrease=min_impurity_decrease,
            bootstrap=bootstrap, oob_score=oob_score,
            n_jobs=n_jobs, random_state=random_state, verbose=verbose, warm_start=warm_start,
            class_weight=class_weight, ccp_alpha=ccp_alpha, max_samples=max_samples,
        )

    def set_params(self, **params):
        param_keys = ["max_depth", "min_samples_leaf", "min_samples_split", "n_estimators"]
        for key in param_keys:
            if key in params and isinstance(params[key], np.ndarray):
                params[key] = int(params[key][0])
        return super().set_params(**params)


# Load the processed data
landmarks_with_coins = pd.read_csv("landmarks_with_coins.csv")

# Specify the number of OPS personality type label columns
num_ops_label_columns = 11  # Update this based on your actual data

# Features (all landmark columns)
X = landmarks_with_coins.iloc[:, num_ops_label_columns + 1:]

# Parameter grid for Randomized Search
param_distributions = {
    "n_estimators": Integer(100, 500),
    "max_depth": Integer(5, 50),
    "min_samples_split": Integer(2, 10),
    "min_samples_leaf": Integer(1, 5),
}

# Iterate over each OPS coin (label column)
for i in range(1, num_ops_label_columns + 1):
    column_name = landmarks_with_coins.columns[i]
    print(f"\nModel for {column_name}:")

    # Current label (OPS coin)
    y = landmarks_with_coins.iloc[:, i]

    # Filter out rows where the current coin value is NaN
    valid_data = landmarks_with_coins[~y.isna()]
    X_filtered = valid_data.iloc[:, num_ops_label_columns + 1:].values
    y_filtered = valid_data.iloc[:, i].values

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X_filtered, y_filtered, test_size=0.2, random_state=1,
                                                        stratify=y_filtered)

    # Apply SMOTE for dealing with class imbalance
    smote = SMOTE(random_state=1)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

    # Initialize the Randomized Search model
    random_search = RandomizedSearchCV(estimator=CustomRandomForestClassifier(random_state=42),
                                       param_distributions=param_distributions, n_iter=100, cv=3, n_jobs=-1, verbose=2,
                                       random_state=1)

    # Fit the random search to the data
    random_search.fit(X_train_resampled, y_train_resampled)
    best_random = random_search.best_estimator_

    # Perform cross-validation on the best model from Randomized Search
    cv_scores_random = cross_val_score(best_random, X_filtered, y_filtered, cv=3)
    print(f"Cross-validation scores for Randomized Search model: {cv_scores_random}")
    print(f"Mean CV score (Randomized Search): {np.mean(cv_scores_random)}")

    # Bayesian Optimization
    bayes_search = BayesSearchCV(estimator=CustomRandomForestClassifier(random_state=42),
                                 search_spaces=param_distributions,
                                 n_iter=32, cv=3, n_jobs=-1, verbose=2)
    bayes_search.fit(X_train_resampled, y_train_resampled)
    best_bayes = bayes_search.best_estimator_

    # Perform cross-validation on the best model from Bayesian Optimization
    cv_scores_bayes = cross_val_score(best_bayes, X_filtered, y_filtered, cv=3)
    print(f"Cross-validation scores for Bayesian Optimization model: {cv_scores_bayes}")
    print(f"Mean CV score (Bayesian Optimization): {np.mean(cv_scores_bayes)}")

    # Print the best parameters
    print("Best parameters (Randomized Search):", random_search.best_params_)
    print("Best parameters (Bayesian Optimization):", bayes_search.best_params_)

    # Predict using the best model from Randomized Search
    y_pred = best_random.predict(X_test)

    # Print out the classification report and accuracy
    print(classification_report(y_test, y_pred))
    print(f"Accuracy: {accuracy_score(y_test, y_pred)}")

    # Feature importance visualization
    result = permutation_importance(best_random, X_test, y_test, n_repeats=10, random_state=42, n_jobs=-1)
    forest_importances = pd.Series(result.importances_mean,
                                   index=landmarks_with_coins.columns[num_ops_label_columns + 1:])
    fig, ax = plt.subplots()
    forest_importances.plot.bar(ax=ax)
    ax.set_title("Feature importances")
    ax.set_ylabel("Mean accuracy decrease")
    fig.tight_layout()
    plt.show()
