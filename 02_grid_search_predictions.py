# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.3.2
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# # Grid Search for optimal parameters
# This notebook conducts a grid search for the optimal KNN parameters along with the optimal number of adjacent wells in the training dataset.

# +
import glob
import warnings
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier

sns.set()


warnings.filterwarnings("ignore")
# %matplotlib inline
# -

# point this to the directory with the generated training data
# with the different number of adjacent wells
TRAINING_FILES = glob.glob(r"F:\Geology\WSGS\Projects\jupyter\*.csv")

# +
ACCURACY_MEASURED = []  # for the cross-validation accuracy
NUM_NEIGHBORS = []  # the number of adjacent wells
grid_params = {
    "n_neighbors": [5, 10, 20, 40, 80],
    "weights": ["uniform", "distance"],
    "metric": ["euclidean", "manhattan"],
    "algorithm": ["ball_tree", "kd_tree", "brute"],
    "leaf_size": [10, 30],
}

for file in TRAINING_FILES:
    print(f"reading {file[-15:]}")
    no_of_neighbors = int(file[33:-13])
    dataset = pd.read_csv(file, index_col=[0])
    data_subset0 = dataset.drop(["class"], axis=1)

    X_train, X_test, y_train, y_test = train_test_split(
        dataset.drop("class", axis=1), dataset["class"], test_size=0.2, random_state=86,
    )
    gs = GridSearchCV(KNeighborsClassifier(), grid_params, verbose=8, cv=5, n_jobs=7)
    gs_results = gs.fit(X_train, y_train)
    neigh = KNeighborsClassifier(**gs.best_params_)

    cved = cross_val_score(
        neigh,
        dataset.drop("class", axis=1),
        dataset["class"],
        cv=10,
        scoring="accuracy",
    )
    ACCURACY_MEASURED.append(cved)
    NUM_NEIGHBORS.append(file[-16:-13])
# -

MEAN_ACCURACY = []
for i in enumerate(ACCURACY_MEASURED):
    plt.plot(ACCURACY_MEASURED[i[0]], label=str(NUM_NEIGHBORS[i[0]]) + " Neighbors")
    MEAN_ACCURACY.append(ACCURACY_MEASURED[i[0]].mean().round(4))
    plt.legend()
plt.xlabel("Fold Number")
plt.ylabel("Accuracy")

plt.plot(NUM_NEIGHBORS, MEAN_ACCURACY)
plt.xlabel("number of adjacent wells")
plt.ylabel("mean cross-validated accuracy")
