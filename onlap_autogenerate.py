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

# +
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from geopandas import GeoDataFrame
from scipy.spatial.distance import pdist, squareform
from shapely.geometry import Point
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scikitplot as skplt


# %matplotlib inline

# +
no_of_neighbors = 20

dataset = pd.read_csv(
    r"F:/Geology/WSGS/Projects/jupyter/0"
    + str(no_of_neighbors)
    + "neighbors.csv",
    index_col=[0],
)


# next let's split our toy data into training and test sets, choose how much with test_size of the data becomes the test set
X_train, X_test, y_train, y_test = train_test_split(
    dataset.iloc[0:, 0:-1].values,
    dataset.iloc[0:, -1].values,
    test_size=0.1,
    random_state=86,
)
tops_api = pd.read_csv(
    r"F:\Geology\WSGS\Projects\Unconformity or onlap\Python\ftunion.csv"
).fillna(
    0
)  # this file is available in the unconformity or onlap folder in the repo

iterable = ["Kfh", "Klz", "Kll", "Klr", "Kl", "Tfc", "Tfb", "Tfob", "Tfu"]
topcombos = list(zip(iterable, iterable[1:]))
topcombos.append(("Kfh", "Kl"))
topcombos.append(("Kl", "Tfu"))
# -

neigh = KNeighborsClassifier(n_neighbors=10)
neigh.fit(X_train, y_train)

# +

grid_params = {
    "n_neighbors": [1, 4, 5, 11, 19, 25],
    "weights": ["uniform", "distance"],
    "metric": ["euclidean", "manhattan"],
}

gs = GridSearchCV(
    KNeighborsClassifier(), grid_params, verbose=1, cv=2, n_jobs=5
)
gs_results = gs.fit(X_train, y_train)
# -

gs_results.best_score_

gs_results.best_estimator_

gs_results.best_params_

neigh = KNeighborsClassifier(
    algorithm="auto",
    leaf_size=30,
    metric="euclidean",
    metric_params=None,
    n_jobs=None,
    n_neighbors=4,
    p=2,
    weights="distance",
)
neigh.fit(X_train, y_train)

y_pred = neigh.predict(X_test)


skplt.metrics.plot_confusion_matrix(y_test, y_pred, normalize=True)
# plt.savefig('confusion matrix figure.pdf')

# +
# run this for all combinations of 2 tops and KNN


for j in enumerate(topcombos):
 
    fmtops = list(topcombos[j[0]])
    fmtops.extend(["x", "y"])
    tops = tops_api[fmtops]

    # calculate thicknesses and neighbors for the two tops
    hood = squareform(pdist(tops.iloc[:, -2:]))
    neighbors = []
    for i in enumerate(hood.argsort()[0:, 1: no_of_neighbors + 1]):
        selected = (
            tops.iloc[hood.argsort()[i[0], 1: no_of_neighbors + 1], 0:-2]
            .stack()
            .to_frame()
            .T
        )
        selected.columns = selected.columns.droplevel()
        neighbors.append(selected)
    frame = pd.concat(neighbors, sort=False)
    frame.index = range(len(frame))
    neighborhood = pd.concat([tops.iloc[:, :-2], frame], axis=1)
    thicknesses = neighborhood.diff(axis=1) * -1
    thicknesses[thicknesses < 0] = 0
    thicknesses.drop(columns=tops.columns[0], inplace=True)
    thicknesses[thicknesses < 0] = 0
    thicknesses[thicknesses > 3000] = 0
    locations = tops[["x", "y"]]
    real_world_log = thicknesses.apply(
        np.log
    )  # take the log of thicknesses for feature engineering
    real_world_pow = thicknesses.apply(
        lambda x: x ** 10
    )  # calculates the power values of thickness for another feature
    rw = (
        pd.concat(
            [thicknesses, real_world_log, real_world_pow, locations],
            axis=1,
            join_axes=[thicknesses.index],
        )
        .dropna()
        .replace(-np.inf, 0)
    )
    normalized_rw = (rw - rw.min()) / (rw.max() - rw.min()).replace(
        0, 0.00001
    )  # normalize the data from 0 to 1
    real_data = normalized_rw.values

    # load up the well location data and merge it with the tops data
    well_locs = pd.read_csv(
        r"F:\Geology\WSGS\Projects\Unconformity or onlap\Python\well_locations.csv",
        encoding="ISO-8859-1",
    )
    well_preds = neigh.predict(real_data)  # knn predictions
    well_prob = neigh.predict_proba(real_data)  # knn predictions
    probs = []

    for i in enumerate(well_prob):
        probs.append(i[1].max())
    tops_api["predictionknn"] = well_preds
    tops_api["probability"] = probs
    merged = pd.merge(tops_api, well_locs, on="API")
    plt.scatter(
        merged[merged["predictionknn"] == 0].LON,
        merged[merged["predictionknn"] == 0].LAT,
        label="Angular Unconformity",
    )
    plt.scatter(
        merged[merged["predictionknn"] == 1].LON,
        merged[merged["predictionknn"] == 1].LAT,
        label="Onlap",
    )
    plt.scatter(
        merged[merged["predictionknn"] == 2].LON,
        merged[merged["predictionknn"] == 2].LAT,
        label="Horizontally Stratified",
    )
    plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))
    plt.title("KNN Predictions")
    plt.savefig(r"new " + str(topcombos[j[0]]) + "_KNN2020.jpg")
    plt.clf()

    # writes the point data to a shapefile in the dir called data.shp
    geometry = [Point(xy) for xy in zip(merged.LON, merged.LAT)]
    crs = {"init": "epsg:3732"}
    geo_df = GeoDataFrame(merged, crs={"init": "epsg:4326"}, geometry=geometry)
    #geo_df.to_file(
    #    driver="ESRI Shapefile",
    #    filename="F:/Geology/WSGS/Projects/Unconformity or onlap/predictions/shapefiles/"
    #    + str(topcombos[j[0]])
    #    + "_knn_predictions.shp",
    #)
