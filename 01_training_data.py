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

# # This is for creating the training data with different numbers of adjacent wells
# It will write 22 `csv` files to the directory for the varying numbers of adjacent wells. Similar to `01_training_data_generation.ipynb` but with more adjacent wells in the feature set.
#
# These datasets can be downloaded from https://osf.io/a6cwh/ inside the `Training Datasets` folder

from scipy.spatial.distance import pdist, squareform
import pandas as pd
import numpy as np


# +
def flatten(container):
    "Flattens lists"
    for i in container:
        if isinstance(i, (list, tuple)):
            for j in flatten(i):
                yield j
        else:
            yield i


NAMES = ["one", "two", "three"]  # this creates dummy NAMES for the formations
NUMBER_OF_LAYERS = 2  # this is the number of tops you want in your training data
SMALLEST = -6
LARGEST = 12
STEP = 2
# how many adjacent wells to use for feature engineering
NEIGHBORS_TO_TEST = [
    0,
    1,
    2,
    3,
    4,
    5,
    10,
    15,
    20,
    25,
    35,
    40,
    50,
    60,
    75,
    85,
    95,
    125,
    150,
    200,
    300,
    399,
]
for i in NEIGHBORS_TO_TEST:

    no_of_neighbors = i

    np.random.seed(19)
    df = pd.DataFrame()
    locations = pd.DataFrame()
    elevation_random = sorted(
        np.random.uniform(SMALLEST, LARGEST, NUMBER_OF_LAYERS - 1)
    )
    print("Getting the features")
    initial = ["thickness", "thickness natural log", "thickness power"]
    features = []
    for item in initial:
        features.append(item)
        for i in range(1, no_of_neighbors + 1):
            features.append(item + " neighbor " + str(i))
    features.append(["x location", "y location", "class"])
    flat_features = list(flatten(features))

    print(f"STARTING with {no_of_neighbors}")
    print("starting with truncation")
    for j in np.arange(SMALLEST, LARGEST, STEP):
        rolling = pd.DataFrame()
        j = np.round(j, decimals=3) + 0.5
        elevation_random = sorted(
            np.random.uniform(SMALLEST, LARGEST, NUMBER_OF_LAYERS - 1)
        )
        for i in range(len(NAMES[0 : NUMBER_OF_LAYERS - 1])):

            basement = (
                0.001
                + (10) * np.sin(1 - np.arange(0, 40, 0.1) / (j * 2) + 0.001)
                + np.random.rand(400) / 5
            )
            elevation = (
                np.full(
                    400,
                    basement.max()
                    + np.random.uniform(basement.min() / 2, basement.max() / 64, 1),
                )
                + np.random.rand(400) / 5
            )
            topbasement = np.where(basement > elevation, elevation, basement)

            rolling["zero"] = topbasement
            layer_elevation = (
                0.001
                + (10) * np.sin(1 - np.arange(0, 40, 0.1) / (j * 2) + 0.001)
                + abs(elevation_random[i])
                + np.random.rand(400) / 5
            )
            layer_elevation = np.where(
                layer_elevation < basement, basement, layer_elevation
            )
            layer_elevation = np.where(
                layer_elevation > elevation, elevation, layer_elevation
            )
            rolling[NAMES[i]] = layer_elevation
        x = np.arange(0, 40, 0.1)
        y = np.random.randint(0, 10, len(x))
        if j % 0.2 > 0.1:
            rolling["ex"] = x * np.cos(-j / 2) - y * np.sin(-j / 2)
            rolling["ey"] = y * np.cos(-j / 2) - x * np.sin(-j / 2)
        else:
            rolling["ex"] = x * np.cos(j / 2) - y * np.sin(j / 2)
            rolling["ey"] = y * np.cos(j / 2) - x * np.sin(j / 2)
        for k in range(100):
            rolling.iloc[
                np.random.randint(0, 399), np.random.randint(0, NUMBER_OF_LAYERS - 1),
            ] = 0
        hood = squareform(pdist(rolling.iloc[:, -2:]))
        neighbors = []
        for i in enumerate(hood.argsort()[0:, 1 : no_of_neighbors + 1]):
            selected = (
                rolling.iloc[hood.argsort()[i[0], 1 : no_of_neighbors + 1], 0:-2]
                .stack()
                .to_frame()
                .T
            )
            selected.columns = selected.columns.droplevel()
            neighbors.append(selected)
        frame = pd.concat(neighbors, sort=False)
        frame.index = range(len(frame))
        neighborhood = pd.concat([rolling.iloc[:, :-2], frame], axis=1)
        thicknesses = neighborhood.diff(axis=1)
        thicknesses[thicknesses < 0] = 0
        thicknesses.drop(columns="zero", inplace=True)
        locations = pd.concat((locations, rolling.iloc[:, -2:]))
        df = pd.concat((df, thicknesses))
    logged = df.apply(np.log)  # take the log of thicknesses for feature engineering
    powered = df.apply(
        lambda x: x ** 10
    )  # calculates the power values of thickness for another feature
    at = (
        pd.concat([df, logged, powered, locations], axis=1)
        .reindex(df.index)
        .dropna()
        .replace(-np.inf, 0)
    )
    print("normalizing the truncation")
    # NORMALIZING THE DATA
    # normalize the data from 0 to 1
    normalized_dfa = (at - at.min()) / (at.max() - at.min()).replace(0, 0.00001)
    normalized_locations = (locations - locations.min()) / (
        locations.max() - locations.min()
    )
    x = normalized_locations.ex.values
    y = normalized_locations.ey.values
    normalized_dfa["ex"] = x
    normalized_dfa["ey"] = y

    np.random.seed(19)
    print("starting with onlap")
    df_onlap = pd.DataFrame()
    locations = pd.DataFrame()
    for j in np.arange(SMALLEST, LARGEST, STEP):
        rolling = pd.DataFrame()
        j = np.round(j, decimals=3) + 0.5
        elevation_random = sorted(
            np.random.uniform(SMALLEST, LARGEST, NUMBER_OF_LAYERS - 1)
        )
        for i in range(len(NAMES[0 : NUMBER_OF_LAYERS - 1])):
            basement = (
                0.001
                + (10) * np.sin(1 - np.arange(0, 40, 0.1) / (j * 2) + 0.001)
                + np.random.rand(400) / 5
            )
            elevation = (
                np.full(
                    400,
                    basement.max()
                    + np.random.uniform(basement.min() / 2, basement.max() / 64, 1),
                )
                + np.random.rand(400) / 5
            )
            topbasement = np.where(basement > elevation, elevation, basement)
            rolling["zero"] = topbasement
            strat_elevation = (
                np.full(400, elevation_random[i]) + np.random.rand(400) / 5
            )
            onlap = np.where(strat_elevation > basement, strat_elevation, basement)
            layer_elevation = np.where(
                layer_elevation < basement, basement, layer_elevation
            )
            layer_elevation = np.where(onlap > elevation, elevation, onlap)
            rolling[NAMES[i]] = layer_elevation
        x = np.arange(0, 40, 0.1)
        y = np.random.randint(0, 10, len(x))
        if j % 0.2 > 0.1:
            rolling["ex"] = x * np.cos(-j / 2) - y * np.sin(-j / 2)
            rolling["ey"] = y * np.cos(-j / 2) - x * np.sin(-j / 2)
        else:
            rolling["ex"] = x * np.cos(j / 2) - y * np.sin(j / 2)
            rolling["ey"] = y * np.cos(j / 2) - x * np.sin(j / 2)
        for k in range(100):
            rolling.iloc[
                np.random.randint(0, 399), np.random.randint(0, NUMBER_OF_LAYERS - 1),
            ] = 0
        hood = squareform(pdist(rolling.iloc[:, -2:]))
        neighbors = []
        for i in enumerate(hood.argsort()[0:, 1 : no_of_neighbors + 1]):
            selected = (
                rolling.iloc[hood.argsort()[i[0], 1 : no_of_neighbors + 1], 0:-2]
                .stack()
                .to_frame()
                .T
            )
            selected.columns = selected.columns.droplevel()
            neighbors.append(selected)
        frame = pd.concat(neighbors, sort=False)
        frame.index = range(len(frame))
        neighborhood = pd.concat([rolling.iloc[:, :-2], frame], axis=1)
        thicknesses = neighborhood.diff(axis=1)
        thicknesses[thicknesses < 0] = 0

        thicknesses.drop(columns="zero", inplace=True)
        locations = pd.concat((locations, rolling.iloc[:, -2:]))
        df_onlap = pd.concat((df_onlap, thicknesses))
    onlaplogged = df_onlap.apply(
        np.log
    )  # take the log of thicknesses for feature engineering
    onlappowered = df_onlap.apply(
        lambda x: x ** 10
    )  # calculates the power values of thickness for another feature
    ot = (
        pd.concat(
            [df_onlap, onlaplogged, onlappowered, locations],
            axis=1,

        )
        .reindex(df_onlap.index)
        .dropna()
        .replace(-np.inf, 0)
    )
    print("normalizing the onlap")
    # NORMALIZING THE DATA
    # normalize the data from 0 to 1
    normalized_dfo = (ot - ot.min()) / (ot.max() - ot.min()).replace(0, 0.00001)
    normalized_locations = (locations - locations.min()) / (
        locations.max() - locations.min()
    )
    x = normalized_locations.ex.values
    y = normalized_locations.ey.values
    normalized_dfo["ex"] = x
    normalized_dfo["ey"] = y

    np.random.seed(19)
    print("starting the horizontal")
    df_horizontal = pd.DataFrame()
    locations = pd.DataFrame()
    for j in np.arange(SMALLEST, LARGEST, STEP):
        rolling = pd.DataFrame()
        j = j + 0.5
        elevation_random = sorted(
            np.random.uniform(SMALLEST, LARGEST, NUMBER_OF_LAYERS - 1)
        )
        for i in range(len(NAMES[0 : NUMBER_OF_LAYERS - 1])):
            strat_elevation = (
                np.full(400, elevation_random[i]) + np.random.rand(400) / 5
            )
            basement = strat_elevation - abs(
                np.random.uniform(SMALLEST, LARGEST, NUMBER_OF_LAYERS - 1)
                + np.random.rand(400) / 5
            )
            elevation = (
                np.full(400, strat_elevation + elevation_random)
                + np.random.rand(400) / 5
            )
            topbasement = np.where(basement > elevation, elevation, basement)
            layer_elevation = np.where(
                strat_elevation > elevation, elevation, strat_elevation
            )
            rolling["zero"] = topbasement
            rolling[NAMES[i]] = layer_elevation
        x = np.arange(0, 40, 0.1)
        y = np.random.randint(0, 10, len(x))
        if j % 0.2 > 0.1:
            rolling["ex"] = x * np.cos(-j / 2) - y * np.sin(-j / 2)
            rolling["ey"] = y * np.cos(-j / 2) - x * np.sin(-j / 2)
        else:
            rolling["ex"] = x * np.cos(j / 2) - y * np.sin(j / 2)
            rolling["ey"] = y * np.cos(j / 2) - x * np.sin(j / 2)
        for k in range(100):
            rolling.iloc[
                np.random.randint(0, 399), np.random.randint(0, NUMBER_OF_LAYERS - 1),
            ] = 0
        hood = squareform(pdist(rolling.iloc[:, -2:]))
        neighbors = []
        for i in enumerate(hood.argsort()[0:, 1 : no_of_neighbors + 1]):
            selected = (
                rolling.iloc[hood.argsort()[i[0], 1 : no_of_neighbors + 1], 0:-2]
                .stack()
                .to_frame()
                .T
            )
            selected.columns = selected.columns.droplevel()
            neighbors.append(selected)
        frame = pd.concat(neighbors, sort=False)
        frame.index = range(len(frame))
        neighborhood = pd.concat([rolling.iloc[:, :-2], frame], axis=1)
        thicknesses = neighborhood.diff(axis=1)
        thicknesses[thicknesses < 0] = 0
        thicknesses.drop(columns="zero", inplace=True)
        locations = pd.concat((locations, rolling.iloc[:, -2:]))
        df_horizontal = pd.concat((df_horizontal, thicknesses))
    horizlogged = df_horizontal.apply(
        np.log
    )  # take the log of thicknesses for feature engineering
    horizpowered = df_horizontal.apply(
        lambda x: x ** 10
    )  # calculates the power values of thickness for another feature
    hs = (
        pd.concat(
            [df_horizontal, horizlogged, horizpowered, locations],
            axis=1,
        )
        .reindex(df_horizontal.index)
        .dropna()
        .replace(-np.inf, 0)
    )
    print("normalizing the horizontal strata")
    # NORMALIZING THE DATA
    # normalize the data from 0 to 1
    normalized_dfh = (hs - hs.min()) / (hs.max() - hs.min()).replace(0, 0.00001)
    normalized_locations = (locations - locations.min()) / (
        locations.max() - locations.min()
    )
    x = normalized_locations.ex.values
    y = normalized_locations.ey.values
    normalized_dfh["ex"] = x
    normalized_dfh["ey"] = y

    # now assign classes to the datasets, 1 is onlap, 0 is angular unconformity
    normalized_dfa["class"] = 0  # truncation
    normalized_dfo["class"] = 1  # onlap
    normalized_dfh["class"] = 2  # horizontal

    dataset = pd.concat((normalized_dfa, normalized_dfo, normalized_dfh))
    dataset.columns = flat_features
    print(f"saving the training data for {no_of_neighbors}")
    dataset.to_csv(str(no_of_neighbors) + "neighbors.csv")
    print(f"Done with {no_of_neighbors} neighbors")
# -

