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

# # Generating training datasets
# This notebook generates the training datasets with varying numbers of adjacent wells used as features. It writes out the training data to `.csv` files for use in training the models

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform


# %matplotlib inline

# +
def flatten(A):
    "Flatttens lists"
    rt = []
    for i in A:
        if isinstance(i, list):
            rt.extend(flatten(i))
        else:
            rt.append(i)
    return rt


names = [
    "one",
    "two",
    "three",
]
# this creates dummy names for the formations

number_of_layers = 2
# this is the number of tops you want in your training data

smallest = -5
largest = 12
step = 0.2

# this loop walks through and creates the training data with adjacent wells as features

for i in range(1, 300):
    # i is the number of adjacent wells
    no_of_neighbors = i

    np.random.seed(19)
    df = pd.DataFrame()
    locations = pd.DataFrame()
    elevation_random = sorted(
        np.random.uniform(smallest, largest, number_of_layers - 1)
    )

    print(f"STARTING with {no_of_neighbors}")
    # Creating the truncation dataset
    for j in np.arange(smallest, largest, step):
        rolling = pd.DataFrame()
        for i in range(len(names[0 : number_of_layers - 1])):
            basement = 0.001 + (10 / j) * np.sin(
                1 - np.arange(0, 40, 0.1) / (j * 10) + 0.001
            )
            elevation = np.full(400, j)
            topbasement = np.where(basement > elevation, elevation, basement)
            rolling["zero"] = topbasement
            layer_elevation = (
                0.001
                + (10 / j) * np.sin(1 - np.arange(0, 40, 0.1) / (j * 10) + 0.001)
                + elevation_random[i]
            )
            layer_elevation = np.where(
                layer_elevation > elevation, elevation, layer_elevation
            )
            rolling[names[i]] = layer_elevation
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
                np.random.randint(0, 399), np.random.randint(0, number_of_layers - 1),
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
        pd.concat([df, logged, powered, locations], axis=1, join_axes=[df.index])
        .dropna()
        .replace(-np.inf, 0)
    )
    print("Getting the features")
    features = [
        "thickness",
        "thickness neighbor 1",
        "thickness neighbor 2",
        "thickness neighbor 3",
        "thickness neighbor 4",
        "thickness neighbor 5",
        "thickness neighbor 6",
        "thickness neighbor 7",
        "thickness neighbor 8",
        "thickness neighbor 9",
        "thickness neighbor 10",
        "thickness neighbor 11",
        "thickness neighbor 12",
        "thickness neighbor 13",
        "thickness neighbor 14",
        "thickness neighbor 15",
        "thickness neighbor 16",
        "thickness neighbor 17",
        "thickness neighbor 18",
        "thickness neighbor 19",
        "thickness neighbor 20",
        "thickness natural log",
        "thickness natural log neighbor 1",
        "thickness natural log neighbor 2",
        "thickness natural log neighbor 3",
        "thickness natural log neighbor 4",
        "thickness natural log neighbor 5",
        "thickness natural log neighbor 6",
        "thickness natural log neighbor 7",
        "thickness natural log neighbor 8",
        "thickness natural log neighbor 9",
        "thickness natural log neighbor 10",
        "thickness natural log neighbor 11",
        "thickness natural log neighbor 12",
        "thickness natural log neighbor 13",
        "thickness natural log neighbor 14",
        "thickness natural log neighbor 15",
        "thickness natural log neighbor 16",
        "thickness natural log neighbor 17",
        "thickness natural log neighbor 18",
        "thickness natural log neighbor 19",
        "thickness natural log neighbor 20",
        "thickness power",
        "thickness power neighbor 1",
        "thickness power neighbor 2",
        "thickness power neighbor 3",
        "thickness power neighbor 4",
        "thickness power neighbor 5",
        "thickness power neighbor 6",
        "thickness power neighbor 7",
        "thickness power neighbor 8",
        "thickness power neighbor 9",
        "thickness power neighbor 10",
        "thickness power neighbor 11",
        "thickness power neighbor 12",
        "thickness power neighbor 13",
        "thickness power neighbor 14",
        "thickness power neighbor 15",
        "thickness power neighbor 16",
        "thickness power neighbor 17",
        "thickness power neighbor 18",
        "thickness power neighbor 19",
        "thickness power neighbor 20",
        "x location",
        "y location",
        "class",
    ]

    featured = [
        features[0 : no_of_neighbors + 1],
        features[21 : 22 + no_of_neighbors],
        features[42 : 43 + no_of_neighbors],
        features[-3:],
    ]
    flat_features = flatten(featured)

    print("normalizing the truncation")
    # NORMALIZING THE TRUNCATION DATA
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

    df_onlap = pd.DataFrame()
    locations = pd.DataFrame()

    # Creating the onlap data
    for j in np.arange(smallest, largest, step):
        rolling = pd.DataFrame()
        for i in range(len(names[0 : number_of_layers - 1])):
            basement = 0.001 + (10 / j) * np.sin(
                1 - np.arange(0, 40, 0.1) / (j * 10) + 0.001
            )
            elevation = np.full(400, j)
            topbasement = np.where(basement > elevation, elevation, basement)
            rolling["zero"] = topbasement
            strat_elevation = np.full(400, elevation_random[i])
            onlap = np.where(strat_elevation > basement, strat_elevation, basement)
            layer_elevation = np.where(onlap > elevation, elevation, onlap)
            rolling[names[i]] = layer_elevation
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
                np.random.randint(0, 399), np.random.randint(0, number_of_layers - 1),
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
            join_axes=[df_onlap.index],
        )
        .dropna()
        .replace(-np.inf, 0)
    )
    print("normalizing the onlap")
    # NORMALIZING THE ONLAP
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

    df_horizontal = pd.DataFrame()
    locations = pd.DataFrame()
    # Creating the horizontally stratified data
    for j in np.arange(smallest, largest, step):
        rolling = pd.DataFrame()
        for i in range(len(names[0 : number_of_layers - 1])):
            basement = np.full(400, 0) - np.random.rand(400) / 100
            elevation = np.full(400, j)
            topbasement = np.where(basement > elevation, elevation, basement)
            rolling["zero"] = topbasement
            strat_elevation = np.full(400, elevation_random[i])
            layer_elevation = np.where(
                strat_elevation > elevation, elevation, strat_elevation
            )
            rolling[names[i]] = layer_elevation
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
                np.random.randint(0, 399), np.random.randint(0, number_of_layers - 1),
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
            join_axes=[df_horizontal.index],
        )
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
    dataset.to_csv(
        str(number_of_layers) + "_layers_" + str(no_of_neighbors) + "neighbors.csv"
    )
    print(f"Done with {no_of_neighbors} neighbors")
