# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:hydrogen
#     text_representation:
#       extension: .py
#       format_name: hydrogen
#       format_version: '1.3'
#       jupytext_version: 1.3.2
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %%
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D 
import seaborn as sns; sns.set()
from scipy.spatial.distance import pdist, squareform
%matplotlib inline

# %%
truncation_color = '#ffffbf'
onlap_color = '#2c7bb6'
horiz_color = '#d7191c'

from matplotlib.colors import LinearSegmentedColormap

truncCmap = LinearSegmentedColormap.from_list('mycmap', ['#ffffff', truncation_color])
onlapCmap = LinearSegmentedColormap.from_list('mycmap', ['#ffffff', onlap_color])
horizCmap = LinearSegmentedColormap.from_list('mycmap', ['#ffffff', horiz_color])


# %%
#data = pd.read_csv(r'F:\Geology\WSGS\Projects\jupyter\20neighbors.csv', index_col=[0])
#data_subset0 = data.drop(['class'], axis=1)

# %%
from sklearn.model_selection import train_test_split
no_of_neighbors =25

# %%
dataset = pd.read_csv(
    r"F:/Geology/WSGS/Projects/jupyter/0"+str(no_of_neighbors)+"neighbors.csv",
    index_col=[0],
)

from sklearn.model_selection import train_test_split

# next let's split our toy data into training and test sets, choose how much with test_size of the data becomes the test set
X_train, X_test, y_train, y_test = train_test_split(
    dataset.drop('class', axis=1),
    dataset['class'],
    test_size=0.1, #don't forget to change this
    random_state=86,
)

# %%
from sklearn.neighbors import KNeighborsClassifier
neigh = KNeighborsClassifier(algorithm='ball_tree', leaf_size=10, metric='manhattan',
                     metric_params=None, n_jobs=None, n_neighbors=5, p=2,
                     weights='distance')
neigh.fit(X_train, y_train)
neigh.score(X_test, y_test)


# %%
def flatten(container):
    for i in container:
        if isinstance(i, (list,tuple)):
            for j in flatten(i):
                yield j
        else:
            yield i


# %%
initial = ["thickness", "thickness natural log", "thickness power"]
features = []
for item in initial:
    features.append(item)
    for i in range(1, no_of_neighbors + 1):
        features.append(item + " neighbor " + str(i))
features.append(["x location", "y location", "class"])
flat_features = list(flatten(features))

# %%
flat_features
thickened =  flat_features[0:no_of_neighbors+1]
thickened.append('class')
logged = flat_features[no_of_neighbors+1:2*no_of_neighbors+2]
logged.append('class')
powered = flat_features[2*no_of_neighbors+2:3*no_of_neighbors+3]
powered.append('class')
location = ['x location', 'y location', 'class']
og_thickness = ['thickness', 'class']

# %%
X_train, X_test, y_train, y_test = train_test_split(
    dataset.drop(thickened, axis=1),
    dataset['class'],
    test_size=0.1, #don't forget to change this
    random_state=86,
)
neigh.fit(X_train, y_train)
thickness_removed = neigh.score(X_test, y_test)
print(f'Without thickness accuracy is {thickness_removed:.3f}')


X_train, X_test, y_train, y_test = train_test_split(
    dataset.drop(logged, axis=1),
    dataset['class'],
    test_size=0.1, #don't forget to change this
    random_state=86,
)
neigh.fit(X_train, y_train)
ln_removed = neigh.score(X_test, y_test)
print(f'Without natural log. Accuracy is {ln_removed:.2f}')

X_train, X_test, y_train, y_test = train_test_split(
    dataset.drop(powered, axis=1),
    dataset['class'],
    test_size=0.1, #don't forget to change this
    random_state=86,
)
neigh.fit(X_train, y_train)
power_removed = neigh.score(X_test, y_test)
print(f'Without power accuracy is {power_removed:.2f}')

X_train, X_test, y_train, y_test = train_test_split(
    dataset.drop(location, axis=1),
    dataset['class'],
    test_size=0.1, #don't forget to change this
    random_state=86,
)
neigh.fit(X_train, y_train)
location_removed = neigh.score(X_test, y_test)
print(f'Without location accuracy is {location_removed:.2f}')

X_train, X_test, y_train, y_test = train_test_split(
    dataset.drop(og_thickness, axis=1),
    dataset['class'],
    test_size=0.1, #don't forget to change this
    random_state=86,
)
neigh.fit(X_train, y_train)
og_t_removed = neigh.score(X_test, y_test)
print(f'Done with well thickness. Accuracy is {og_t_removed:.2f}')

# %%
X_train, X_test, y_train, y_test = train_test_split(
    dataset.drop('class', axis=1),
    dataset['class'],
    test_size=0.1, #don't forget to change this
    random_state=86,
)
neigh = KNeighborsClassifier(algorithm='ball_tree', leaf_size=10, metric='manhattan',
                     metric_params=None, n_jobs=None, n_neighbors=5, p=2,
                     weights='distance')
neigh.fit(X_train, y_train)


# %%
tops_api = pd.read_csv(r"F:\Geology\WSGS\Projects\Unconformity or onlap\Python\ftunion.csv").fillna(
    0
)  # this file is available in the unconformity or onlap folder in the repo
iterable = ["Kfh",  "Kl", "Tfu"]
topcombos = list(zip(iterable, iterable[1:]))

#topcombos.append(("Kfh", "Kl"))
#topcombos.append(("Kl", "Tfu"))

# %%
# run this for all combinations of 2 tops and KNN
results = []
norm_all = []
probs_all = []
full_probs = []

for j in enumerate(topcombos):
    print(topcombos[j[0]])
    tops_api = pd.read_csv(r"F:\Geology\WSGS\Projects\Unconformity or onlap\Python\ftunion.csv").fillna(
        0
    )  # this file is available in the unconformity or onlap folder in the repo
    fmtops = list(topcombos[j[0]])
    fmtops.extend(["x", "y"])
    tops = tops_api[fmtops]

    # calculate thicknesses and neighbors for the two tops
    hood = squareform(pdist(tops.iloc[:, -2:]))
    neighbors = []
    for i in enumerate(hood.argsort()[0:, 1 : no_of_neighbors + 1]):
        selected = (
            tops.iloc[hood.argsort()[i[0], 1 : no_of_neighbors + 1], 0:-2]
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

    well_preds = neigh.predict(real_data) #knn predictions
    well_prob = neigh.predict_proba(real_data) #knn predictions
    full_probs.append(well_prob)
    probs = []
    for i in range(len(well_prob)):
        probs.append(well_prob[i].max())
    probs_all.append(probs)
    results.append(well_preds)
    norm_all.append(normalized_rw)

# %%
normalized_kl = norm_all[0]
normalized_tfu = norm_all[1]

normalized_kl.columns = dataset.columns[0:-1].values
normalized_kl['class'] = results[0]
normalized_kl['prob'] = probs_all[0]

normalized_tfu.columns = dataset.columns[0:-1].values
normalized_tfu['class'] = results[1]
normalized_tfu['prob'] = probs_all[1]

#df_subset = data#.sample(80000, random_state=20)
#df_subset['model'] = 'geometric' #this is geometric
normalized_kl['Formation'] = 'Kl' # this is lance
normalized_tfu['Formation'] = 'Tfu' # this is ft union

#df_combined = df_subset.append(normalized_kl, sort=False)
df_combined = normalized_kl
df_combined1 = df_combined.append(normalized_tfu, sort=False)


# %%
#df_subset = data.sample(5000)
df_subset1 = df_combined1.drop(['class', 'Formation', 'prob'], axis=1)
tsne = TSNE(n_components=2, verbose=0.2, perplexity=50, n_iter=1500, learning_rate=500,random_state=20) #per=250, iter = 500, lr=50
tsne_results = tsne.fit_transform(df_subset1)

# %%
probabilities = np.vstack(full_probs)

df_combined1['trunc_prob'] = probabilities[:,0]
df_combined1['onlap_prob'] = probabilities[:,1]
df_combined1['horiz_prob'] = probabilities[:,2]

# %%
df_combined1['tsne-2d-one'] = tsne_results[:,0]
df_combined1['tsne-2d-two'] = tsne_results[:,1]
#df_combined1['tsne-2d-three'] = tsne_results[:,2]
color_pals = ['#ffffbf', '#2c7bb6', '#d7191c']
plt.figure(figsize=(10,10))
# 0 is truncation, 1 is onlap, 2 is horizontal
sns.scatterplot(
    x=df_combined1["tsne-2d-one"], y=df_combined1["tsne-2d-two"],
    hue=df_combined1["trunc_prob"],
    style=df_combined1['Formation'],
    palette=truncCmap,
    data=df_combined1,
    legend=None,
    alpha=1,
    edgecolor='none',
    vmin=-1, vmax=1
)

sns.scatterplot(
    x=df_combined1["tsne-2d-one"], y=df_combined1["tsne-2d-two"],
    hue=df_combined1["onlap_prob"],
    style=df_combined1['Formation'],
    palette=onlapCmap,
    data=df_combined1,
    legend=None,
    #alpha=0.5,
    edgecolor='none'
)

horizontals = df_combined1[(df_combined1.horiz_prob>0.)]
sns.scatterplot(
    x=horizontals["tsne-2d-one"], y=horizontals["tsne-2d-two"],
    hue=horizontals["horiz_prob"],
    style=horizontals['Formation'],
    palette=horizCmap,
    data=horizontals,
    legend=None,
    alpha=1,
    edgecolor='none',
    vmin=0,
    vmax=1
)
plt.xlabel('t-SNE Dimension 1')
plt.ylabel('t-SNE Dimension 2')
#plt.xlim(-7,7)
#plt.ylim(-13,13)
#plt.savefig('tsne.pdf', bbox_inches='tight')


# %%
x_locs = tops_api.x.append(tops_api.x,  ignore_index=True)
y_locs = tops_api.y.append(tops_api.y,  ignore_index=True)
api = tops_api.API.append(tops_api.API,  ignore_index=True)

df_combined1['x_locs'] =x_locs
df_combined1['y_locs'] = y_locs
df_combined1['api'] = api

# %%
ftunion = df_combined1[df_combined1['Formation']=='Tfu']
lancer = df_combined1[df_combined1['Formation']=='Kl']

# %%
from sklearn.neighbors import KNeighborsClassifier
import itertools
from geopandas import GeoDataFrame
from shapely.geometry import Point
import fiona

# %% [markdown]
# geometry = [Point(xy) for xy in zip(ftunion.x_locs, ftunion.y_locs)]
# crs = {"init": "epsg:3732"}
# geo_df = GeoDataFrame(ftunion, crs={"init": "epsg:4326"}, geometry=geometry)
# geo_df.to_file(
#     driver="ESRI Shapefile",
#     filename=r"F:\Geology\WSGS\Projects\Unconformity or onlap\predictions\shapefiles\ftunion_KNN_predictions_prob.shp",
# )
#
# geometry = [Point(xy) for xy in zip(lancer.x_locs, lancer.y_locs)]
# crs = {"init": "epsg:3732"}
# geo_df = GeoDataFrame(lancer, crs={"init": "epsg:4326"}, geometry=geometry)
# geo_df.to_file(
#     driver="ESRI Shapefile",
#     filename=r"F:\Geology\WSGS\Projects\Unconformity or onlap\predictions\shapefiles\lance_KNN_predictions_prob.shp",
# )
#

# %%
plt.figure(figsize=(15,15))

plt.scatter(lancer['x_locs'], lancer['y_locs'], c=lancer['trunc_prob'], cmap=truncCmap, vmin=0, vmax=1)
plt.savefig('lance_1.pdf')
plt.clf()
plt.scatter(lancer['x_locs'], lancer['y_locs'], c=lancer['onlap_prob'], cmap=onlapCmap, vmin=0, vmax=1)
plt.savefig('lance_2.pdf')
plt.clf()
plt.scatter(lancer['x_locs'], lancer['y_locs'], c=lancer['horiz_prob'], cmap=horizCmap, vmin=0, vmax=1)
plt.savefig('lance_2.pdf')

# %%
plt.figure(figsize=(15,15))

plt.scatter(ftunion['x_locs'], ftunion['y_locs'], c=ftunion['trunc_prob'], cmap=truncCmap, vmin=0, vmax=1)
plt.savefig('union_1.pdf')
plt.clf()
plt.scatter(ftunion['x_locs'], ftunion['y_locs'], c=ftunion['onlap_prob'], cmap=onlapCmap, vmin=0, vmax=1)
plt.savefig('union_2.pdf')
plt.clf()
plt.scatter(ftunion['x_locs'], ftunion['y_locs'], c=ftunion['horiz_prob'], cmap=horizCmap, vmin=0, vmax=1)
plt.savefig('Union_3.pdf')

# %%
import matplotlib.pyplot as plt,numpy as np

def gauplot(centers, radiuses, xr=None, yr=None):
        nx, ny = 1000.,1000.
        xgrid, ygrid = np.mgrid[xr[0]:xr[1]:(xr[1]-xr[0])/nx,yr[0]:yr[1]:(yr[1]-yr[0])/ny]
        im = xgrid*0 + np.nan
        xs = np.array([np.nan])
        ys = np.array([np.nan])
        fis = np.concatenate((np.linspace(-np.pi,np.pi,100), [np.nan]) )
        cmap = plt.cm.gray
        cmap.set_bad(truncCmap)
        thresh = 3
        for curcen,currad in zip(centers,radiuses):
                curim=(((xgrid-curcen[0])**2+(ygrid-curcen[1])**2)**.5)/currad*thresh
                im[curim<thresh]=np.exp(-.5*curim**2)[curim<thresh]
                xs = np.append(xs, curcen[0] + currad * np.cos(fis))
                ys = np.append(ys, curcen[1] + currad * np.sin(fis))
        plt.imshow(im.T, cmap=cmap, extent=xr+yr)
        plt.plot(xs, ys, 'r-')
gauplot([(0,0)], [1], [-1,10], [-1,10])

# %%
import ternary
fig, tax = ternary.figure(scale=1)
fig.set_size_inches(5, 4.5)

tax.scatter(lancer[['trunc_prob', 'onlap_prob', 'horiz_prob']].values)
tax.left_axis_label("Truncation Probability", fontsize=12, offset=0.08)
tax.right_axis_label("Onlap Probability", fontsize=12, offset=0.08)
tax.bottom_axis_label("Horizontal Probability", fontsize=12, offset=-0.08)
tax.gridlines(multiple=20)
tax.get_axes().axis('off')

tax.boundary(linewidth=1)
tax.gridlines(multiple=0.10, color="gray")
tax.ticks(axis='lbr', linewidth=1, multiple=0.20)
tax.get_axes().axis('off')


# %%
!pip install plotly

# %%


import plotly.io as pio
pio.renderers.default = "browser"


# %%
import plotly.express as px
px.scatter_ternary(lancer, a="horiz_prob", b="trunc_prob", c="onlap_prob", color="prob", color_continuous_scale='fall')


# %%
lancer.head()

# %%
lahoriz = lancer[(lancer.horiz_prob>0.)]

plt.figure(figsize=(30,10))
plt.subplot(121)
plt.scatter(lancer['x_locs'], lancer['y_locs'], c=lancer['trunc_prob']*-1, cmap=truncCmap, vmin=-1, vmax=0)
plt.scatter(lancer['x_locs'], lancer['y_locs'], c=lancer['onlap_prob'], cmap=onlapCmap, vmin=0, vmax=1)
plt.scatter(lahoriz['x_locs'], lahoriz['y_locs'], c=lahoriz['horiz_prob'], cmap=horizCmap, vmin=0, vmax=1)
plt.colorbar()
plt.subplot(122)
plt.scatter(lancer['x_locs'], lancer['y_locs'], alpha=0.1, c='k')
plt.scatter(lancer['x_locs'], lancer['y_locs'], c=lancer['onlap_prob'], cmap=onlapCmap)
plt.colorbar()
#plt.savefig('kla_probabilites.pdf')

# %% [markdown]
# test = df_combined1[df_combined1.Formation=='Kl']
# test['t'] = full_probs[0][:,0]
# test['o'] = full_probs[0][:,1]
# test['h'] = full_probs[0][:,2]
# trunc = test[(test.o <.6)&(test['class']==1)]
# onl = test[test.o <0.6]
# hor = test[test.h <0.6]

# %%
fig, ax = plt.subplots()
scatter = ax.scatter(lancer['x_locs'], lancer.y_locs, c=lancer['class'])
legend1 = ax.legend(*scatter.legend_elements(),
                    loc="lower left", title="Classes")
ax.add_artist(legend1)

# %%
fig, ax = plt.subplots()
scatter = ax.scatter(ftunion['x_locs'], ftunion.y_locs, c=ftunion['class'])
legend1 = ax.legend(*scatter.legend_elements(),
                    loc="lower left", title="Classes")
ax.add_artist(legend1)

# %%
