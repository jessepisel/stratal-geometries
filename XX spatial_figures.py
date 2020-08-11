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

from rasterio.warp import calculate_default_transform, reproject, Resampling
from matplotlib.colors import LinearSegmentedColormap
import pandas as pd
import matplotlib.pyplot as plt
import geopandas as gpd
import seaborn as sns
import rasterio
import rasterio.plot

sns.set()
sns.set_style("darkgrid", {"legend.frameon": True})
# %matplotlib inline

# +

fu = rasterio.open("ftunion2864.tif")
la = rasterio.open("lance2864.tif")
tfuwells = gpd.read_file(
    r"F:\Geology\WSGS\Projects\Unconformity or onlap\
predictions\shapefiles\ftunion_KNN_predictions_prob.shp"
)
klawells = gpd.read_file(
    r"F:\Geology\WSGS\Projects\Unconformity or onlap\
predictions\shapefiles\lance_KNN_predictions_prob.shp"
)
lance = gpd.read_file(r"shapefiles/lance_outcrop.shp")
ftun = gpd.read_file(r"shapefiles/ftunion_outcrop.shp")
faults = gpd.read_file(r"shapefiles/simp_faults.shp")
sections = gpd.read_file(r"shapefiles/crossection.shp",)
# -

dst_crs = "EPSG:4326"
with rasterio.open("lance2864.tif") as src:
    transform, width, height = calculate_default_transform(
        src.crs, dst_crs, src.width, src.height, *src.bounds
    )
    kwargs = src.meta.copy()
    kwargs.update(
        {
            "crs": dst_crs,
            "transform": transform,
            "width": width,
            "height": height,
        }
    )

    with rasterio.open("lance2864_reproject.tif", "w", **kwargs) as dst:
        for i in range(1, src.count + 1):
            reproject(
                source=rasterio.band(src, i),
                destination=rasterio.band(dst, i),
                src_transform=src.transform,
                src_crs=src.crs,
                dst_transform=transform,
                dst_crs=dst_crs,
                resampling=Resampling.nearest,
            )

fu = rasterio.open("ftunion2864_reproject.tif")
la = rasterio.open("lance2864_reproject.tif")

# +
filepath = "ftunion2864.tif"
fig, ax = plt.subplots(figsize=(10, 10))

with rasterio.open(filepath) as src:
    e = src.read().astype(int)
plt.imshow(e[0], cmap="gray", vmin=0, vmax=5200 * 0.3048)
plt.colorbar(label="Thickness (m)")
plt.savefig("fortunioncolorbar.pdf")

filepath = "lance2864.tif"
fig, ax = plt.subplots(figsize=(10, 10))

with rasterio.open(filepath) as src:
    e = src.read().astype(int)
plt.imshow(e[0], cmap="gray", vmin=0, vmax=4961 * 0.3048)
plt.colorbar(label="Thickness (m)")
plt.savefig("lancecolorbar.pdf")
# -

tfuwells = tfuwells.to_crs("epsg:4326")
klawells = klawells.to_crs("epsg:4326")
lance = lance.to_crs("epsg:4326")
ftun = ftun.to_crs("epsg:4326")
faults = faults.to_crs("epsg:4326")
sections = sections.to_crs("epsg:4326")

# +
truncation_color = "#ffffbf"
onlap_color = "#2c7bb6"
horiz_color = "#d7191c"


truncCmap = LinearSegmentedColormap.from_list(
    "mycmap", [onlap_color, truncation_color]
)
onlapCmap = LinearSegmentedColormap.from_list(
    "mycmap", [truncation_color, onlap_color]
)
horizCmap = LinearSegmentedColormap.from_list(
    "mycmap", [onlap_color, horiz_color]
)

# -

fthoriz = tfuwells[(tfuwells.horiz_prob > 0.0)]
lahoriz = klawells[(klawells.horiz_prob > 0.0)]


# +
fig, ax = plt.subplots(figsize=(10, 10))
rasterio.plot.show(fu, ax=ax, cmap="gray", zorder=1)
tfuwells.plot(
    ax=ax, column="trunc_prob", cmap=truncCmap, vmin=0, vmax=1, zorder=4
)
fthoriz.plot(
    ax=ax, column="horiz_prob", cmap=horizCmap, vmin=0, vmax=1, zorder=5
)

kwarg2s = {
    "facecolor": "#957F56",
    "edgecolor": "black",
    "linewidth": 0.5,
    "hatch": "",
}
ftun.plot(ax=ax, zorder=3, label="Fort Union Outcrop", alpha=0.8, **kwarg2s)
kwarg3s = {
    "facecolor": "#A6C551",
    "edgecolor": "black",
    "linewidth": 0.5,
    "hatch": "",
}
lance.plot(ax=ax, **kwarg3s, zorder=3, alpha=0.9)
# tfuwells[tfuwells.probabilit < 0.6].plot(ax=ax, zorder=7, color='none', edgecolor='red', label='Probability < 0.6', markersize=80)

# plt.legend()
plt.title("Fort Union Predictions", size="xx-large")
plt.xlim(-108.88, -107.35)
plt.ylim(40.97, 42.3)
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.savefig("fortunion.pdf", bbox_inches="tight")

# -


fig, ax = plt.subplots(figsize=(10, 10))
rasterio.plot.show(la, ax=ax, cmap="gray", zorder=1)
klawells.plot(
    ax=ax, column="trunc_prob", cmap=truncCmap, vmin=0, vmax=1, zorder=4
)
lahoriz.plot(
    ax=ax, column="horiz_prob", cmap=horizCmap, vmin=0, vmax=1, zorder=5
)
lance.plot(ax=ax, color="#A6C551", edgecolor="black", zorder=3, alpha=0.9)
plt.legend()
plt.title("Lance Formation Predictions", size="xx-large")
plt.xlim(-108.88, -107.35)
plt.ylim(40.97, 42.3)
plt.xlabel("Easting (m)")
plt.ylabel("Northing (m)")
plt.savefig("lance_prob.pdf", bbox_inches="tight")

klawells["form"] = "Kl"
tfuwells["form"] = "Tfu"
full = pd.DataFrame(klawells.append(tfuwells))


# +

names = ("Lance Formation", "Fort Union Formation")
plt.bar(
    [0], klawells.prediction.value_counts().values[0], color="#ffffbf"
)  # truncation
plt.bar(
    [0],
    klawells.prediction.value_counts().values[1],
    color="#2c7bb6",
    bottom=klawells.prediction.value_counts().values[0],
)  # onlap
plt.bar(
    [0],
    klawells.prediction.value_counts().values[2],
    color="#d7191c",
    bottom=klawells.prediction.value_counts().values[0]
    + klawells.prediction.value_counts().values[1],
)  # horizontal

plt.bar([1], tfuwells.prediction.value_counts().values[0], color="#ffffbf")
plt.bar(
    [1],
    tfuwells.prediction.value_counts().values[1],
    color="#2c7bb6",
    bottom=tfuwells.prediction.value_counts().values[0],
)
plt.bar(
    [1],
    tfuwells.prediction.value_counts().values[2],
    color="#d7191c",
    bottom=tfuwells.prediction.value_counts().values[0]
    + tfuwells.prediction.value_counts().values[1],
)


plt.xticks([0, 1], names)
plt.xlabel("Formation")
plt.ylabel("Counts")
plt.savefig("valuecounts histogram.pdf")
# -

fig, ax = plt.subplots(figsize=(10, 10))
tfuwells.plot(
    ax=ax,
    color="white",
    edgecolor="k",
    label="Well",
    legend=True,
    alpha=0.9,
    zorder=3,
)
kwarg2s = {
    "facecolor": "#957F56",
    "edgecolor": "black",
    "linewidth": 0.5,
    "hatch": "",
}
ftun.plot(ax=ax, zorder=2, label="Fort Union Outcrop", alpha=0.8, **kwarg2s)
kwarg3s = {
    "facecolor": "#A6C551",
    "edgecolor": "black",
    "linewidth": 0.5,
    "hatch": "",
}
lance.plot(ax=ax, **kwarg3s, zorder=1, alpha=0.9)
faults.plot(ax=ax, color="#474747", label="Faults", legend=True, zorder=4)
sections.plot(
    ax=ax,
    color="red",
    label="Cross Sections of Lynds and Lichtner (2016)",
    legend=True,
    zorder=5,
)
plt.legend()
plt.title("Eastern Greater Green River Basin", size="xx-large")
plt.xlim(-108.88, -107.35)
plt.ylim(40.97, 42.3)
plt.xlabel("Easting (m)")
plt.ylabel("Northing (m)")
plt.savefig("easternbasin.pdf")
