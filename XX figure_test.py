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
import matplotlib.pyplot as plt
import geopandas as gpd
import seaborn as sns

sns.set()
sns.set_style("darkgrid", {"legend.frameon": True})
# %matplotlib inline

TFUWELLS = gpd.read_file(
    r"D:\Geology\WSGS\Projects\Unconformity or onlap\predictions\shapefiles\
    ('Kl', 'Tfu')_KNN_predictions.shp"
)
KLAWELLS = gpd.read_file(
    r"D:\Geology\WSGS\Projects\Unconformity or onlap\predictions\shapefiles\
    ('Kfh', 'Kl')_KNN_predictions.shp"
)
LANCE = gpd.read_file(r"shapefiles/LANCE_outcrop.shp")
FTUN = gpd.read_file(r"shapefiles/FTUNion_outcrop.shp")
FAULTS = gpd.read_file(r"shapefiles/simp_FAULTS.shp")
SECTIONS = gpd.read_file(r"shapefiles/crossection.shp",)
# -

TFUWELLS.head()

plt.figure(FIGsize=(10, 10))
plt.scatter(
    TFUWELLS.LON,
    TFUWELLS.LAT,
    c=TFUWELLS.probabilit,
    cmap="viridis",
    vmin=0.5,
    vmAX=1,
)
plt.colorbar()

NEWSCORE = 50 * (TFUWELLS.probabilit)
TFUWELLS["newprob"] = NEWSCORE
KLAWELLS["newprob"] = NEWSCORE
FUTRUNCATION = TFUWELLS[TFUWELLS["prediction"] == 0]
FUONLAP = TFUWELLS[TFUWELLS["prediction"] == 1]
FUHORIZONTAL = TFUWELLS[TFUWELLS["prediction"] == 2]
LATRUNCATION = KLAWELLS[KLAWELLS["prediction"] == 0]
LAONLAP = KLAWELLS[KLAWELLS["prediction"] == 1]
LAHORIZONTAL = KLAWELLS[KLAWELLS["prediction"] == 2]

FIG, AX = plt.subplots(FIGsize=(10, 10))
FUHORIZONTAL.plot(
    AX=AX,
    markersize=FUHORIZONTAL["newprob"] / 4,
    color="#d7191c",
    label="Horizontal",
    legend=True,
    alpha=0.9,
    zorder=6,
)
FUTRUNCATION.plot(
    AX=AX,
    markersize=FUTRUNCATION["newprob"] / 4,
    color="#ffffbf",
    label="Truncation",
    legend=True,
    alpha=0.9,
    zorder=4,
)
FUONLAP.plot(
    AX=AX,
    markersize=FUONLAP["newprob"] / 4,
    color="#2c7bb6",
    label="Onlap",
    legend=True,
    alpha=0.9,
    zorder=5,
)
KWARG2S = {
    "facecolor": "#957F56",
    "edgecolor": "black",
    "linewidth": 0.5,
    "hatch": "",
}
FTUN.plot(AX=AX, zorder=3, label="Fort Union Outcrop", alpha=0.8, **KWARG2S)
KWARG3S = {
    "facecolor": "#A6C551",
    "edgecolor": "black",
    "linewidth": 0.5,
    "hatch": "",
}
LANCE.plot(AX=AX, **KWARG3S, zorder=3, alpha=0.9)
plt.legend()
plt.title("Fort Union Predictions", size="xx-large")
plt.xlabel("Easting (m)")
plt.ylabel("Northing (m)")
# plt.saveFIG('fortunion.pdf', bbox_inches='tight')

# +
FIG, AX = plt.subplots(FIGsize=(10, 10))

LAHORIZONTAL.plot(
    AX=AX,
    markersize=LAHORIZONTAL["newprob"] / 4,
    color="#d7191c",
    label="Horizontal",
    legend=True,
    zorder=6,
)
LATRUNCATION.plot(
    AX=AX,
    markersize=LATRUNCATION["newprob"] / 4,
    color="#ffffbf",
    label="Truncation",
    legend=True,
    zorder=4,
)
LAONLAP.plot(
    AX=AX,
    markersize=LAONLAP["newprob"] / 4,
    color="#2c7bb6",
    label="Onlap",
    legend=True,
    zorder=5,
)
KWARG3S = {
    "facecolor": "#A6C551",
    "edgecolor": "black",
    "linewidth": 1.5,
    "hatch": "xx",
}
LANCE.plot(AX=AX, **KWARG3S, zorder=3, alpha=0.9)
plt.legend()
plt.title("LANCE Formation Predictions", size="xx-large")
plt.xlabel("Easting (m)")
plt.ylabel("Northing (m)")
# plt.saveFIG('LANCE.pdf', bbox_inches='tight')
# -
