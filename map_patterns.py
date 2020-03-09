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

# # Ambiguous map patterns from stratigraphy

# +
# some imports, using verde to grid
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import verde as vd

import warnings
warnings.filterwarnings("ignore")
# %matplotlib inline
# -

# ## Here we make some data with angular unconformity and onlap

# +
NAMES = [
    "one",
    "two",
    "three",
    "four",
    "five",
    "six",
    "seven",
    "eight",
    "nine",
    "ten",
    "eleven",
    "twelve",
    "thirteen",
    "fourteen",
    "fifteen",
    "sixteen",
]  # this creates dummy NAMES for the formations
NUMBER_OF_LAYERS = (
    6  # this is the number of tops you want in your training data
)

# this section builds some toy data with an angular unconformity

DF = pd.DataFrame()
for j in np.arange(-5, 7, 2):
    rolling = pd.DataFrame()
    rolling["zero"] = (
        10 * np.sin(1 - np.arange(0, 40, 0.1) / 15.28)
        + np.random.rand(400) / 100
    )
    for i in range(len(NAMES[0:NUMBER_OF_LAYERS])):
        layer_elevation = 10 * np.sin(
            1 - np.arange(0, 40, 0.1) / 15.28
        ) + np.random.uniform(0, j)
        elevation = np.full(400, j) + np.random.rand(400) / 100
        layer_elevation = (
            np.where(layer_elevation > elevation, elevation, layer_elevation)
            + np.random.rand(400) / 100
        )
        rolling[NAMES[i]] = layer_elevation
    x = np.arange(0, 40, 0.1)
    y = np.random.randint(0, 90, len(x))
    rolling["ex"] = x
    rolling["ey"] = y
    DF = pd.concat((DF, rolling))
ADEC = []
for i in range(len(DF)):
    ADEC.append(DF.iloc[i, 1:-2].sort_values()[0:].values)
S3 = pd.DataFrame(ADEC)
S3.index = DF.index.values
ANGULAR_THICKNESSES = S3.diff(axis=1)


# now for onlap training data construction
DF_ONLAP = pd.DataFrame()
for j in np.arange(-5, 10, 2):
    rolling = pd.DataFrame()
    rolling["zero"] = 10 * np.sin(1 - np.arange(0, 40, 0.1) / 15.28)
    for i in range(len(NAMES[0:NUMBER_OF_LAYERS])):
        zero = (
            10 * np.sin(1 - np.arange(0, 40, 0.1) / 15.28)
            + np.random.rand(400) / 100
        )
        randomness = np.random.uniform(0, j)
        elevation = np.full(400, j) + np.random.rand(400) / 100
        onlap = np.where(
            np.full(400, randomness) > zero,
            np.full(400, randomness) + np.random.rand(400) / 100,
            zero,
        )
        layer_elevation = np.where(onlap > elevation, elevation, onlap)
        rolling[NAMES[i]] = layer_elevation - np.arange(0, 10, 0.025)
    x = np.arange(0, 40, 0.1)
    y = np.random.randint(0, 90, len(x))
    rolling["ex"] = x
    rolling["ey"] = y
    DF_ONLAP = pd.concat((DF_ONLAP, rolling))
DEC = []
for i in range(len(DF_ONLAP)):
    DEC.append(DF_ONLAP.iloc[i, 1:-2].sort_values()[0:].values)
S = pd.DataFrame(DEC)
S.index = DF_ONLAP.index.values
ONLAP_THICKNESSES = S.diff(axis=1)
# -

# ### This is what the data looks like in cross section

DF_ONLAP.iloc[2000:2400, 1:-2].plot(
    legend=False, figsize=(20, 10), cmap="copper"
).set_aspect(6)
plt.title("A")
DF.iloc[2000:2400, 1:-2].plot(
    legend=False, figsize=(20, 10), cmap="copper"
).set_aspect(
    6
)  # each x-section is 400 points long
plt.title("B")


# ## now let's grid the data using Verde and plot it up


SPLINE = vd.Spline()
SPLINE.fit(
    (DF.iloc[2000:2400, -2] * 10, DF.iloc[2000:2400, -1] * 10),
    ANGULAR_THICKNESSES.iloc[2000:2400, 1] * 100,
)
AUIGRID = SPLINE.grid(spacing=1, data_names=["thickness"])
THICK_PROFILE = SPLINE.profile((0,450),(400,450), size=400).scalars
AUIGRID.thickness.plot.pcolormesh(cmap="magma", vmin=0, vmax=150)
plt.scatter(DF.iloc[2000:2400, -2]*10, DF.iloc[2000:2400,-1]*10, color='None', edgecolor='gray', label='Well location', s=15)
plt.title("Angular Unconformity Isochore")

SPLINE = vd.Spline()
SPLINE.fit(
    (DF.iloc[2000:2400, -2] * 10, DF.iloc[2000:2400, -1] * 10),
    DF.iloc[2000:2400, 1] * 100,
)
AUSGRID = SPLINE.grid(spacing=1, data_names=["depth"])
DEPTH_PROFILE = SPLINE.profile((0,450),(400,450), size=400).scalars
AUSGRID.depth.plot.pcolormesh(cmap="viridis", vmin=100, vmax=-500)
plt.scatter(DF.iloc[2000:2400, -2]*10, DF.iloc[2000:2400,-1]*10, color='None', edgecolor='gray', label='Well location', s=15)
plt.title("Angular Unconformity Structure")

plt.plot(DEPTH_PROFILE)
plt.plot(DEPTH_PROFILE-THICK_PROFILE)

# +
SPLINE = vd.Spline()
SPLINE.fit(
    (DF_ONLAP.iloc[2000:2400, -2] * 10, DF_ONLAP.iloc[2000:2400, -1] * 10),
    ONLAP_THICKNESSES.iloc[2000:2400, 3] * 100,
)
OLIGRID = SPLINE.grid(spacing=1, data_names=["thickness"])
OLTHICK_PROFILE = SPLINE.profile((0,450),(400,450), size=400).scalars
OLIGRID.thickness.plot.pcolormesh(cmap="magma", vmin=0, vmax=150)
plt.scatter(DF_ONLAP.iloc[2000:2400, -2]*10, DF_ONLAP.iloc[2000:2400,-1]*10, color='None', edgecolor='gray', label='Well location', s=15)

plt.title("Onlap Isochore")

# +
SPLINE.fit(
    (DF_ONLAP.iloc[2000:2400, -2] * 10, DF_ONLAP.iloc[2000:2400, -1] * 10),
    DF_ONLAP.iloc[2000:2400, 3] * 100,
)
OLSGRID = SPLINE.grid(spacing=1, data_names=["depth"])
OLDEPTH_PROFILE = SPLINE.profile((0,450),(400,450), size=400).scalars
OLSGRID.depth.plot.pcolormesh(cmap="viridis", vmin=100, vmax=-500)
plt.scatter(DF_ONLAP.iloc[2000:2400, -2]*10, DF_ONLAP.iloc[2000:2400,-1]*10, color='None', edgecolor='gray', label='Well location', s=15)

plt.title("Onlap Structure")
# -

# ### we can decimate the data to see what it would look like with fewer 'wells' in the cross section

DECIMATE_ONL = DF_ONLAP.iloc[2000:2400].sample(
    frac=0.99
)  # fraction is the percentage of the data you want to sample ie-0.9 means
# sample 90% of the data
len(DECIMATE_ONL)

DEC = []
for i in range(len(DECIMATE_ONL)):
    DEC.append(
        DECIMATE_ONL.sort_index().iloc[i, 1:-2].sort_values()[0:6].values
    )
S2 = pd.DataFrame(DEC)
S2.index = DECIMATE_ONL.sort_index().index.values
S2.plot(legend=False, figsize=(20, 10), cmap="copper").set_aspect(10)
plt.title("A.")

DECIMATE_ANG = DF.iloc[2000:2400].loc[S2.diff(axis=1).index.values]

ADEC = []
for i in range(len(DECIMATE_ANG)):
    ADEC.append(
        DECIMATE_ANG.sort_index().iloc[i, 1:-2].sort_values()[0:6].values
    )
S3 = pd.DataFrame(ADEC)
S3.index = DECIMATE_ANG.sort_index().index.values
S3.plot(legend=False, figsize=(20, 10), cmap="copper").set_aspect(10)
plt.title("B.")

# +
plt.figure(figsize=(30, 20))
ax1 = plt.subplot(321)
OLSGRID.depth.plot.pcolormesh(cmap="viridis", vmin=100, vmax=-500)
plt.scatter(DF_ONLAP.iloc[2000:2400, -2]*10, DF_ONLAP.iloc[2000:2400,-1]*10, color='None', edgecolor='gray', label='Well location', s=15)
plt.xlim(0,400)
plt.ylim(0,900)
plt.title("Onlap Structure")


plt.subplot(322, sharex=ax1)
AUIGRID.thickness.plot.pcolormesh(cmap="magma", vmin=0, vmax=150)
plt.scatter(DF.iloc[2000:2400, -2]*10, DF.iloc[2000:2400,-1]*10, color='None', edgecolor='gray', label='Well location', s=15)
plt.xlim(0,400)
plt.ylim(0,900)
plt.title("Truncation Isochore")


plt.subplot(323, sharex=ax1)
OLIGRID.thickness.plot.pcolormesh(cmap="magma", vmin=0, vmax=120)
plt.scatter(DF_ONLAP.iloc[2000:2400, -2]*10, DF_ONLAP.iloc[2000:2400,-1]*10, color='None', edgecolor='gray', label='Well location', s=15)
plt.xlim(0,400)
plt.ylim(0,900)
plt.title("Onlap Isochore")

plt.subplot(324, sharex=ax1)
AUSGRID.depth.plot.pcolormesh(cmap="viridis", vmin=100, vmax=-500)
plt.scatter(DF.iloc[2000:2400, -2]*10, DF.iloc[2000:2400,-1]*10, color='None', edgecolor='gray', label='Well location', s=15)
plt.xlim(0,400)
plt.ylim(0,900)
plt.title("Truncation Structure")

plt.subplot(325, sharex=ax1)
plt.plot(S2)
plt.title("Cross Section")

plt.subplot(326, sharex=ax1)
plt.plot(S3 * 100)
plt.title("Cross Section")


# +
fig, axes = plt.subplots(nrows=3, ncols=2, sharex=True, figsize=(20, 25))
fig.set_size_inches(7.5, 9.75)
fig.subplots_adjust(
    left=None, bottom=None, right=None, top=None, wspace=0.2, hspace=0.03
)

axes[0, 0].plot(S2 * 100, c="black")
axes[0, 0].plot(S2[3] * 100, c="orange")
axes[0, 0].plot(S2[2] * 100, c="orange")

axes[0, 1].plot(S3 * 100, c="black")
axes[0, 1].plot(S3[3] * 100, c="orange")
axes[0, 1].plot(S3[2] * 100, c="orange")

IM0 = axes[1, 0].imshow(
    OLSGRID.depth,
    extent=[0, 400, 0, 900],
    aspect="auto",
    cmap="viridis",
    vmin=-900,
    vmax=500,
)
#axes[1,0].scatter(DF_ONLAP.iloc[2000:2400, -2]*10, DF_ONLAP.iloc[2000:2400,-1]*10, color='None', edgecolor='gray', label='Well location', s=15)
#plt.xlim(0,400)
#plt.ylim(0,900)


IM1 = axes[1, 1].imshow(
    AUSGRID.depth,
    extent=[0, 400, 0, 900],
    aspect="auto",
    cmap="viridis",
    vmin=-900,
    vmax=500,
)

#axes[1,1].scatter(DF.iloc[2000:2400, -2]*10, DF.iloc[2000:2400,-1]*10, color='None', edgecolor='gray', label='Well location', s=15)
#plt.xlim(0,400)
#plt.ylim(0,900)

IM2 = axes[2, 0].imshow(
    OLIGRID.thickness,
    extent=[0, 400, 0, 900],
    aspect="auto",
    cmap="plasma",
    vmin=0,
    vmax=225,
)
#axes[2,0].scatter(DF_ONLAP.iloc[2000:2400, -2]*10, DF_ONLAP.iloc[2000:2400,-1]*10, color='None', edgecolor='gray', label='Well location', s=15)
#plt.xlim(0,400)
#plt.ylim(0,900)

IM3 = axes[2, 1].imshow(
    AUIGRID.thickness,
    extent=[0, 400, 0, 900],
    aspect="auto",
    cmap="plasma",
    vmin=0,
    vmax=225,
)
#axes[2,1].scatter(DF.iloc[2000:2400, -2]*10, DF.iloc[2000:2400,-1]*10, color='None', edgecolor='gray', label='Well location', s=15)
#plt.xlim(0,400)
#plt.ylim(0,900)

fig.subplots_adjust(left=0.07, right=0.87)
# Add the colorbar outside...
box = axes[1, 1].get_position()
pad, width = 0.005, 0.01
cax = fig.add_axes([box.xmax + pad, box.ymin, width, box.height])
fig.colorbar(IM0, cax=cax, label="Depth")


box = axes[2, 1].get_position()
pad, width = 0.005, 0.01
cax = fig.add_axes([box.xmax + pad, box.ymin, width, box.height])
fig.colorbar(IM3, cax=cax, label="Thickness")


#fig.savefig("figure2.pdf")
# -

profile_east_max

# +
onlap_unit = 5
angular_unit = 1

for i in range(1,150):

    FRACTION = i*0.01
    DECIMATE_ANG = DF.iloc[2000:2400].sample(
        frac=FRACTION, random_state=20
    )  
    SPLINE.fit(
        (DECIMATE_ANG.iloc[:, -2] * 10, DECIMATE_ANG.iloc[:, -1] * 10),
        DECIMATE_ANG.iloc[:, angular_unit] * 100,
    )
    AUSGRID = SPLINE.grid(spacing=1, data_names=["depth"])
    AUDEPTH_PROFILE = SPLINE.profile((0,450),(400,450), size=400).scalars

    SPLINE.fit(
        (DECIMATE_ANG.iloc[:, -2] * 10, DECIMATE_ANG.iloc[:, -1] * 10),
        DECIMATE_ANG.diff(axis=1).iloc[:, angular_unit] * 50,
    )
    AUIGRID = SPLINE.grid(spacing=1, data_names=["thickness"])
    AUTHICK_PROFILE = SPLINE.profile((0,450),(400,450), size=400).scalars
    AUTHICK_PROFILE = np.where(AUTHICK_PROFILE<0,0,AUTHICK_PROFILE)

    
    DECIMATE_ONL = DF_ONLAP.iloc[2000:2400].sample(
        frac=FRACTION, random_state=20
    )  
    SPLINE.fit(
        (DECIMATE_ONL.iloc[:, -2] * 10, DECIMATE_ONL.iloc[:, -1] * 10),
        DECIMATE_ONL.iloc[:, onlap_unit] * 100,
    )
    OLSGRID = SPLINE.grid(spacing=1, data_names=["depth"])
    OLDEPTH_PROFILE = SPLINE.profile((0,450),(400,450), size=400).scalars

    SPLINE.fit(
        (DECIMATE_ONL.iloc[:, -2] * 10, DECIMATE_ONL.iloc[:, -1] * 10),
        DECIMATE_ONL.diff(axis=1).iloc[:, onlap_unit] * 100,
    )
    OLIGRID = SPLINE.grid(spacing=1, data_names=["thickness"])
    OLTHICK_PROFILE = SPLINE.profile((0,450),(400,450), size=400).scalars
    OLTHICK_PROFILE = np.where(OLTHICK_PROFILE<0,0,OLTHICK_PROFILE)

    
    
    
    
    
    DEC = []
    for i in range(len(DECIMATE_ONL)):
        DEC.append(
            DECIMATE_ONL.sort_index().iloc[i, 1:-2].sort_values()[0:6].values
        )
    S2 = pd.DataFrame(DEC)
    S2.index = DECIMATE_ONL.sort_index().index.values
    
    ADEC = []
    for i in range(len(DECIMATE_ANG)):
        ADEC.append(
            DECIMATE_ANG.sort_index().iloc[i, 1:-2].sort_values()[0:6].values
        )
    S3 = pd.DataFrame(ADEC)
    S3.index = DECIMATE_ANG.sort_index().index.values
    

    fig, axes = plt.subplots(nrows=3, ncols=2, sharex=True, figsize=(20, 25))
    fig.suptitle(str(len(DECIMATE_ANG))+' Wells Drilled')

    fig.set_size_inches(7.5, 9.75)
    fig.subplots_adjust(
        left=None, bottom=None, right=None, top=None, wspace=0.2, hspace=0.03
    )

    #axes[0, 0].plot(S2 * 100, c="black")
    #axes[0, 0].plot(S2[3] * 100, c="orange")
    #axes[0, 0].plot(S2[2] * 100, c="orange")
    #axes[0, 0].plot(OLDEPTH_PROFILE)
    #axes[0, 0].plot(OLDEPTH_PROFILE-OLTHICK_PROFILE)
    axes[0, 0].fill_between(OLDEPTH_PROFILE.index, OLDEPTH_PROFILE, OLDEPTH_PROFILE-OLTHICK_PROFILE, color='gray')
    axes[0, 0].set_ylim([-1000,500])
    
    
    #axes[0, 1].plot(S3 * 100, c="black")
    #axes[0, 1].plot(S3[3] * 100, c="orange")
    #axes[0, 1].plot(S3[2] * 100, c="orange")
    #axes[0, 1].plot(AUDEPTH_PROFILE)
    #axes[0, 1].plot(AUDEPTH_PROFILE-AUTHICK_PROFILE)
    axes[0, 1].fill_between(AUDEPTH_PROFILE.index, AUDEPTH_PROFILE, AUDEPTH_PROFILE-AUTHICK_PROFILE, color='gray')
    axes[0, 1].set_ylim([-1000,500])

    IM0 = axes[1, 0].imshow(
        OLSGRID.depth,
        extent=[0, 400, 0, 900],
        aspect="auto",
        cmap="viridis",
        vmin=-900,
        vmax=500,
    )
    axes[1,0].scatter(DECIMATE_ONL.iloc[:, -2]*10, DECIMATE_ONL.iloc[:,-1]*10, color='None', edgecolor='gray', label='Well location', s=15)
    axes[1, 0].set_ylim([0,900])
    axes[1, 0].set_xlim([0,400])


    IM1 = axes[1, 1].imshow(
        AUSGRID.depth,
        extent=[0, 400, 0, 900],
        aspect="auto",
        cmap="viridis",
        vmin=-900,
        vmax=500,
    )

    axes[1,1].scatter(DECIMATE_ANG.iloc[:, -2]*10, DECIMATE_ANG.iloc[:,-1]*10, color='None', edgecolor='gray', label='Well location', s=15)
    axes[1, 1].set_ylim([0,900])
    axes[1, 1].set_xlim([0,400])

    IM2 = axes[2, 0].imshow(
        OLIGRID.thickness,
        extent=[0, 400, 0, 900],
        aspect="auto",
        cmap="plasma",
        vmin=0,
        vmax=500,
    )
    axes[2,0].scatter(DECIMATE_ONL.iloc[:, -2]*10, DECIMATE_ONL.iloc[:,-1]*10, color='None', edgecolor='gray', label='Well location', s=15)
    axes[2, 0].set_ylim([0,900])
    axes[2, 0].set_xlim([0,400])

    IM3 = axes[2, 1].imshow(
        AUIGRID.thickness,
        extent=[0, 400, 0, 900],
        aspect="auto",
        cmap="plasma",
        vmin=0,
        vmax=500,
    )
    axes[2,1].scatter(DECIMATE_ANG.iloc[:, -2]*10, DECIMATE_ANG.iloc[:,-1]*10, color='None', edgecolor='gray', label='Well location', s=15)
    axes[2, 1].set_ylim([0,900])
    axes[2, 1].set_xlim([0,400])

    fig.subplots_adjust(left=0.07, right=0.87)
    # Add the colorbar outside...
    box = axes[1, 1].get_position()
    pad, width = 0.005, 0.01
    cax = fig.add_axes([box.xmax + pad, box.ymin, width, box.height])
    fig.colorbar(IM0, cax=cax, label="Depth")


    box = axes[2, 1].get_position()
    pad, width = 0.005, 0.01
    cax = fig.add_axes([box.xmax + pad, box.ymin, width, box.height])
    fig.colorbar(IM3, cax=cax, label="Thickness")

    fig.savefig(str(len(DECIMATE_ANG))+'_wells.jpg')

# -


