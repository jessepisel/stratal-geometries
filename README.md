# Predicting Stratigraphic Geometries from Subsurface Well Logs
![Classification T-SNE embedding](https://github.com/jessepisel/stratal-geometries/blob/master/repo_base.png)
This repository contains jupyter notebooks to create basin-scale stratigraphic geometries from a conceptual geometric model,
train a K-nearest neighbor classifier on the dataset, and perform a grid search to determine optimal parameters for the model.

Notebooks are organized as follows:

* 00_map_patterns.ipynb - Creates ambiguous map patterns from truncated and onlapping stratal geometries
* 01_training_data.ipynb - Creates training data from a geometric model for a varying number of adjacent wells. This dataset can be downloaded from https://osf.io/a6cwh/
* 02_grid_search_predictions.ipynb - Runs a grid search to optimize the number of adjacent wells and KNN classifier hyperparameters for the training datasets
* 03_predictions_tsne.ipynb - Investigates which feature group is the most important for classification, makes predictions on a subsurface dataset available from both https://osf.io/a6cwh/ (`subsurface_data.csv`)and [Here](http://sales.wsgs.wyo.gov/stratigraphy-and-hydrocarbon-potential-of-the-fort-union-and-lance-formations-in-the-great-divide-and-washakie-basins-south-central-wyoming-2016/) (Appendix 1, tab 4). The subsurface predictions are saved to shapefiles
* 04_spatial_prediction_viz.ipynb - Visualizes the spatial distribution of predictions and plots the predictions with isochore maps and outcrop exposures of the formations

Training data can be generated locally, or downloaded from https://osf.io/a6cwh/
Subsurface data is also located at https://osf.io/a6cwh/ or direct from the [Wyoming State Geological Survey](http://sales.wsgs.wyo.gov/stratigraphy-and-hydrocarbon-potential-of-the-fort-union-and-lance-formations-in-the-great-divide-and-washakie-basins-south-central-wyoming-2016/)

To run this repository, we have included a Python virtual environment. Install the virtual environment by changing directories to this repository, then:

    $ conda env create -f strata.yml

After the virtual environment is installed, activate it and start a jupyter notebook or jupyterlab session
    
    $ conda activate strata
    $ jupyter lab
    
    or
    
    $ jupyter notebook
    
When the jupyter server has loaded, you can walk through the notebooks in order to reproduce the results from this study.