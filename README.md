# stratal-geometries
This repository contains jupyter notebooks for creating basin-scale stratigraphic geometries from a conceptual geometric model,
training a K-nearest neighbor classifier on the dataset, and performing a grid search to determine optimal parameters for the model.

Notebooks are organized as follows:

* 00_map_patterns.ipynb - Creates ambiguous map patterns from truncated and onlapping stratal geometries
* 01_training_data.ipynb - Creates training data from a geometric model for a varying number of adjacent wells. This dataset can be downloaded from https://osf.io/a6cwh/
* 02_grid_search_predictions.ipynb - Runs a grid search to optimize the number of adjacent wells and KNN classifier hyperparameters for the training datasets
* 03_predictions_tsne.ipynb - Investigates which feature group is the most important for classification, makes predictions on a subsurface dataset available from both https://osf.io/a6cwh/ (`subsurface_data.csv`)and http://sales.wsgs.wyo.gov/stratigraphy-and-hydrocarbon-potential-of-the-fort-union-and-lance-formations-in-the-great-divide-and-washakie-basins-south-central-wyoming-2016/ (Appendix 1, tab 4). The subsurface predictions are saved to shapefiles
* 04_spatial_prediction_viz.ipynb - Visualizes the spatial distribution of predictions and plots the predictions with isochore maps and outcrop exposures of the formations

To 