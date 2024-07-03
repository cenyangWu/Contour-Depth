# Paper figures

This folder includes the scripts necessary to generate the figures and tables used in the paper. Below we include an index of the different files.

## Index

- Fig 2 (`inclusion_mat.py`): Computes inclusion matrices for simple ensemble of contours (mock-ensemble). We then use the values in the plots to construct the depiction in the paper.
- Fig 3 (`red_explainer.py`): explainer plot of the different components of the ReD criteria and how it helps identifying outliers that sil might not catch.
- Fig 4 (`fast_depth_computation_benchmark.py`): generates a plot of number of contours vs time and a scatter plot comparing the depth scores, making evident the MSE is 0.
- Fig 5 (`progressive_demo.py`): generates a plot that shows the time that it takes to compute the depths of an ensemble in a progressive manner. The idea is to show the time that the user would need to wait.
- Fig 6 (`num_clust_selection.py`): generates a plot showing how ReD can help determining the optimal number of clusters. It also generates small multiples showing the clusterings obtained with different k's.
- Fig 7 (`clustering_differences.py` and `outlier_separation.py`)
- Fig 8 (`rd_han.py`): generates a plot that illustrates a multi-modal (clustering) depth-based analysis on the head-and-neck segmentations dataset.
- Fig 9 (`rd_meteo.py`): generates a plot that illustrates a multi-modal (clustering) depth-based analysis on the meteorological forecasting dataset.

