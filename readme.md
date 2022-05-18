# Convolutional neural networks for wind energy prediction

- In the `ree` folder there are two scripts used to obtain plots of the installed power capacity and generated energy in Spain directly from Red Eléctica's API.
- In the `estimators` folder one can find a custom MLP created from scratch, and two transformers developed to handle non-tabular data within sklearn's Pipelines: `MatrixToTensor` and `TensorStandardScaler`.
- The notebook `data_exploration.ipynb` contains a basic exploratory analysis of the data.
- All the models used, as well as the hyperparameter searches can be found in the `models.py` and `hyperparameters.py` files.
- The file `benchmark.py` runs a comparison between different regression methods. To use Intel's extension, run
  ```bash
  python -m sklearnex benchmark.py
  ```
