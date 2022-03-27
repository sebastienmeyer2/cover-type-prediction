# Cover Type Prediction

Predicting forest cover type from cartographic variables only (no remotely sensed data). The actual forest cover type for a given observation (30 x 30 meter cell) was determined from US Forest Service (USFS) Region 2 Resource Information System (RIS) data. Independent variables were derived from data originally obtained from US Geological Survey (USGS) and USFS data. Data is in raw form (not scaled) and contains binary (0 or 1) columns of data for qualitative independent variables (wilderness areas and soil types).

The data set is available at: https://archive.ics.uci.edu/ml/datasets/covertype.

Please refer to the following sections for more information about the package usage:

1. [Our results](#our-results)
2. [Installation](#installation-instructions)
3. [Description](#package-description)
4. [Usage via command lines](#package-usage)
5. [Documentation](#documentation)

## Our results

A brief summary of our results is available in our report under *report/report.pdf*. Below, we only give a summary table of the test accuracy of different models.

| Model         | Test accuracy | Features                                                                              |
| ------------- | ------------- | ------------------------------------------------------------------------------------- |
| Random Forest | 0.76173       | base + merged binary features                                                        |
| LightGBM      | 0.78374       | base + merged binary features                                                         |
| Extra Trees   | 0.82622       | base + some interaction terms and statistics + shifted features                       |
| Stacking      | 0.83822       | base + some interaction terms and statistics + shifted features                       |
| Extra Trees   | 0.85905       | base + some interaction terms and statistics + shifted features + groups + resampling |

## Installation instructions

In order to use our package and run your own experiments, we advise you to set up a virtual environment. The package has been tested under Python version 3.7.12, you will also need the *virtualenv* package:

    pip3 install virtualenv

Then, you can create a virtual environment and switch to it with the following commands:

    python3 -m venv myvenv
    source myvenv/bin/activate (Linux)
    myvenv/Scripts/Activate.ps1 (Windows PowerShell)

All the needed packages are listed in the *requirements.txt* file, you can install them with:

    pip3 install -r requirements.txt

## Package description

Below, we give a brief tree view of our package.

    .
    ├── data
    |   └── training_ids.txt  # subset of ids for training
    ├── doc  # contains a generated documentation of src/ in html
    ├── report  # contains our complete report in pdf format
    |   └── figures
    ├── src  # source code
    |   ├── engine
    |   |   ├── models
    |   |   |   ├── __init__.py
    |   |   |   └── base.py  # scikit-learn compatible classifiers and manual stacking
    |   |   ├── __init__.py
    |   |   ├── gridsearch.py
    |   |   ├── hub.py  # to prepare data and create models
    |   |   └── training.py 
    |   ├── preprocessing
    |   |   ├── features  # multiple files for each type of features
    |   |   ├── reduction  # multiple files for feature selection
    |   |   ├── __init__.py
    |   |   ├── reader.py  # to read preprocessed files
    |   |   └── resampling.py  # undersampling/oversampling operations
    |   ├── utils 
    |   ├── __init__.py
    |   ├── data_preparation.py  # main file to compute features
    |   └── main.py  # main file to run gridsearch
    ├── README.md
    ├── first_glimpse.ipynb  # first look at the data set
    ├── feature_engineering.ipynb  # large part of our research for valuable features
    ├── model_selection.ipynb  # selection of features and models
    └── requirements.txt  # contains the necessary Python packages to run our files

## Package usage

### Downloading the data

We expect you to download and extract the data set from https://archive.ics.uci.edu/ml/datasets/covertype. Then, put the data file in a *data/* folder, along with the *training_ids.txt* file.

### Notebooks

In order to use the notebooks, you will also need to install jupyter:

    pip3 install jupyter notebook ipykernel
    ipython kernel install --user --name=myvenv

There are three available notebooks:

- *first_glimpse.ipynb*: this notebook studies the train and test dataframes to understand the classification task and look for missing values
- *feature_engineering.ipynb*: this notebook gathers many if not all features that were tested for this data set
- *model_selection.ipynb*: this notebook allows to test different machine learning models and subset of features

### Feature engineering

In order to create your own data set for training, you can use the *src/data_preparation.py* file to create features:

    python3 src/data_preparation.py [options]

A selection of best features and options is already available as default arguments. Here is a description of the available options:

- `--seed`: Seed to use everywhere for reproducbility. Default: 42.

- `--correct-feat`: List of features where a specific value has to be replaced.
- `--value`: Value to replace by predictions. Default: 0.

- `--binary-feat`: List of binary features to merge. The name of a binary feature is expected to be the base and final feature name, for example "Soil_Type".

- `--log-feat`: List of logarithm features. The name of a log feature is expected to be an existing feature.

- `--stat-feat`: List of stat features. The name of a stat feature is expected to be either "Max", "Min", "Mean", "Median" or "Std".

- `--pm-feat`: List of feature to sum and substract. The name of a pm feature is expected to be an existing features.

- `--kd-feat`: List of knowledge-domain features. The name of a knowledge-domain feature is expected to be either "Distance_To_Hydrology", "Mean_Distance_To_Points_Of_Interest", "Elevation_Shifted_Horizontal_Distance_To_Hydrology", "Elevation_Shifted_Vertical_Distance_To_Hydrology" or "Elevation_Shifted_Horizontal_Distance_To_Roadways".

- `--families-feat`: List of families features to compute. The name of a binary feature is expected to be either "Ratake", "Vanet", "Catamount", "Leighan", "Bullwark", "Como", "Moran" or "Other".

- `--soil-feat`: List of soil types features to compute. The name of a binary feature is expected to be either "Stony", "Rubly" or "Other".

- `--fixed-poly-feat`: List of specific polynomial features. A fixed polynomial feature must be of the form "A B C". A, B and C can be features or powers of features, and their product will be computed. Example: --fixed-poly-feat "char_count^2 group_overlap".
- `--poly-feat`: List of polynomial features to compute of which interaction terms will be computed.
- `--all-poly-feat`: Use this option to activate polynomial interaction of all features. Use with caution. Default: Deactivated.
- `--poly-degree`: Define the degree until which products and powers of features are computed. If 1 or less, there will be no polynomial features. Default: 2.

- `--excl-feat`: List of features names to drop after computation. Example: --excl-feat "char_count^2 group_overlap".

- `--max-correlation`: Correlation threshold to select features. Default: 1.0.

- `--rescale-data`: Use this option to activate rescaling the data sets. Default: Activated.
- `--no-rescale-data`: Use this option to deactivate rescaling the data sets. Default: Activated.
- `--scaling-method`: If "standard", features are rescaled with zero mean and unit variance. If "positive", features are rescaled between zero and one. Default: "standard".

- `--pca-ratio`: Variance ratio parameter for the Principal Component Analysis. Default: 1.0.

- `--nb-sampling`: Number of predicted resampling operations. Default: 0.
- `--manual-resampling`: Manual selection of number of samples for each class. Example: "1:5500,2:7000,3:1000,4:50,5:450,6:500,7:650".

- `--save-data`: Use this option to activate saving the data sets. Default: Activated.
- `--no-save-data`: Use this option to deactivate saving the data sets. Default: Activated.
- `--file-suffix`: Suffix to append to the training and test files if **save_data** is True. Default: "final".

### Gridsearch

Then, you can use the *src/main.py* file to try multiple gridsearch and models. The command is as follows:

    python3 src/main.py [options]

- `--seed`: Seed to use everywhere for reproducbility. Default: 42.

- `--models-names`: Choose models names. Available models: "rfc", "etc", "ova", "xgboost", "lightgbm", "catboost", "logreg", "stacking" and "resampling".

- `--data-path`: Path to the directory where the data is stored. Default: "data/".
- `--file-suffix`: Suffix to append to the training and test files. Default: "final".

- `--trials`: Choose the number of gridsearch trials. Default: 25.

- `--trial`: Use this option to activate submitting a file. Default: Activated.
- `--no-trial`: Use this option to deactivate submitting a file. Default: Activated.
- `--metric`: Evaluation metric for parameters gridsearch. Available metrics: "accuracy" and"f1_weighted". Default: "accuracy".

## Documentation

A complete documentation is available in the *doc/src/* folder. If it is not
generated, you can run from the root folder:

    python3 -m pdoc -o doc/ --html --config latex_math=True --force src/

Then, open *doc/src/index.html* in your browser and follow the guide!
