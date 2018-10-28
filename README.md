# This Kaggle code is to predict Gstore product revenues 

##### Links : https://www.kaggle.com/c/ga-customer-revenue-prediction

#### Reference 
- 


# Example of data science project structure 

```sh
├── LICENSE
├── Makefile           <- Makefile with commands like `make data` or `make train`
├── README.md          <- The top-level README for developers using this project.
├── data
│   ├── external       <- Data from third party sources.
│   ├── interim        <- Intermediate data that has been transformed.
│   ├── processed      <- The final, canonical data sets for modeling.
│   └── raw            <- The original, immutable data dump.
│
├── docs               <- A default Sphinx project; see sphinx-doc.org for details
│
├── models             <- Trained and serialized models, model predictions, or model summaries
│
├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
│                         the creator's initials, and a short `-` delimited description, e.g.
│                         `1.0-jqp-initial-data-exploration`.
│
├── references         <- Data dictionaries, manuals, and all other explanatory materials.
│
├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures        <- Generated graphics and figures to be used in reporting
│
├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
│                         generated with `pip freeze > requirements.txt`
│
├── setup.py           <- Make this project pip installable with `pip install -e`
├── src                <- Source code for use in this project.
│   ├── __init__.py    <- Makes src a Python module
│   │
│   ├── data           <- Scripts to download or generate data
│   │   └── make_dataset.py
│   │
│   ├── features       <- Scripts to turn raw data into features for modeling
│   │   └── build_features.py
│   │
│   ├── models         <- Scripts to train models and then use trained models to make
│   │   │                 predictions
│   │   ├── predict_model.py
│   │   └── train_model.py
│   │
│   └── visualization  <- Scripts to create exploratory and results oriented visualizations
│       └── visualize.py
│
└── tox.ini            <- tox file with settings for running tox; see tox.testrun.org
```

# Data Fields
- `fullVisitorId`- A unique identifier for each user of the Google Merchandise Store.
- `channelGrouping` - The channel via which the user came to the Store.
- `date` - The date on which the user visited the Store.
- `device` - The specifications for the device used to access the Store.
- `geoNetwork` - This section contains information about the geography of the user.
- `sessionId` - A unique identifier for this visit to the store.
- `socialEngagementType` - Engagement type, either "Socially Engaged" or "Not Socially Engaged".
- `totals` - This section contains aggregate values across the session.
- `trafficSource` - This section contains information about the Traffic Source from which the session originated.
- `visitId` - An identifier for this session. This is part of the value usually stored as the _utmb cookie. This is only unique to the user. For a completely unique ID, you should use a combination of fullVisitorId and visitId.
- `visitNumber` - The session number for this user. If this is the first session, then this is set to 1.
- `visitStartTime` - The timestamp (expressed as POSIX time).


# Visualization : plots instead of boxplots,
 
 
# Missing values : Ask you this question 

- How prevalent is the missing data?
- Is missing data random or does it have a pattern?

# Check Out liars!


# Test Statistique

1. Normality
When we talk about normality what we mean is that the data should look like a normal distribution. This is important because several statistic tests rely on this (e.g. t-statistics). In this exercise we'll just check univariate normality for 'SalePrice' (which is a limited approach). Remember that univariate normality doesn't ensure multivariate normality (which is what we would like to have), but it helps. Another detail to take into account is that in big samples (>200 observations) normality is not such an issue. However, if we solve normality, we avoid a lot of other problems (e.g. heteroscedacity) so that's the main reason why we are doing this analysis.

- Histogram - Kurtosis and skewness. 
- Normal probability plot - Data distribution should closely follow the diagonal that represents the normal distribution.

2. Homoscedasticity
I just hope I wrote it right. Homoscedasticity refers to the 'assumption that dependent variable(s) exhibit equal levels of variance across the range of predictor variable(s)' (Hair et al., 2013). Homoscedasticity is desirable because we want the error term to be the same across all values of the independent variables.

The best approach to test homoscedasticity for two metric variables is graphically. Departures from an equal dispersion are shown by such shapes as cones (small dispersion at one side of the graph, large dispersion at the opposite side) or diamonds (a large number of points at the center of the distribution).

3. Linearity
The most common way to assess linearity is to examine scatter plots and search for linear patterns. If patterns are not linear, it would be worthwhile to explore data transformations. However, we'll not get into this because most of the scatter plots we've seen appear to have linear relationships.

4. Absence of correlated errors
Correlated errors, like the definition suggests, happen when one error is correlated to another. For instance, if one positive error makes a negative error systematically, it means that there's a relationship between these variables. This occurs often in time series, where some patterns are time related. We'll also not get into this. However, if you detect something, try to add a variable that can explain the effect you're getting. That's the most common solution for correlated errors.


# Use this step again 
Notebook Content
1- Introduction
2- Machine learning workflow
3- Problem Definition
3-1 Problem feature
3-2 Aim
3-3 Variables
4- Inputs & Outputs
4-1 Inputs
4-2 Outputs
5- Installation
5-1 jupyter notebook
5-2 kaggle kernel
5-3 Colab notebook
5-4 install python & packages
5-5 Loading Packages
6- Exploratory data analysis
6-1 Data Collection
6-2 Visualization
6-2-1 Scatter plot
6-2-2 Box
6-2-3 Histogram
6-2-4 Multivariate Plots
6-2-5 Violinplots
6-2-6 Pair plot
6-2-7 Kde plot
6-2-8 Joint plot
6-2-9 Andrews curves
6-2-10 Heatmap
6-2-11 Radviz
6-3 Data Preprocessing
6-4 Data Cleaning
7- Model Deployment
8- Conclusion
9- References


# Step in data science projet 
1. Define Problem
2. Specify Inputs & Outputs
3. Exploratory data analysis
4. Data Collection
5. Data Preprocessing
6. Data Cleaning
7. Visualization
8. Model Design, Training, and Offline Evaluation
9. Model Deployment, Online Evaluation, and Monitoring
10. Model Maintenance, Diagnosis, and Retraining


# Data preprocessing 

Data preprocessing refers to the transformations applied to our data before feeding it to the algorithm.
Data Preprocessing is a technique that is used to convert the raw data into a clean data set. In other words, whenever the data is gathered from different sources it is collected in raw format which is not feasible for the analysis. there are plenty of steps for data preprocessing and we just listed some of them :

1. removing Target column (id)
2. Sampling (without replacement)
3. Making part of iris unbalanced and balancing (with undersampling and SMOTE)
4. Introducing missing values and treating them (replacing by average values)
5. Noise filtering
6. Data discretization
7. Normalization and standardization
8. PCA analysis
9. Feature selection (filter, embedded, wrapper)

# Data cleaning 
These include missing value imputation, outliers detection, transformations, integrity constraints violations detection and repair, consistent query answering, deduplication, and many other related problems such as profiling and constraints mining
