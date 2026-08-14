# Kernel and Non-Kernel Learning on Real-World Data

## Overview

This project investigates the behavior of kernel and non-kernel machine learning methods when applied to real-world datasets with different data quality and structural challenges.

The main focus is not only on predictive performance, but also on understanding how missing values, outliers, class imbalance, feature quality, nonlinear structure, and preprocessing decisions influence the behavior of different learning algorithms.

Four real-world datasets are studied:

- Airbnb Open Data
- NYC 311 Service Requests
- IBM HR Employee Attrition
- Online Retail Transactions

The project combines data engineering, exploratory analysis, feature engineering, machine learning algorithms implemented from scratch, kernel methods, error analysis, and cross-dataset comparison.

---

## Project Objectives

The main objectives of this project are:

- Investigate data quality problems in real-world datasets
- Design dataset-specific preprocessing pipelines
- Analyze missing values and outliers
- Detect redundant, near-constant, and potentially leaking features
- Engineer meaningful features for each prediction task
- Study class imbalance and sampling strategies
- Implement classical machine learning algorithms from scratch
- Implement kernel-based learning methods
- Compare kernel and non-kernel approaches
- Analyze model failures instead of relying only on aggregate metrics
- Study computational cost and scalability
- Investigate bias and real-world reliability
- Compare learning behavior across different dataset structures

---

# Datasets

## 1. Airbnb Open Data

The Airbnb dataset is used for both regression and classification.

### Prediction Tasks

- **Regression:** Predict listing price
- **Classification:** Predict superhost status

### Main Data Challenges

The Airbnb data contains several characteristics commonly found in real-world marketplace datasets:

- Missing values
- Numerical and categorical variables
- Potentially irrelevant features
- Price outliers
- Geographic information
- Host-related attributes
- Possible redundant or correlated variables

The preprocessing pipeline investigates missing-value strategies, feature selection, outlier handling, and transformations required before model training.

---

## 2. NYC 311 Service Requests

The NYC 311 dataset contains public service requests submitted by residents.

### Prediction Tasks

- **Classification:** Predict complaint category
- **Regression:** Predict resolution time

### Main Data Challenges

This dataset introduces several practical problems:

- Large number of observations
- Duplicate or repeated reports
- Temporal information
- Geographic variables
- Multiple complaint categories
- Potentially inconsistent records
- Sampling challenges

Temporal features are extracted from timestamps, and sampling is performed carefully to preserve the structure of the original data.

---

## 3. IBM HR Employee Attrition

The IBM HR dataset is used to investigate employee attrition.

### Prediction Task

- **Classification:** Predict whether an employee leaves the company

### Main Data Challenges

The main challenges include:

- Class imbalance
- Categorical and numerical features
- Redundant variables
- Near-constant features
- Feature-selection requirements
- Potential relationships between employee characteristics and attrition

Sampling strategies are considered to reduce the effect of class imbalance.

---

## 4. Online Retail Transactions

The Online Retail dataset contains transactional customer purchase records.

### Prediction Tasks

- **Classification:** Customer segmentation
- **Regression:** Future spending prediction

### Main Data Challenges

Unlike the other datasets, the original observations are individual transactions rather than customer-level samples.

Therefore, raw transaction logs must first be transformed into meaningful customer-level features.

This requires aggregation and behavioral feature engineering before machine learning models can be applied.

---

# Data Quality Investigation

A major part of this project focuses on understanding the data before training predictive models.

The investigation includes four main areas.

## Missing Data Analysis

Missing values are examined to determine whether they are consistent with:

- MCAR — Missing Completely At Random
- MAR — Missing At Random
- MNAR — Missing Not At Random

Different imputation strategies are considered depending on the characteristics of each feature and dataset.

The objective is to avoid applying a single universal missing-value strategy to all variables.

---

## Outlier Analysis

Several approaches are considered for identifying unusual observations:

- Z-score
- Interquartile Range (IQR)
- Isolation Forest

Their behavior is compared because different datasets have different distributions and types of anomalies.

An observation detected as an outlier is not automatically removed. Its meaning and possible effect on the prediction task are also considered.

---

## Feature Quality Analysis

Features are investigated for problems such as:

- Redundancy
- Strong correlation
- Near-zero variance
- Data leakage
- Irrelevant information
- High-cardinality variables

Feature selection and engineering decisions are made separately for each dataset.

---

## Label Quality Analysis

Target variables are also investigated rather than assumed to be perfectly reliable.

Potential issues include:

- Ambiguous labels
- Historical collection bias
- Measurement errors
- Inconsistent categories
- Sampling bias

These issues are considered when interpreting model performance.

---

# Feature Engineering

Because the four datasets represent different types of real-world data, each requires a different feature-engineering strategy.

## Airbnb

Feature engineering focuses on variables related to:

- Listing characteristics
- Host information
- Geographic information
- Pricing
- Availability
- Property characteristics

## NYC 311

Temporal and operational information is transformed into model-ready features.

Examples include information derived from:

- Request timestamps
- Resolution timestamps
- Complaint characteristics
- Geographic information

## IBM HR

The HR analysis focuses on identifying variables associated with employee attrition and removing features that provide little useful information.

Special attention is given to the imbalance between attrition and non-attrition classes.

## Online Retail

Transaction-level observations are aggregated into customer-level behavioral representations.

The resulting features summarize customer purchasing behavior and make classification and regression possible at the customer level.

---

# Machine Learning Models

The project compares non-kernel and kernel-based learning approaches.

Core algorithms are implemented from scratch using NumPy and SciPy rather than relying on prebuilt implementations for the main models.

---

# Non-Kernel Methods

## Logistic Regression

Logistic Regression is implemented for binary classification tasks.

The model estimates the probability of class membership using the logistic function.

It provides a linear decision boundary and serves as an important baseline for comparison with nonlinear kernel methods.

---

## Linear Regression

Linear Regression is used as a baseline for continuous prediction tasks.

The model estimates a linear relationship between the input variables and the target.

Its performance helps determine whether nonlinear modeling is necessary for a particular dataset.

---

## K-Nearest Neighbors

KNN predicts observations according to nearby training samples in feature space.

Because it is distance-based, its performance is strongly influenced by:

- Feature scaling
- Dimensionality
- Noise
- Irrelevant features
- Local data density

These properties make preprocessing particularly important.

---

## Decision Tree

A Decision Tree is implemented as a nonlinear non-kernel baseline.

Unlike linear models, decision trees can capture nonlinear relationships and feature interactions without explicitly transforming the feature space.

---

# Kernel Methods

Kernel methods allow algorithms to represent nonlinear relationships by computing similarities between observations.

The project investigates several kernel-based approaches.

---

## Kernel Support Vector Machine

Kernel SVM is used for nonlinear classification.

Kernel functions allow the classifier to construct nonlinear decision boundaries without explicitly calculating the coordinates of a high-dimensional transformed feature space.

---

## Kernel Ridge Regression

Kernel Ridge Regression combines ridge regression with kernel-based nonlinear representation.

It is used for regression problems where a linear relationship between the original input features and target may not be sufficient.

---

## Kernel K-Nearest Neighbors

Kernel KNN uses kernel-derived similarities rather than relying only on standard geometric distance.

This allows neighborhood relationships to be investigated in a transformed similarity space.

---

## Kernel PCA

Kernel Principal Component Analysis is used for nonlinear dimensionality reduction.

The transformed representation can then be passed to a downstream classifier.

This provides a way to study whether nonlinear feature extraction improves classification performance.

---

# Kernel Investigation

The project does not evaluate kernels only according to the final metric.

Instead, the analysis investigates why a kernel works well or poorly for a specific dataset.

Important considerations include:

- Local nonlinear structure
- Feature interactions
- Geographic relationships
- Temporal relationships
- High-dimensional representations
- Feature scaling
- Noise
- Class overlap

The purpose is to connect empirical results with the geometry and characteristics of each dataset.

---

# Experimental Pipeline

The general experimental workflow is:

1. Load the raw dataset
2. Inspect its structure
3. Analyze missing values
4. Investigate outliers
5. Detect problematic features
6. Clean the data
7. Engineer dataset-specific features
8. Encode categorical variables when necessary
9. Scale numerical features when required
10. Handle class imbalance where applicable
11. Construct training and testing sets
12. Train non-kernel models
13. Train kernel models
14. Evaluate predictive performance
15. Compare kernel configurations
16. Identify poor predictions
17. Perform failure analysis
18. Analyze computational requirements
19. Compare results across datasets

---

# Evaluation

Classification and regression tasks require different evaluation criteria.

## Classification

Classification experiments can be evaluated using metrics such as:

- Accuracy
- Precision
- Recall
- F1-score
- Confusion matrix

For imbalanced datasets, accuracy alone may provide a misleading picture of model quality.

## Regression

Regression experiments can be evaluated using:

- RMSE
- MAE
- R²

These metrics provide different views of prediction error and model fit.

---

# Failure Analysis

Aggregate metrics alone cannot explain why a model succeeds or fails.

For each dataset, the worst predictions are investigated individually.

The analysis attempts to determine whether each failure is mainly caused by:

- Poor data quality
- Outliers
- Missing information
- Label ambiguity
- Insufficient feature representation
- Class overlap
- Model assumptions
- Kernel choice
- Sampling decisions

The objective is to distinguish failures caused by limitations of the algorithm from failures caused by limitations of the data.

---

# Bias and Fairness

Real-world datasets can contain systematic biases that affect model reliability.

The analysis considers:

- Sampling bias
- Geographic bias
- Temporal bias
- Historical bias
- Representation imbalance
- Distribution shift

These issues are important because high predictive performance on a test set does not necessarily guarantee reliable behavior in real-world deployment.

---

# Computational Analysis

Kernel methods can become computationally expensive as the number of observations increases.

The project considers both training time and memory usage when comparing algorithms.

Kernel matrices may require approximately quadratic memory with respect to the number of samples.

Some kernel operations can also involve expensive matrix decomposition or inversion.

This becomes particularly important for large datasets such as NYC 311.

The computational analysis therefore considers the trade-off between:

- Predictive performance
- Training time
- Memory requirements
- Dataset size
- Kernel complexity

---

# Cross-Dataset Comparison

One of the main goals of the project is to compare machine learning behavior across fundamentally different real-world datasets.

The final analysis investigates questions such as:

- Which dataset was the most difficult to clean?
- Which dataset required the most feature engineering?
- Which dataset was most affected by missing values?
- Which dataset suffered most from class imbalance?
- Which dataset benefited most from nonlinear kernels?
- Which dataset benefited least from kernel transformations?
- When were simple models competitive with kernel methods?
- How did dataset size affect computational feasibility?

This comparison emphasizes that model selection cannot be separated from the characteristics of the underlying data.

---

# Theory vs. Real-World Data

Clean benchmark datasets often hide many of the difficulties encountered in practical machine learning.

Real-world datasets may contain:

- Missing observations
- Incorrect records
- Outliers
- Duplicate samples
- Biased labels
- Imbalanced classes
- Temporal drift
- Geographic bias
- Irrelevant features
- Leakage
- Large computational requirements

For this reason, successful machine learning depends not only on selecting an algorithm but also on understanding how the data was generated and how preprocessing decisions affect the final model.

---

# Project Structure

```text
kernel-and-nonkernel-learning/
|
├── README.md
├── Airbnb.ipynb
├── NYC311.ipynb
├── IBM_HR.ipynb
├── Online_Retail.ipynb
└── report/
    └── Homework3_Report.pdf
```

The exact notebook names can be adjusted to match the files in the repository.

---

# Notebook Description

| Notebook | Description |
| --- | --- |
| `Airbnb.ipynb` | Data analysis, preprocessing, regression, and classification experiments on Airbnb data |
| `NYC311.ipynb` | Sampling, temporal feature engineering, classification, and resolution-time regression |
| `IBM_HR.ipynb` | Employee attrition analysis, feature selection, imbalance handling, and classification |
| `Online_Retail.ipynb` | Transaction aggregation, customer-level feature engineering, segmentation, and spending prediction |

---

# Technologies and Methods

- Python
- NumPy
- SciPy
- Pandas
- Matplotlib
- Jupyter Notebook
- Data Cleaning
- Feature Engineering
- Missing Data Analysis
- Outlier Detection
- Class Imbalance Handling
- Logistic Regression
- Linear Regression
- K-Nearest Neighbors
- Decision Trees
- Support Vector Machines
- Kernel Ridge Regression
- Kernel KNN
- Kernel PCA
- Kernel Methods
- Classification
- Regression
- Failure Analysis
- Computational Analysis

---

# Key Concepts Demonstrated

This project demonstrates several practical and theoretical machine learning concepts:

- Learning from imperfect real-world data
- Dataset-specific preprocessing
- Missing-data investigation
- Outlier analysis
- Feature quality assessment
- Label quality analysis
- Feature engineering
- Class imbalance
- Linear and nonlinear learning
- Distance-based learning
- Kernel methods
- Kernel trick
- Nonlinear dimensionality reduction
- Model comparison
- Error analysis
- Bias investigation
- Computational complexity
- Theory versus practical machine learning

---

# Course Information

**Course:** Graduate Machine Learning  
**Semester:** Spring 2026  
**University:** Shiraz University

---

# Author

Saghar Kheradmand
