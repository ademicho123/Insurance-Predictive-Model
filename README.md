# Insurance-Predictive-Model
This repository contains code for a machine learning project that predicts insurance charges based on age using a linear regression model. The project utilizes the popular scikit-learn library for machine learning in Python.

# Dataset
The dataset used for this project is stored in the "data" directory and is named "insurance.csv". It includes the following columns:

- age: Age of the individual.
- sex: Gender of the individual (0 for male, 1 for female).
- bmi: Body Mass Index (BMI) of the individual.
- children: Number of children/dependents covered by the insurance.
- smoker: Smoking status (0 for non-smoker, 1 for smoker).
- region: Geographic region of the individual.
- charges: Insurance charges.

# Data Preprocessing
- Converting Categorical Features
  - The "smoker" column was converted to a numerical format (0 for non-smoker, 1 for smoker).
  - The "sex" column was converted to a numerical format (0 for male, 1 for female).
- Handling Null Values
  - Any rows with null values in the "smoker" column were dropped.

# Exploratory Data Analysis (EDA)
A scatterplot was created to visualize the relationship between age and charges, with additional information encoded by smoker status, sex, and BMI.

# Model Training
- The model was trained using the RandomForestRegressor from scikit-learn.
- Features used for training: 'sex', 'smoker', 'bmi', and 'age'.
- The dataset was split into training and testing sets using a 80-20 split.

# Results
The model achieved a Mean Absolute Error (MAE) of approximately $9050.63 on the test set, indicating the average absolute difference between the predicted and actual insurance charges.