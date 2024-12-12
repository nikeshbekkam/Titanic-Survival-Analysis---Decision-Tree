# Titanic-Survival-Analysis---Decision-Tree
This project aims to predict the survival of passengers aboard the Titanic using machine learning techniques. The analysis leverages various classification algorithms, including Decision Trees, Logistic Regression, and Random Forests, to predict whether a passenger survived based on their personal information.

## Table of Contents
Objective
Dataset Overview
Key Features
Techniques Used
Model Implementation
Results
Installation
Usage
Contributing
License
References

## Objective
The main objective of this project is to predict whether a passenger survived or not based on various features such as age, sex, class, and other passenger details. This is a binary classification problem, with the goal of helping improve predictive models for similar types of historical data and providing insights into factors that may have influenced survival.

## Dataset Overview
The Titanic dataset used in this project is publicly available on Kaggle. The dataset contains information about passengers aboard the Titanic, including personal details such as:
PassengerId: An identifier for each passenger
Pclass: The class of the ticket (1st, 2nd, 3rd)
Name: The name of the passenger
Sex: The gender of the passenger
Age: The age of the passenger
SibSp: Number of siblings or spouses aboard
Parch: Number of parents or children aboard
Ticket: Ticket number
Fare: The fare paid for the ticket
Cabin: The cabin number (if available)
Embarked: The port of embarkation (C = Cherbourg; Q = Queenstown; S = Southampton)
Survived: 0 = No, 1 = Yes (the target variable)
Key Features
Sex: Gender of the passenger
Age: Age of the passenger
Pclass: Class of the ticket (1st, 2nd, 3rd)
Fare: Fare paid by the passenger
Embarked: Port of embarkation
SibSp: Number of siblings or spouses aboard
Techniques Used
This project implements various classification models to predict survival:

## Logistic Regression: A statistical model used for binary classification.

Decision Tree Classifier: A tree-based model that splits the data based on feature values.
Random Forest Classifier: An ensemble method that uses multiple decision trees to improve performance.

Additionally, data preprocessing steps like handling missing values, encoding categorical variables, and scaling numerical features are performed.

## Model Implementation
## Data Preprocessing:
Missing values are handled, and categorical variables (like 'Sex' and 'Embarked') are encoded into numerical formats.
## Feature Engineering:
Additional features such as the passenger's family size (sum of 'SibSp' and 'Parch') are created to improve model performance.
## Model Training: 
Various models are trained on the dataset, and hyperparameters are tuned to improve accuracy.
## Evaluation:
Model performance is evaluated using accuracy, confusion matrix, precision, recall, and F1-score.
## Results
Training Accuracy: 100%
Test Accuracy: 100%
## Confusion Matrix:
A 2x2 matrix showing the number of true positives, true negatives, false positives, and false negatives.
## Classification Report:
Precision, recall, and F1-score metrics are calculated for each class.
The models performed excellently on this dataset, with perfect accuracy. However, in a real-world scenario, overfitting should be checked to ensure the model generalizes well.



## License
This project is licensed under the MIT License - see the LICENSE file for details.
