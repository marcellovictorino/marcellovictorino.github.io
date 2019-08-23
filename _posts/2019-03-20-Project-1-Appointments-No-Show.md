---
title: Project 1 - Appointments No-Show
excerpt: Analysis of the Appointments No-Show dataset, based on medical appointments data in Brazil. Particularly, investigating if there are any factors that influence the appointment attendance (independent variable) and evaluating predictive analytics.
tags: 
categories: [Project]
date: 2019-04-19 00:00:00 +0000
thumbnail: /images/Project1-Appointment-No-Show/Doctor-appointment.jpg
---
# Introduction

This project was completed as part of the course requirements of Udacity's [Data Analyst Nanodegree](https://www.udacity.com/course/data-analyst-nanodegree--nd002) certification.

The complete code and analysis rationale can be found in this [browser friendly version](https://github.com/marcellovictorino/DAND-Project_1/blob/master/Project2%20-%20No%20Show%20Appointments%20V2.html) or the actual [jupyter notebook](https://github.com/marcellovictorino/DAND-Project_1/blob/master/Project2%20-%20No%20Show%20Appointments%20V2.ipynb).

## Overview

Analysis of the [Appointments No-Show dataset](https://d17h27t6h515a5.cloudfront.net/topher/2017/October/59dd2e9a_noshowappointments-kagglev2-may-2016/noshowappointments-kagglev2-may-2016.csv),  based on medical appointments data in Brazil. Particularly, investigating if there are any factors that influence the appointment  attendance (independent variable).

This dataset required some **wrangling** to address: missing data, correcting data types, removing outliers, and fixing miss-entries. Since the response class is very unbalanced (4:1 ratio), we used an oversampling technique ([SMOTE](https://arxiv.org/pdf/1106.1813.pdf)) to avoid model overfitting.

After investigating which variables correlate to the attendance rate, and performing some feature engineering to create dummy variables, we conduct a [Recursive Feature Elimination with Cross-Validation](https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.RFECV.html#sklearn.feature_selection.RFECV) to identify the most important explanatory variables.

Finally, the Logistic Regression model is fitted and evaluated. 

## Technologies Used

+ **Python**:
  + Pandas, Numpy, Matplotlib, Statmodels, Scikit-Learn, Jupyter Notebook, imblearn
+ **Model**:
  + Logistic Regression: Odds Ratio, Confusion Matrix
  + Oversampling (Synthetic Minority Oversampling Technique - SMOTE)

## Key Findings

- Gender does not seem to have any impact over the attendance rate;
- Age groups:
  - **Less than 5 years and over 45 years old**: seems to be more likely to show up
  - **Between 5 and 45 years old**: seems to be less likely to show up
- Date difference between appointment and when it was scheduled:
  - **Appointments scheduled up to 4 days in advance**: seems to be more likely to show up
  - **Appointments scheduled more than 7 days in advance**: less likely to attend
- Neighborhood groups:
  - **Group 1** (Jardim Camburi,  Jardim da Penha, Santa Martha): seems to be more likely to show up
  - **Group 2** (Itararé, Jesus de Nazareth, Santos Dumont): seems to be less likely to show up
- Patients participating in the government's assistance program, "*Bolsa Familia*", are less likely to show up
- Medical Conditions: patients suffering from alcoholism and diabetes seems to be slightly more likely to attend
- Even though there is evidence receiving an SMS  reminder seems to make the patients less likely to show up, we decided to ignore this correlation due to lack of explainability

Overall, the Logistic model predicts the right outcome 63% of the time ( (TP + TN) / Total). Which is better than flipping a coin, but still has room for improvement before it can be considered useful.



## Recommendation for Future Research

The advantage of using a Logistic Regression model is it's  explanatory power: it allows for interesting inference capabilities such  as Odd's Ratio and Elasticity analysis. Although it lacks prediction  accuracy power.

For this particular case study, the inference power of the Logit  model is rendered useless, since the doctors cannot control the  characteristics of their patients. Their main goal is to accurately  predict whether or not a patient is going to show up for their scheduled  appointment.

Therefore, it would be interesting to fit the data using more advanced methods, such as Naive Bayes, Boosting, Random Forest, and Artificial Neural Network to see if they would perform better at predicting, using the same set of explanatory variables - or if further feature engineering is required.
