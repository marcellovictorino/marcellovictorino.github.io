---
title: Regression Basics - Linear Model
excerpt: Basic overview of linear regression, covering simple linear model and multiple linear model, working with quantitative and categorical data. More specifically, how to model and interpret the results. It also covers Flexible linear models, when to implement them (higher order and/or interaction terms) and the trade-off between flexibility (prediction power) and interpretability. We briefly mention some of the potential modeling problems regarding the basic assumptions of Linear Models, covering some strategies on identifying and addressing multicollinearity.
tags: [Data Analysis]
categories: [Post]
date: 2019-04-04T09:38:23.535Z
toc: true
classes: wide
---

# Introduction
Linear Regression is the linear approach to modeling the relationship between a quantitative response ($y$) and one or more explanatory variables ($X$); also known as Response and Features, respectively.

This notebook covers the cases of **Simple** and **Multiple Linear Regression**.

Let's load the data and necessary libraries. The toy dataset used in this example contain information on house prices and its characteristics, such as Neighborhood, square footage, # of bedrooms and bathrooms.

```python
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline

df = pd.read_csv('house_prices.csv')
df.drop(columns='house_id', inplace=True)
df.head(3)
```

<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>neighborhood</th>
      <th>area</th>
      <th>bedrooms</th>
      <th>bathrooms</th>
      <th>style</th>
      <th>price</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>B</td>
      <td>1188</td>
      <td>3</td>
      <td>2</td>
      <td>ranch</td>
      <td>598291</td>
    </tr>
    <tr>
      <th>1</th>
      <td>B</td>
      <td>3512</td>
      <td>5</td>
      <td>3</td>
      <td>victorian</td>
      <td>1744259</td>
    </tr>
    <tr>
      <th>2</th>
      <td>B</td>
      <td>1134</td>
      <td>3</td>
      <td>2</td>
      <td>ranch</td>
      <td>571669</td>
    </tr>
  </tbody>
</table>
</div>

## Correlation Coefficient, $R$
Measure of strength and direction of linear relationship between a pair of variables. Also know as Pearson's correlation coefficient.

+ Values varies between [-1, 1], representing negative and positive relationship
+ Strength:
    - $0 \leq R < 0.3$: Weak correlation
    - $0.3 \leq R < 0.7$: Moderate correlation
    - $R \geq 0.7$: Strong correlation

```python
df.corr().style.background_gradient(cmap='Wistia')
```

<style  type="text/css" >
    #T_705afc4a_5625_11e9_a06a_305a3ae4f5fbrow0_col0 {
            background-color:  #fc7f00;
        }    #T_705afc4a_5625_11e9_a06a_305a3ae4f5fbrow0_col1 {
            background-color:  #ffb000;
        }    #T_705afc4a_5625_11e9_a06a_305a3ae4f5fbrow0_col2 {
            background-color:  #ffb300;
        }    #T_705afc4a_5625_11e9_a06a_305a3ae4f5fbrow0_col3 {
            background-color:  #ffda12;
        }    #T_705afc4a_5625_11e9_a06a_305a3ae4f5fbrow1_col0 {
            background-color:  #ffc706;
        }    #T_705afc4a_5625_11e9_a06a_305a3ae4f5fbrow1_col1 {
            background-color:  #fc7f00;
        }    #T_705afc4a_5625_11e9_a06a_305a3ae4f5fbrow1_col2 {
            background-color:  #fd8c00;
        }    #T_705afc4a_5625_11e9_a06a_305a3ae4f5fbrow1_col3 {
            background-color:  #e7fc6f;
        }    #T_705afc4a_5625_11e9_a06a_305a3ae4f5fbrow2_col0 {
            background-color:  #ffd10c;
        }    #T_705afc4a_5625_11e9_a06a_305a3ae4f5fbrow2_col1 {
            background-color:  #fd8d00;
        }    #T_705afc4a_5625_11e9_a06a_305a3ae4f5fbrow2_col2 {
            background-color:  #fc7f00;
        }    #T_705afc4a_5625_11e9_a06a_305a3ae4f5fbrow2_col3 {
            background-color:  #e4ff7a;
        }    #T_705afc4a_5625_11e9_a06a_305a3ae4f5fbrow3_col0 {
            background-color:  #e4ff7a;
        }    #T_705afc4a_5625_11e9_a06a_305a3ae4f5fbrow3_col1 {
            background-color:  #e4ff7a;
        }    #T_705afc4a_5625_11e9_a06a_305a3ae4f5fbrow3_col2 {
            background-color:  #e4ff7a;
        }    #T_705afc4a_5625_11e9_a06a_305a3ae4f5fbrow3_col3 {
            background-color:  #fc7f00;
        }</style>  
<table id="T_705afc4a_5625_11e9_a06a_305a3ae4f5fb" > 
<thead>    <tr> 
        <th class="blank level0" ></th> 
        <th class="col_heading level0 col0" >area</th> 
        <th class="col_heading level0 col1" >bedrooms</th> 
        <th class="col_heading level0 col2" >bathrooms</th> 
        <th class="col_heading level0 col3" >price</th> 
    </tr></thead> 
<tbody>    <tr> 
        <th id="T_705afc4a_5625_11e9_a06a_305a3ae4f5fblevel0_row0" class="row_heading level0 row0" >area</th> 
        <td id="T_705afc4a_5625_11e9_a06a_305a3ae4f5fbrow0_col0" class="data row0 col0" >1</td> 
        <td id="T_705afc4a_5625_11e9_a06a_305a3ae4f5fbrow0_col1" class="data row0 col1" >0.901623</td> 
        <td id="T_705afc4a_5625_11e9_a06a_305a3ae4f5fbrow0_col2" class="data row0 col2" >0.891481</td> 
        <td id="T_705afc4a_5625_11e9_a06a_305a3ae4f5fbrow0_col3" class="data row0 col3" >0.823454</td> 
    </tr>    <tr> 
        <th id="T_705afc4a_5625_11e9_a06a_305a3ae4f5fblevel0_row1" class="row_heading level0 row1" >bedrooms</th> 
        <td id="T_705afc4a_5625_11e9_a06a_305a3ae4f5fbrow1_col0" class="data row1 col0" >0.901623</td> 
        <td id="T_705afc4a_5625_11e9_a06a_305a3ae4f5fbrow1_col1" class="data row1 col1" >1</td> 
        <td id="T_705afc4a_5625_11e9_a06a_305a3ae4f5fbrow1_col2" class="data row1 col2" >0.972768</td> 
        <td id="T_705afc4a_5625_11e9_a06a_305a3ae4f5fbrow1_col3" class="data row1 col3" >0.743435</td> 
    </tr>    <tr> 
        <th id="T_705afc4a_5625_11e9_a06a_305a3ae4f5fblevel0_row2" class="row_heading level0 row2" >bathrooms</th> 
        <td id="T_705afc4a_5625_11e9_a06a_305a3ae4f5fbrow2_col0" class="data row2 col0" >0.891481</td> 
        <td id="T_705afc4a_5625_11e9_a06a_305a3ae4f5fbrow2_col1" class="data row2 col1" >0.972768</td> 
        <td id="T_705afc4a_5625_11e9_a06a_305a3ae4f5fbrow2_col2" class="data row2 col2" >1</td> 
        <td id="T_705afc4a_5625_11e9_a06a_305a3ae4f5fbrow2_col3" class="data row2 col3" >0.735851</td> 
    </tr>    <tr> 
        <th id="T_705afc4a_5625_11e9_a06a_305a3ae4f5fblevel0_row3" class="row_heading level0 row3" >price</th> 
        <td id="T_705afc4a_5625_11e9_a06a_305a3ae4f5fbrow3_col0" class="data row3 col0" >0.823454</td> 
        <td id="T_705afc4a_5625_11e9_a06a_305a3ae4f5fbrow3_col1" class="data row3 col1" >0.743435</td> 
        <td id="T_705afc4a_5625_11e9_a06a_305a3ae4f5fbrow3_col2" class="data row3 col2" >0.735851</td> 
        <td id="T_705afc4a_5625_11e9_a06a_305a3ae4f5fbrow3_col3" class="data row3 col3" >1</td> 
    </tr></tbody> 
</table> 

A scatter plot between two variables is a good way to visually inspect correlation:

```python
sns.pairplot(data=df,y_vars='area', x_vars=['bedrooms', 'bathrooms','price']);
```
{%include figure image_path=''
caption=''
alt=''%}
![png](output_6_0.png)


## Least Squares Algorithm
The main algorithm used to find the line that best fit the data. It minimizes the sum of squared vertical distance between the fitted line and the actual points. 
$$\sum^n_{i=1}(y_i - \hat y_i)^2$$

Basically, it tries to minimize the error.

![Linear%20Regression.jpg](attachment:Linear%20Regression.jpg)

A **Simple Linear Regression** Model can be written as:
$$\hat y = b_0 + b_1.x_1 $$

Where,

+ $\hat y$: predicted response
+ $b_0$: intercept. Height of the fitted line when $x=0$. That is, no influence from explanatory variables
+ $b_1$: coefficient. The slope of the fitted line. Represents the **weight** of the explanatory variable

The Statsmodels package is a good way to obtain linear models, providing pretty and informative results, such as intercept, coefficients, p-value, and R-squared.
>**Note 1:** Statsmodels requires us to manually create an intercept = 1 so the algorithm can use it. Otherwise, the intercept is fixed at 0.


```python
import statsmodels.api as sm
df['intercept'] = 1
# Note how Statsmodels uses the order (y, X)
model = sm.OLS(df.price, df[['intercept', 'area']])
results = model.fit()
results.summary2()
```




<table class="simpletable">
<tr>
        <td>Model:</td>               <td>OLS</td>         <td>Adj. R-squared:</td>      <td>0.678</td>   
</tr>
<tr>
  <td>Dependent Variable:</td>       <td>price</td>             <td>AIC:</td>         <td>169038.2643</td>
</tr>
<tr>
         <td>Date:</td>        <td>2019-04-03 16:30</td>        <td>BIC:</td>         <td>169051.6726</td>
</tr>
<tr>
   <td>No. Observations:</td>        <td>6028</td>         <td>Log-Likelihood:</td>     <td>-84517.</td>  
</tr>
<tr>
       <td>Df Model:</td>              <td>1</td>           <td>F-statistic:</td>      <td>1.269e+04</td> 
</tr>
<tr>
     <td>Df Residuals:</td>          <td>6026</td>       <td>Prob (F-statistic):</td>    <td>0.00</td>    
</tr>
<tr>
      <td>R-squared:</td>            <td>0.678</td>            <td>Scale:</td>        <td>8.8297e+10</td> 
</tr>
</table>
<table class="simpletable">
<tr>
      <td></td>        <th>Coef.</th>   <th>Std.Err.</th>      <th>t</th>     <th>P>|t|</th>   <th>[0.025</th>     <th>0.975]</th>  
</tr>
<tr>
  <th>intercept</th> <td>9587.8878</td> <td>7637.4788</td>  <td>1.2554</td>  <td>0.2094</td> <td>-5384.3028</td> <td>24560.0784</td>
</tr>
<tr>
  <th>area</th>      <td>348.4664</td>   <td>3.0930</td>   <td>112.6619</td> <td>0.0000</td>  <td>342.4029</td>   <td>354.5298</td> 
</tr>
</table>
<table class="simpletable">
<tr>
     <td>Omnibus:</td>    <td>368.609</td>  <td>Durbin-Watson:</td>    <td>2.007</td> 
</tr>
<tr>
  <td>Prob(Omnibus):</td>  <td>0.000</td>  <td>Jarque-Bera (JB):</td> <td>349.279</td>
</tr>
<tr>
       <td>Skew:</td>      <td>0.534</td>      <td>Prob(JB):</td>      <td>0.000</td> 
</tr>
<tr>
     <td>Kurtosis:</td>    <td>2.499</td>   <td>Condition No.:</td>    <td>4928</td>  
</tr>
</table>



## Interpretation
In this simple example, we are tying to predict house prices as a function of their area (in square foot).

+ **intercept**: it means the starting point for every house is 9,588 price unit


+ **coefficient**: being a quantitative variable, it means that for every 1 area unit increase, we expect 345.5 increase in the price unit, on top of the intercept.


+ **p-value**: represents if a specific variable is significant in the model. If p-value > $\alpha$, then we can discard that variable from the model without significantly impacting its predictive power.

    It always test the following Hypothesis:
$$H_0: \beta =0 $$
$$H_1: \beta \neq 0$$


+ **R-squared**: it is a metric of model performance. Represents the amount of observations that can be explained by the model. In this case, 0.678 or 67.8%.
    - It is calculated as the square of the Correlation Coefficient, hence its value varies between [0, 1].

> **Note:** **R-squared** is a poor metric. Better and more robust forms of model evaluation are covered in a later section, such as accuracy, precision, and recall.

# Multiple Linear Regression
But it is also possible to incorporate more than just one explanatory variable in our linear model:

$$\hat y = b_0 + b_1.x_1 + b_2.x_2 + ... + b_n.x_n$$

Let's fit a linear model using all quantitative variables available:


```python
df['intercept'] = 1
# Note how Statsmodels uses the order (y, X)
model = sm.OLS(df.price, df[['intercept', 'area','bedrooms','bathrooms']])
results = model.fit()
results.summary2()
```




<table class="simpletable">
<tr>
        <td>Model:</td>               <td>OLS</td>         <td>Adj. R-squared:</td>      <td>0.678</td>   
</tr>
<tr>
  <td>Dependent Variable:</td>       <td>price</td>             <td>AIC:</td>         <td>169041.9009</td>
</tr>
<tr>
         <td>Date:</td>        <td>2019-04-03 16:30</td>        <td>BIC:</td>         <td>169068.7176</td>
</tr>
<tr>
   <td>No. Observations:</td>        <td>6028</td>         <td>Log-Likelihood:</td>     <td>-84517.</td>  
</tr>
<tr>
       <td>Df Model:</td>              <td>3</td>           <td>F-statistic:</td>        <td>4230.</td>   
</tr>
<tr>
     <td>Df Residuals:</td>          <td>6024</td>       <td>Prob (F-statistic):</td>    <td>0.00</td>    
</tr>
<tr>
      <td>R-squared:</td>            <td>0.678</td>            <td>Scale:</td>        <td>8.8321e+10</td> 
</tr>
</table>
<table class="simpletable">
<tr>
      <td></td>         <th>Coef.</th>    <th>Std.Err.</th>     <th>t</th>     <th>P>|t|</th>   <th>[0.025</th>      <th>0.975]</th>  
</tr>
<tr>
  <th>intercept</th> <td>10072.1070</td> <td>10361.2322</td> <td>0.9721</td>  <td>0.3310</td> <td>-10239.6160</td> <td>30383.8301</td>
</tr>
<tr>
  <th>area</th>       <td>345.9110</td>    <td>7.2272</td>   <td>47.8627</td> <td>0.0000</td>  <td>331.7432</td>    <td>360.0788</td> 
</tr>
<tr>
  <th>bedrooms</th>  <td>-2925.8063</td> <td>10255.2414</td> <td>-0.2853</td> <td>0.7754</td> <td>-23029.7495</td> <td>17178.1369</td>
</tr>
<tr>
  <th>bathrooms</th>  <td>7345.3917</td> <td>14268.9227</td> <td>0.5148</td>  <td>0.6067</td> <td>-20626.8031</td> <td>35317.5865</td>
</tr>
</table>
<table class="simpletable">
<tr>
     <td>Omnibus:</td>    <td>367.658</td>  <td>Durbin-Watson:</td>    <td>2.007</td> 
</tr>
<tr>
  <td>Prob(Omnibus):</td>  <td>0.000</td>  <td>Jarque-Bera (JB):</td> <td>350.116</td>
</tr>
<tr>
       <td>Skew:</td>      <td>0.536</td>      <td>Prob(JB):</td>      <td>0.000</td> 
</tr>
<tr>
     <td>Kurtosis:</td>    <td>2.503</td>   <td>Condition No.:</td>    <td>11643</td> 
</tr>
</table>



## Interpretation
Since the p-value of bedrooms and bathrooms is greater than $\alpha$ (0.05), it means these variables are not significant in the model. Which explains why the R-squared didn't improve after adding more features.

As a matter of fact, it is all about **multicollinearity**: as it happens, there is an <u>intrinsic correlation between these explanatory variables</u>.

>You would expect a house with more bedrooms and bathrooms to have a higher square footage!

It is also the reason behind the unexpected flipped sign for the bedrooms coefficient. 

>Just as well, you would expect a house with more bedrooms to become more and more expensive; not cheaper!

This and other **potential modeling problems** will be covered later with more details. But basically, in situations like this, we would remove the correlated features from our model, retaining only the the most important one. 

The *most important one* can be understood as:
+ A specific variable we have particular interest in understanding/capturing
+ A more granular variable, allowing for a better representation of the characteristics our model is trying to capture
+ A variable easier to obtain

# Dealing with Multicollinearity
We want the explanatory variables to be correlated to the response, not between each other.

In case of correlated features, the model can yield:
+ Inverted coefficient sign from what we would expect
+ Unreliable p-value for the Hypothesis Test of variable being significantly $\neq$ 0

To investigate and identify multicollinearity between explanatory variables, we can use two approaches:
+ **Visual:** Scatter plot between two variables and look for correlation (positive or negative)

+ **Metric:**
    - **Correlation Coefficient**, $R$
    - **Variance Inflation Factor (VIF)**: The general rule of thumb is that VIFs exceeding 4 warrant further investigation, while VIFs exceeding 10 are signs of serious multicollinearity requiring correction


```python
# Visually investigating correlation
g = sns.pairplot(data=df,y_vars='area', x_vars=['bedrooms', 'bathrooms','price'],
            kind='reg')

g.fig.suptitle('Bivariate Plot: correlation between variables', y=1.1);
```


![png](output_17_0.png)



```python
# Metric investigation of correlation: Pearson R

# Compute the correlation matrix of the dataset
corr = df[['area','bedrooms','bathrooms', 'price']].corr()

# Generate mask to hide the upper triangle/diagonal
mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True

# plotting
fig, ax = plt.subplots(figsize=(8,5))

# Generate a custom diverging colormap
cmap = sns.diverging_palette(10, 220, as_cmap=True)

# Draw heatmap with mask
sns.heatmap(corr, mask=mask, cmap=cmap, vmin=-1,vmax=1, center=0,
            square=True, annot=np.where(abs(corr)>0.1, corr,0),
            linewidths=0.5, cbar_kws={"shrink": 0.75})

plt.title("Pearson Correlation Matrix Heatmap");
```


![png](output_18_0.png)


According to the Pearson Correlation coefficient, the variable **area** is strongly correlated to **bedrooms** (0.9) and **bathrooms** (0.89).

Let's investigate the VIF:


```python
# Metric investigation of correlation: VIF
from patsy import dmatrices
from statsmodels.stats.outliers_influence import variance_inflation_factor

model_formula = 'price ~ area + bedrooms + bathrooms'
y,X = dmatrices(model_formula, df, return_type='dataframe')
vif = pd.DataFrame()
vif['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])] # over all columns
vif['feature'] = X.columns
vif
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>VIF</th>
      <th>feature</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>7.327102</td>
      <td>Intercept</td>
    </tr>
    <tr>
      <th>1</th>
      <td>5.458190</td>
      <td>area</td>
    </tr>
    <tr>
      <th>2</th>
      <td>20.854484</td>
      <td>bedrooms</td>
    </tr>
    <tr>
      <th>3</th>
      <td>19.006851</td>
      <td>bathrooms</td>
    </tr>
  </tbody>
</table>
</div>



Looks like there is a serious sign of multicollinearity between the explanatory variables. Let's remove the highest one, **bedrooms** and calculate the VIF again:


```python
# Removing correlated features
model_formula = 'price ~ area + bathrooms'
y,X = dmatrices(model_formula, df, return_type='dataframe')
vif = pd.DataFrame()
vif['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])] # over all columns
vif['feature'] = X.columns
vif
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>VIF</th>
      <th>feature</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>4.438137</td>
      <td>Intercept</td>
    </tr>
    <tr>
      <th>1</th>
      <td>4.871816</td>
      <td>area</td>
    </tr>
    <tr>
      <th>2</th>
      <td>4.871816</td>
      <td>bathrooms</td>
    </tr>
  </tbody>
</table>
</div>



Removing **bedrooms** has improved the multicollinearity condition, but we still have **bathrooms** with a value greater than 4 - which requires further investigation.

Since our previous analysis using the Pearson correlation coefficient showed **bathroom** to be strongly correlated to **area**, let's remove it and calculate the VIF one more time:


```python
# Removing correlated features
model_formula = 'price ~ area'
y,X = dmatrices(model_formula, df, return_type='dataframe')
vif = pd.DataFrame()
vif['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])] # over all columns
vif['feature'] = X.columns
vif
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>VIF</th>
      <th>feature</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>3.98224</td>
      <td>Intercept</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1.00000</td>
      <td>area</td>
    </tr>
  </tbody>
</table>
</div>



As expected, since we are only using one explanatory variable, the VIF reaches it's lowest possible value of 1.

Therefore, we should regress our model using only the variable **area**:


```python
df['intercept'] = 1
# Note how Statsmodels uses the order (y, X)
model = sm.OLS(df.price, df[['intercept', 'area']])
results = model.fit()
results.summary2()
```




<table class="simpletable">
<tr>
        <td>Model:</td>               <td>OLS</td>         <td>Adj. R-squared:</td>      <td>0.678</td>   
</tr>
<tr>
  <td>Dependent Variable:</td>       <td>price</td>             <td>AIC:</td>         <td>169038.2643</td>
</tr>
<tr>
         <td>Date:</td>        <td>2019-04-03 16:30</td>        <td>BIC:</td>         <td>169051.6726</td>
</tr>
<tr>
   <td>No. Observations:</td>        <td>6028</td>         <td>Log-Likelihood:</td>     <td>-84517.</td>  
</tr>
<tr>
       <td>Df Model:</td>              <td>1</td>           <td>F-statistic:</td>      <td>1.269e+04</td> 
</tr>
<tr>
     <td>Df Residuals:</td>          <td>6026</td>       <td>Prob (F-statistic):</td>    <td>0.00</td>    
</tr>
<tr>
      <td>R-squared:</td>            <td>0.678</td>            <td>Scale:</td>        <td>8.8297e+10</td> 
</tr>
</table>
<table class="simpletable">
<tr>
      <td></td>        <th>Coef.</th>   <th>Std.Err.</th>      <th>t</th>     <th>P>|t|</th>   <th>[0.025</th>     <th>0.975]</th>  
</tr>
<tr>
  <th>intercept</th> <td>9587.8878</td> <td>7637.4788</td>  <td>1.2554</td>  <td>0.2094</td> <td>-5384.3028</td> <td>24560.0784</td>
</tr>
<tr>
  <th>area</th>      <td>348.4664</td>   <td>3.0930</td>   <td>112.6619</td> <td>0.0000</td>  <td>342.4029</td>   <td>354.5298</td> 
</tr>
</table>
<table class="simpletable">
<tr>
     <td>Omnibus:</td>    <td>368.609</td>  <td>Durbin-Watson:</td>    <td>2.007</td> 
</tr>
<tr>
  <td>Prob(Omnibus):</td>  <td>0.000</td>  <td>Jarque-Bera (JB):</td> <td>349.279</td>
</tr>
<tr>
       <td>Skew:</td>      <td>0.534</td>      <td>Prob(JB):</td>      <td>0.000</td> 
</tr>
<tr>
     <td>Kurtosis:</td>    <td>2.499</td>   <td>Condition No.:</td>    <td>4928</td>  
</tr>
</table>



## Interpretation
The adjusted R-squared remained at 0.678, indicating we didn't lose much by dropping **bedrooms** and **bathrooms** from our model.

So, considering the quantitative variables available, we should only use **area**. But what about the categorical variables?

# Working with Categorical Data
So far we have only worked with quantitative variables in our examples. But it is also possible to work with Categorical data, such as Neighborhood or Style.

We can use Hot-One encoding, creating new dummy variables receiving value of 1 if the category is true, or 0 otherwise.

We can easily implement this by using the `pandas.get_dummies` function.

>**Note**: we use the `pd.get_dummies` function inside the `df.join`, in order to add the created columns to the already existing dataframe.


```python
df = df.join(pd.get_dummies(df.neighborhood, prefix='neighb'))
df = df.join(pd.get_dummies(df['style'], prefix='style'))
```


```python
df.head(2)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>neighborhood</th>
      <th>area</th>
      <th>bedrooms</th>
      <th>bathrooms</th>
      <th>style</th>
      <th>price</th>
      <th>intercept</th>
      <th>neighb_A</th>
      <th>neighb_B</th>
      <th>neighb_C</th>
      <th>style_lodge</th>
      <th>style_ranch</th>
      <th>style_victorian</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>B</td>
      <td>1188</td>
      <td>3</td>
      <td>2</td>
      <td>ranch</td>
      <td>598291</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>B</td>
      <td>3512</td>
      <td>5</td>
      <td>3</td>
      <td>victorian</td>
      <td>1744259</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>



But before we can incorporate the categorical data in the model, it is necessary to make sure the selected features $X$ are <u>Full Rank</u>. That is, <u>all explanatory variables are linearly independent</u>.

>If a dummy variable holds the value 1, it specifies the value for the categorical variable. But we can just as easily identify it if all other dummy variables are 0.

Therefore, all we need to do is leave one of the dummy variables out of the model - for each group of categorical variable.
This "left out" variable will be the **baseline** for comparison among the others from the same category.

Let's use only *Neighborhood* (A, B, C) and *Style* (Victorian, Lodge, Ranch) in our model:


```python
df.columns
```




    Index(['neighborhood', 'area', 'bedrooms', 'bathrooms', 'style', 'price',
           'intercept', 'neighb_A', 'neighb_B', 'neighb_C', 'style_lodge',
           'style_ranch', 'style_victorian'],
          dtype='object')




```python
# neighborhood Baseline = A
# style Baseline = Victorian
feature_list = ['intercept','neighb_B', 'neighb_C',
                'style_lodge','style_ranch']

# Note how Statsmodels uses the order (y, X)
model = sm.OLS(df.price, df[feature_list])
results = model.fit()
results.summary2()
```




<table class="simpletable">
<tr>
        <td>Model:</td>               <td>OLS</td>         <td>Adj. R-squared:</td>      <td>0.584</td>   
</tr>
<tr>
  <td>Dependent Variable:</td>       <td>price</td>             <td>AIC:</td>         <td>170590.6058</td>
</tr>
<tr>
         <td>Date:</td>        <td>2019-04-03 16:30</td>        <td>BIC:</td>         <td>170624.1267</td>
</tr>
<tr>
   <td>No. Observations:</td>        <td>6028</td>         <td>Log-Likelihood:</td>     <td>-85290.</td>  
</tr>
<tr>
       <td>Df Model:</td>              <td>4</td>           <td>F-statistic:</td>        <td>2113.</td>   
</tr>
<tr>
     <td>Df Residuals:</td>          <td>6023</td>       <td>Prob (F-statistic):</td>    <td>0.00</td>    
</tr>
<tr>
      <td>R-squared:</td>            <td>0.584</td>            <td>Scale:</td>        <td>1.1417e+11</td> 
</tr>
</table>
<table class="simpletable">
<tr>
       <td></td>           <th>Coef.</th>     <th>Std.Err.</th>      <th>t</th>     <th>P>|t|</th>    <th>[0.025</th>       <th>0.975]</th>   
</tr>
<tr>
  <th>intercept</th>    <td>836696.6000</td>  <td>8960.8404</td>  <td>93.3726</td> <td>0.0000</td>  <td>819130.1455</td>  <td>854263.0546</td>
</tr>
<tr>
  <th>neighb_B</th>     <td>524695.6439</td> <td>10388.4835</td>  <td>50.5074</td> <td>0.0000</td>  <td>504330.4978</td>  <td>545060.7899</td>
</tr>
<tr>
  <th>neighb_C</th>     <td>-6685.7296</td>  <td>11272.2342</td>  <td>-0.5931</td> <td>0.5531</td>  <td>-28783.3433</td>  <td>15411.8841</td> 
</tr>
<tr>
  <th>style_lodge</th> <td>-737284.1846</td> <td>11446.1540</td> <td>-64.4133</td> <td>0.0000</td> <td>-759722.7434</td> <td>-714845.6259</td>
</tr>
<tr>
  <th>style_ranch</th> <td>-473375.7836</td> <td>10072.6387</td> <td>-46.9962</td> <td>0.0000</td> <td>-493121.7607</td> <td>-453629.8065</td>
</tr>
</table>
<table class="simpletable">
<tr>
     <td>Omnibus:</td>    <td>930.122</td>  <td>Durbin-Watson:</td>     <td>2.010</td> 
</tr>
<tr>
  <td>Prob(Omnibus):</td>  <td>0.000</td>  <td>Jarque-Bera (JB):</td> <td>2984.126</td>
</tr>
<tr>
       <td>Skew:</td>      <td>0.787</td>      <td>Prob(JB):</td>       <td>0.000</td> 
</tr>
<tr>
     <td>Kurtosis:</td>    <td>6.067</td>   <td>Condition No.:</td>       <td>4</td>   
</tr>
</table>



## Interpretation
Using both categorical data at the same time, our model resulted in an R-squared of 0.584. Which is less than before using only quantitative data, but means there is some explanatory power to these categorical variables.

All variables seem to be significant, with p-value < $\alpha$. Except for Neighborhood C. 
>This means `neighb_C` is not statistically significant <u>in comparison to the **baseline**</u>, `neighb_A`.

>To interpret the significance between a dummy and the other variables, we look at the <u>confidence interval</u> (given by [0.025 and 0.975]): if they do not overlap, then it is considered significant. In this case, `neighb_C` is significant to the model, just not when compared to the baseline... suggesting that **Neighborhood A** represents the characteristics of **Neighborhood C**.

+ **intercept**: represents the impact of the Baselines. In this case, we expect a *Victorian* house located in *Neighborhood A* to cost *836,696* price unit.

+ **Coefficients**: since we only have dummy variables, they are interpreted against their own category *baseline*
    - **neighb_B**: we predict a house in *Neighborhood B* to cost *524,695* more than in *Neighborhood A* (baseline). For a *Victorian* house, it would cost *836,696 + 524,695 = 1,361,391* price unit.
    - **neighb_C**: since its p-value > $\alpha$, it is not significant compared to the baseline - so we ignore the interpretation of this coefficient
    - **style_lodge**: we predict a *Lodge* house to cost *737,284* less than a *Victorian* (baseline). In *Neighborhood A*, it would cost *836,696 - 737,284 = 99,412* price unit
    - **style_ranch**: we predict a *Ranch* to cost *473,375* less than a *Victorian* (baseline). In *Neighborhood A*, it would cost *836,696 - 473,375 = 363,321* price unit

# Putting it all together
Now that we covered how to work with and interpret quantitative and categorical variables, as well as dealing with multicollinearity, it is time to put it all together:

>**Note**: we leave **Neighborhood C** out of the model, since our previous regression - using only categorical variables - showed it to be insignificant.

In this case, **Neighborhood A** is non longer the Baseline and needs to be explicitly added to the model:


```python
# style Baseline = Victorian
feature_list = ['intercept','area','neighb_A', 'neighb_B',
                'style_lodge', 'style_ranch']

# Note how Statsmodels uses the order (y, X)
model = sm.OLS(df.price, df[feature_list])
results = model.fit()
results.summary2()
```




<table class="simpletable">
<tr>
        <td>Model:</td>               <td>OLS</td>         <td>Adj. R-squared:</td>      <td>0.919</td>   
</tr>
<tr>
  <td>Dependent Variable:</td>       <td>price</td>             <td>AIC:</td>         <td>160707.1908</td>
</tr>
<tr>
         <td>Date:</td>        <td>2019-04-03 16:30</td>        <td>BIC:</td>         <td>160747.4158</td>
</tr>
<tr>
   <td>No. Observations:</td>        <td>6028</td>         <td>Log-Likelihood:</td>     <td>-80348.</td>  
</tr>
<tr>
       <td>Df Model:</td>              <td>5</td>           <td>F-statistic:</td>      <td>1.372e+04</td> 
</tr>
<tr>
     <td>Df Residuals:</td>          <td>6022</td>       <td>Prob (F-statistic):</td>    <td>0.00</td>    
</tr>
<tr>
      <td>R-squared:</td>            <td>0.919</td>            <td>Scale:</td>        <td>2.2153e+10</td> 
</tr>
</table>
<table class="simpletable">
<tr>
       <td></td>           <th>Coef.</th>    <th>Std.Err.</th>      <th>t</th>     <th>P>|t|</th>    <th>[0.025</th>       <th>0.975]</th>   
</tr>
<tr>
  <th>intercept</th>   <td>-204570.6449</td> <td>7699.7035</td> <td>-26.5686</td> <td>0.0000</td> <td>-219664.8204</td> <td>-189476.4695</td>
</tr>
<tr>
  <th>area</th>          <td>348.7375</td>    <td>2.2047</td>   <td>158.1766</td> <td>0.0000</td>   <td>344.4155</td>     <td>353.0596</td>  
</tr>
<tr>
  <th>neighb_A</th>      <td>-194.2464</td>  <td>4965.4594</td>  <td>-0.0391</td> <td>0.9688</td>  <td>-9928.3245</td>    <td>9539.8317</td> 
</tr>
<tr>
  <th>neighb_B</th>     <td>524266.5778</td> <td>4687.4845</td> <td>111.8439</td> <td>0.0000</td>  <td>515077.4301</td>  <td>533455.7254</td>
</tr>
<tr>
  <th>style_lodge</th>   <td>6262.7365</td>  <td>6893.2931</td>  <td>0.9085</td>  <td>0.3636</td>  <td>-7250.5858</td>   <td>19776.0588</td> 
</tr>
<tr>
  <th>style_ranch</th>   <td>4288.0333</td>  <td>5367.0317</td>  <td>0.7990</td>  <td>0.4243</td>  <td>-6233.2702</td>   <td>14809.3368</td> 
</tr>
</table>
<table class="simpletable">
<tr>
     <td>Omnibus:</td>    <td>114.369</td>  <td>Durbin-Watson:</td>    <td>2.002</td> 
</tr>
<tr>
  <td>Prob(Omnibus):</td>  <td>0.000</td>  <td>Jarque-Bera (JB):</td> <td>139.082</td>
</tr>
<tr>
       <td>Skew:</td>      <td>0.271</td>      <td>Prob(JB):</td>      <td>0.000</td> 
</tr>
<tr>
     <td>Kurtosis:</td>    <td>3.509</td>   <td>Condition No.:</td>    <td>13461</td> 
</tr>
</table>



Looks like **Neighborhood A** is not significant to the model. Let's remove it and interpret the results again:


```python
# style Baseline = Victorian
feature_list = ['intercept','area','neighb_B',
                'style_lodge', 'style_ranch']

# Note how Statsmodels uses the order (y, X)
model = sm.OLS(df.price, df[feature_list])
results = model.fit()
results.summary2()
```




<table class="simpletable">
<tr>
        <td>Model:</td>               <td>OLS</td>         <td>Adj. R-squared:</td>      <td>0.919</td>   
</tr>
<tr>
  <td>Dependent Variable:</td>       <td>price</td>             <td>AIC:</td>         <td>160705.1923</td>
</tr>
<tr>
         <td>Date:</td>        <td>2019-04-03 16:30</td>        <td>BIC:</td>         <td>160738.7132</td>
</tr>
<tr>
   <td>No. Observations:</td>        <td>6028</td>         <td>Log-Likelihood:</td>     <td>-80348.</td>  
</tr>
<tr>
       <td>Df Model:</td>              <td>4</td>           <td>F-statistic:</td>      <td>1.715e+04</td> 
</tr>
<tr>
     <td>Df Residuals:</td>          <td>6023</td>       <td>Prob (F-statistic):</td>    <td>0.00</td>    
</tr>
<tr>
      <td>R-squared:</td>            <td>0.919</td>            <td>Scale:</td>        <td>2.2149e+10</td> 
</tr>
</table>
<table class="simpletable">
<tr>
       <td></td>           <th>Coef.</th>    <th>Std.Err.</th>      <th>t</th>     <th>P>|t|</th>    <th>[0.025</th>       <th>0.975]</th>   
</tr>
<tr>
  <th>intercept</th>   <td>-204669.1638</td> <td>7275.5954</td> <td>-28.1309</td> <td>0.0000</td> <td>-218931.9350</td> <td>-190406.3926</td>
</tr>
<tr>
  <th>area</th>          <td>348.7368</td>    <td>2.2045</td>   <td>158.1954</td> <td>0.0000</td>   <td>344.4152</td>     <td>353.0583</td>  
</tr>
<tr>
  <th>neighb_B</th>     <td>524367.7695</td> <td>3908.8100</td> <td>134.1502</td> <td>0.0000</td>  <td>516705.1029</td>  <td>532030.4361</td>
</tr>
<tr>
  <th>style_lodge</th>   <td>6259.1344</td>  <td>6892.1067</td>  <td>0.9082</td>  <td>0.3638</td>  <td>-7251.8617</td>   <td>19770.1304</td> 
</tr>
<tr>
  <th>style_ranch</th>   <td>4286.9410</td>  <td>5366.5142</td>  <td>0.7988</td>  <td>0.4244</td>  <td>-6233.3477</td>   <td>14807.2296</td> 
</tr>
</table>
<table class="simpletable">
<tr>
     <td>Omnibus:</td>    <td>114.373</td>  <td>Durbin-Watson:</td>    <td>2.002</td> 
</tr>
<tr>
  <td>Prob(Omnibus):</td>  <td>0.000</td>  <td>Jarque-Bera (JB):</td> <td>139.090</td>
</tr>
<tr>
       <td>Skew:</td>      <td>0.271</td>      <td>Prob(JB):</td>      <td>0.000</td> 
</tr>
<tr>
     <td>Kurtosis:</td>    <td>3.509</td>   <td>Condition No.:</td>    <td>13175</td> 
</tr>
</table>



Looking at the high p-values of **style_lodge** and **style_ranch**, they do not seem to be significant in our model.

Let's regress our model again, removing **style_lodge**.
>**Note**: since we are removing Lodge, Victorian style no longer is the baseline, hence we have to explicitly add it to the model


```python
feature_list = ['intercept','area','neighb_B','style_victorian', 'style_ranch']

# Note how Statsmodels uses the order (y, X)
model = sm.OLS(df.price, df[feature_list])
results = model.fit()
results.summary2()
```




<table class="simpletable">
<tr>
        <td>Model:</td>               <td>OLS</td>         <td>Adj. R-squared:</td>      <td>0.919</td>   
</tr>
<tr>
  <td>Dependent Variable:</td>       <td>price</td>             <td>AIC:</td>         <td>160705.1923</td>
</tr>
<tr>
         <td>Date:</td>        <td>2019-04-03 16:30</td>        <td>BIC:</td>         <td>160738.7132</td>
</tr>
<tr>
   <td>No. Observations:</td>        <td>6028</td>         <td>Log-Likelihood:</td>     <td>-80348.</td>  
</tr>
<tr>
       <td>Df Model:</td>              <td>4</td>           <td>F-statistic:</td>      <td>1.715e+04</td> 
</tr>
<tr>
     <td>Df Residuals:</td>          <td>6023</td>       <td>Prob (F-statistic):</td>    <td>0.00</td>    
</tr>
<tr>
      <td>R-squared:</td>            <td>0.919</td>            <td>Scale:</td>        <td>2.2149e+10</td> 
</tr>
</table>
<table class="simpletable">
<tr>
         <td></td>             <th>Coef.</th>    <th>Std.Err.</th>      <th>t</th>     <th>P>|t|</th>    <th>[0.025</th>       <th>0.975]</th>   
</tr>
<tr>
  <th>intercept</th>       <td>-198410.0294</td> <td>4886.8434</td> <td>-40.6009</td> <td>0.0000</td> <td>-207989.9917</td> <td>-188830.0672</td>
</tr>
<tr>
  <th>area</th>              <td>348.7368</td>    <td>2.2045</td>   <td>158.1954</td> <td>0.0000</td>   <td>344.4152</td>     <td>353.0583</td>  
</tr>
<tr>
  <th>neighb_B</th>         <td>524367.7695</td> <td>3908.8100</td> <td>134.1502</td> <td>0.0000</td>  <td>516705.1029</td>  <td>532030.4361</td>
</tr>
<tr>
  <th>style_victorian</th>  <td>-6259.1344</td>  <td>6892.1067</td>  <td>-0.9082</td> <td>0.3638</td>  <td>-19770.1304</td>   <td>7251.8617</td> 
</tr>
<tr>
  <th>style_ranch</th>      <td>-1972.1934</td>  <td>5756.6924</td>  <td>-0.3426</td> <td>0.7319</td>  <td>-13257.3710</td>   <td>9312.9843</td> 
</tr>
</table>
<table class="simpletable">
<tr>
     <td>Omnibus:</td>    <td>114.373</td>  <td>Durbin-Watson:</td>    <td>2.002</td> 
</tr>
<tr>
  <td>Prob(Omnibus):</td>  <td>0.000</td>  <td>Jarque-Bera (JB):</td> <td>139.090</td>
</tr>
<tr>
       <td>Skew:</td>      <td>0.271</td>      <td>Prob(JB):</td>      <td>0.000</td> 
</tr>
<tr>
     <td>Kurtosis:</td>    <td>3.509</td>   <td>Condition No.:</td>    <td>10991</td> 
</tr>
</table>



Once again, neither Victorian nor Ranch proved to be significant in the model.

Let's remove **style_ranch**:


```python
feature_list = ['intercept','area','neighb_B','style_victorian']

# Note how Statsmodels uses the order (y, X)
model = sm.OLS(df.price, df[feature_list])
results = model.fit()
results.summary2()
```




<table class="simpletable">
<tr>
        <td>Model:</td>               <td>OLS</td>         <td>Adj. R-squared:</td>      <td>0.919</td>   
</tr>
<tr>
  <td>Dependent Variable:</td>       <td>price</td>             <td>AIC:</td>         <td>160703.3098</td>
</tr>
<tr>
         <td>Date:</td>        <td>2019-04-03 16:30</td>        <td>BIC:</td>         <td>160730.1265</td>
</tr>
<tr>
   <td>No. Observations:</td>        <td>6028</td>         <td>Log-Likelihood:</td>     <td>-80348.</td>  
</tr>
<tr>
       <td>Df Model:</td>              <td>3</td>           <td>F-statistic:</td>      <td>2.287e+04</td> 
</tr>
<tr>
     <td>Df Residuals:</td>          <td>6024</td>       <td>Prob (F-statistic):</td>    <td>0.00</td>    
</tr>
<tr>
      <td>R-squared:</td>            <td>0.919</td>            <td>Scale:</td>        <td>2.2146e+10</td> 
</tr>
</table>
<table class="simpletable">
<tr>
         <td></td>             <th>Coef.</th>    <th>Std.Err.</th>      <th>t</th>     <th>P>|t|</th>    <th>[0.025</th>       <th>0.975]</th>   
</tr>
<tr>
  <th>intercept</th>       <td>-199292.0351</td> <td>4153.3840</td> <td>-47.9831</td> <td>0.0000</td> <td>-207434.1541</td> <td>-191149.9162</td>
</tr>
<tr>
  <th>area</th>              <td>348.5163</td>    <td>2.1083</td>   <td>165.3055</td> <td>0.0000</td>   <td>344.3833</td>     <td>352.6494</td>  
</tr>
<tr>
  <th>neighb_B</th>         <td>524359.1981</td> <td>3908.4435</td> <td>134.1606</td> <td>0.0000</td>  <td>516697.2501</td>  <td>532021.1461</td>
</tr>
<tr>
  <th>style_victorian</th>  <td>-4716.5525</td>  <td>5217.5622</td>  <td>-0.9040</td> <td>0.3660</td>  <td>-14944.8416</td>   <td>5511.7367</td> 
</tr>
</table>
<table class="simpletable">
<tr>
     <td>Omnibus:</td>    <td>118.288</td>  <td>Durbin-Watson:</td>    <td>2.002</td> 
</tr>
<tr>
  <td>Prob(Omnibus):</td>  <td>0.000</td>  <td>Jarque-Bera (JB):</td> <td>143.564</td>
</tr>
<tr>
       <td>Skew:</td>      <td>0.278</td>      <td>Prob(JB):</td>      <td>0.000</td> 
</tr>
<tr>
     <td>Kurtosis:</td>    <td>3.512</td>   <td>Condition No.:</td>    <td>6807</td>  
</tr>
</table>



Looks like the house style is not significant at all in our model.

Let's also remove Style Victorian from our explanatory variables:


```python
feature_list = ['intercept','area', 'neighb_B']

# Note how Statsmodels uses the order (y, X)
model = sm.OLS(df.price, df[feature_list])
results = model.fit()
results.summary2()
```




<table class="simpletable">
<tr>
        <td>Model:</td>               <td>OLS</td>         <td>Adj. R-squared:</td>      <td>0.919</td>   
</tr>
<tr>
  <td>Dependent Variable:</td>       <td>price</td>             <td>AIC:</td>         <td>160702.1274</td>
</tr>
<tr>
         <td>Date:</td>        <td>2019-04-03 16:30</td>        <td>BIC:</td>         <td>160722.2399</td>
</tr>
<tr>
   <td>No. Observations:</td>        <td>6028</td>         <td>Log-Likelihood:</td>     <td>-80348.</td>  
</tr>
<tr>
       <td>Df Model:</td>              <td>2</td>           <td>F-statistic:</td>      <td>3.430e+04</td> 
</tr>
<tr>
     <td>Df Residuals:</td>          <td>6025</td>       <td>Prob (F-statistic):</td>    <td>0.00</td>    
</tr>
<tr>
      <td>R-squared:</td>            <td>0.919</td>            <td>Scale:</td>        <td>2.2146e+10</td> 
</tr>
</table>
<table class="simpletable">
<tr>
      <td></td>          <th>Coef.</th>    <th>Std.Err.</th>      <th>t</th>     <th>P>|t|</th>    <th>[0.025</th>       <th>0.975]</th>   
</tr>
<tr>
  <th>intercept</th> <td>-198881.7929</td> <td>4128.4536</td> <td>-48.1734</td> <td>0.0000</td> <td>-206975.0391</td> <td>-190788.5467</td>
</tr>
<tr>
  <th>area</th>        <td>347.2235</td>    <td>1.5490</td>   <td>224.1543</td> <td>0.0000</td>   <td>344.1868</td>     <td>350.2602</td>  
</tr>
<tr>
  <th>neighb_B</th>   <td>524377.5839</td> <td>3908.3313</td> <td>134.1692</td> <td>0.0000</td>  <td>516715.8561</td>  <td>532039.3117</td>
</tr>
</table>
<table class="simpletable">
<tr>
     <td>Omnibus:</td>    <td>125.346</td>  <td>Durbin-Watson:</td>    <td>2.002</td> 
</tr>
<tr>
  <td>Prob(Omnibus):</td>  <td>0.000</td>  <td>Jarque-Bera (JB):</td> <td>152.989</td>
</tr>
<tr>
       <td>Skew:</td>      <td>0.287</td>      <td>Prob(JB):</td>      <td>0.000</td> 
</tr>
<tr>
     <td>Kurtosis:</td>    <td>3.528</td>   <td>Condition No.:</td>    <td>6083</td>  
</tr>
</table>



## Intepretation
Finally, all features are significant (p-value < $\alpha$).

Interestingly, the adjusted R-squared never changed through the process of eliminating insignificant explanatory variables, meaning these final ones are indeed the most important ones for predicting house price.

Basically, it means that for every 1 area unit increase, we expect an increase of 347 price unit. If a house is situated in Neighborhood B, we expect it to cost, on average, 524,377 more than in Neighborhoods A or C.

Even though the intercept has a negative value, this equation resulted in the best predicting performance so far (adjusted R-squared of 0.919). One can argue that every house - even the smallest one - still have area greater than 0. For houses in Neighborhood A and C with square footage greater than 580, we start to predict positive values for price.

# Flexible Linear Models
The linear model is also capable of incorporating non-linear relationships between the explanatory variables and the response, such as: **Higher Order** and **Interaction**

### Higher Order
Whenever there is a clear curve-like pattern between the pair plot of an explanatory variable $x_i$ and the response $y$
+ the higher order should always be the amount of peaks + 1. For example, in a U shaped pair plot, there is 1 peak, hence the feature should be of order 2: $x_i^2$


+ whenever using higher order terms in the model, always include the lower order terms as well. In this example of order 2, we would have the following terms: $\hat y = b_0 + b_1.x_1 + b_2.x_1^2$


### Interaction
Whenever the relationship between a feature ($x_1$) and the response ($y$) change according to another feature ($x_2$)
+ the need for an **interaction** term can be visually investigated by adding the *hue* parameter to the typical pair plot. If the relationship of $x_1$ and $y$ have different slopes based on $x_2$, then we should add the interaction term: $x_1.x_2$


+ Similarly to higher order, if implemented in the model, it is also necessary to add the "lower" terms, so the model would look like this: $\hat y = b_0 + b_1.x_1 + b_2.x_2 + b_3.x_1.x_2$

### Downside
But there is a huge caveat: while the prediction power of the linear model improves by adding flexibility (higher order and interaction), it loses the ease of interpretation of the coefficients.

It is no longer straightforward - as seen in the previous models - to interpret the impact of quantitative or categorical variables. Any change in the explanatory variable would be reflected in the simple term, $b_1.x_1$ as well as the higher order one, $b_2.x_1^2$.

Enter the trade-off of linear models: flexible terms might improve model performance, but at the cost of interpretability.

>If you are interested in <u>understanding the impact</u> of each variable over the response, it is recommended to shy away from using more flexible terms.

>... and if you are only interested in <u>making accurate predictions</u>, than Linear Models would not be your best bet anyway!


```python
sns.pairplot(data=df, y_vars='price', x_vars=['area', 'bedrooms','bathrooms']);
```


![png](output_47_0.png)


The relationship between the features and the response does not present any curves. Hence, there is no base to implement a higher order term in the model.


```python
# Pair plot by Neighborhood
sns.lmplot(x='area', y='price', data=df, hue='neighborhood', size=4,
           markers=['o','x','+'], legend_out=False, aspect=1.5)
plt.title('Pair plot of Price and Area by Neighborhood');
```


![png](output_49_0.png)


It is clear to see in the image above that the relationship between **area** and **price** does vary based on the **neighborhood**. Which suggests we should add the interaction term.

In this particular case, Neighborhood A and C have the exact same behavior. Therefore, we only need to verify whether the house is located in Neighborhood B or not (using the dummy variable, **neighb_B**).


```python
# Creating interaction term to be incorporated in the model
df['area_neighb'] = df.area * df.neighb_B
```


```python
df['intercept'] = 1
features_list = ['intercept', 'area','neighb_B','area_neighb']
model = sm.OLS(df.price, df[features_list])
result = model.fit()
result.summary2()
```




<table class="simpletable">
<tr>
        <td>Model:</td>               <td>OLS</td>         <td>Adj. R-squared:</td>      <td>1.000</td>  
</tr>
<tr>
  <td>Dependent Variable:</td>       <td>price</td>             <td>AIC:</td>         <td>71986.6799</td>
</tr>
<tr>
         <td>Date:</td>        <td>2019-04-03 16:30</td>        <td>BIC:</td>         <td>72013.4966</td>
</tr>
<tr>
   <td>No. Observations:</td>        <td>6028</td>         <td>Log-Likelihood:</td>     <td>-35989.</td> 
</tr>
<tr>
       <td>Df Model:</td>              <td>3</td>           <td>F-statistic:</td>      <td>6.131e+10</td>
</tr>
<tr>
     <td>Df Residuals:</td>          <td>6024</td>       <td>Prob (F-statistic):</td>    <td>0.00</td>   
</tr>
<tr>
      <td>R-squared:</td>            <td>1.000</td>            <td>Scale:</td>          <td>8986.8</td>  
</tr>
</table>
<table class="simpletable">
<tr>
       <td></td>          <th>Coef.</th>   <th>Std.Err.</th>      <th>t</th>       <th>P>|t|</th>   <th>[0.025</th>     <th>0.975]</th>  
</tr>
<tr>
  <th>intercept</th>   <td>12198.0608</td>  <td>3.1492</td>   <td>3873.3747</td>  <td>0.0000</td> <td>12191.8872</td> <td>12204.2343</td>
</tr>
<tr>
  <th>area</th>         <td>248.1610</td>   <td>0.0013</td>  <td>194094.9191</td> <td>0.0000</td>  <td>248.1585</td>   <td>248.1635</td> 
</tr>
<tr>
  <th>neighb_B</th>     <td>132.5642</td>   <td>4.9709</td>    <td>26.6682</td>   <td>0.0000</td>  <td>122.8195</td>   <td>142.3089</td> 
</tr>
<tr>
  <th>area_neighb</th>  <td>245.0016</td>   <td>0.0020</td>  <td>121848.3887</td> <td>0.0000</td>  <td>244.9976</td>   <td>245.0055</td> 
</tr>
</table>
<table class="simpletable">
<tr>
     <td>Omnibus:</td>    <td>214.393</td>  <td>Durbin-Watson:</td>    <td>1.979</td> 
</tr>
<tr>
  <td>Prob(Omnibus):</td>  <td>0.000</td>  <td>Jarque-Bera (JB):</td> <td>398.418</td>
</tr>
<tr>
       <td>Skew:</td>     <td>-0.277</td>      <td>Prob(JB):</td>      <td>0.000</td> 
</tr>
<tr>
     <td>Kurtosis:</td>    <td>4.131</td>   <td>Condition No.:</td>    <td>12098</td> 
</tr>
</table>



## Interpretation
Our best model performance, with an impressive adjusted R-squared of 100%. All variables are significant and their signs match the expected direction (positive).

Although the interpretation is not as easy as before, we can still try to make  some sense of it because this interaction is the special case involving a binary variable.

+ **intercept**: we expect all houses to have the initial price of 12,198 price unit.
+ **Coefficients**:
    - for houses in Neighborhood A and C, we expect the price to increase by 248 price unit for every additional area unit. So, the average price of a house of 120 sqft situated in Neighborhood A or C is: $12,198 + 248*120 = 41,958$
    - for houses in Neighborhood B, we expect the price to increase by **493**  (*248* + *245*) price unit for every additional area unit. There would also be an additional 132 price unit on top of the intercept, independent of the area. So the average price of a house of 120 sqft located in Neighborhood B is: $12,198 + (132) + 120*(248 + 245) = 71,490$

# Conclusion
In this post we reviewed the basics of Linear Regression, covering how to model and interpret the results of Simple and Multiple Linear Regression - working with quantitative as well as categorical variables.

We also covered Flexible Linear Models, when to implement them, and the intrinsic trade-off between flexibility and interpretability.

We briefly mentioned some of the potential modeling problems regarding the basic assumptions required for the regression model, covering some strategies on how to identify and address multicollinearity.

In the next post, we will introduce the basics of Logistic Regression (when to use it, how to interpret the results) and how to evaluate Model Performance (accuracy, precision, recall, AUROC *etc.*).

