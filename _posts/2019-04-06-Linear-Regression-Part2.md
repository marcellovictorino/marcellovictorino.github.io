---
title: Linear Regression Part 2 - Potential Modeling Problems
excerpt: Second of a 3-part series on Linear Regression, this post covers some of the potential modeling problems, as well as the required assumptions of Linear Regression. Namely, 1) Outliers and Leverage Points, 2) Multicollinearity, 3) Linearity, 4) Correlated Errors, 5) Normality of Residuals, and 6) Constant Variance of Errors. 
tags: [Data Analysis]
categories: [Post]
date: 2019-04-10 00:00:00 +0000
---

# Introduction
Linear Regression is the linear approach to modeling the relationship between a quantitative response ($$y$$) and one or more explanatory variables ($$X$$); also known as Response and Features, respectively.

This post focuses on the **potential modeling problems** that might arise due to the required assumptions for Linear Regression.

The code for this post can be found in [this Notebook](https://github.com/marcellovictorino/Practical-Statistics/blob/master/4_%20Linear%20Regression%20-%20Interpreting%20Results%20and%20Model%20Performance/Part%202%20-%20Potential%20Modeling%20Problems/Linear%20Regression%20Part%202of3%20-%20Potential%20Modeling%20Problems.ipynb).

Let's load the data and necessary libraries. The toy dataset used in this example contain information on house prices and its characteristics, such as Neighborhood, square footage, # of bedrooms and bathrooms.

```python
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm
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
<table border="0" class="dataframe">
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

Now let's fit the final model identified in the Part 1 of Linear Regression series:

```python
# Creating dummy variable for Neighborhoods
df = df.join(pd.get_dummies(df.neighborhood, prefix='neighb'))

# Creating interaction term to be incorporated in the model
df['area_neighb'] = df.area * df.neighb_B

df['intercept'] = 1
features_list = ['intercept', 'area','neighb_B','area_neighb']

model = sm.OLS(df.price, df[features_list])
fitted_model = model.fit()
fitted_model.summary2()
```

<table class="simpletable">
<tr>
        <td>Model:</td>               <td>OLS</td>         <td>Adj. R-squared:</td>      <td>1.000</td>  
</tr>
<tr>
  <td>Dependent Variable:</td>       <td>price</td>             <td>AIC:</td>         <td>71986.6799</td>
</tr>
<tr>
         <td>Date:</td>        <td>2019-04-10 10:22</td>        <td>BIC:</td>         <td>72013.4966</td>
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

<br>
# Potential Modeling Problems
Depending on the main objective of your model (prediction, inference, most important features), you might need to address some specific problems - but not necessarily all of them.

These are the potential problems related to [Multiple Linear Regression assumptions](http://people.duke.edu/~rnau/testing.htm): some should be checked when choosing the predictors for the model; others only after the model has been fitted - while some should be checked in both situations.

<br>
## 1. Outliers & Leverage Points
Outliers and leverage points lie far away from the regular trends of the data. These points can have a large influence on the fitted model and the required assumptions, such as linearity and normality of errors. Therefore, this should be the very first step to check when fitting a model. 

+ **Outliers**: extreme values on the y axis, do not follow the trend
+ **Leverage Points**: follows the trend, but extreme values on the x axis

Most of the time, these points are resulted by typos (data entry), malfunctioning sensors (bad data collected), or even just extremely rare events. 
>$$\implies$$ These cases can be removed from the dataset or inputed using k-nearest neighbors, mean, or median.

Other times, these are actually correct and true data points (known as **Natural Outliers**), not necessarily measurement or data entry errors. 

This situation would **not** call for removal, instead it requires further exploration to better understand the data. For example, fraud detection goal is exactly identifying extreme out-of-pattern transactions. 
>$$\implies$$ 'Fixing' is more subjective. If there is a large number of outliers, it is advised to create separate models.

{%include figure image_path='images\Linear-Regression-Part2\Extreme Values.png'
caption='Example of Outliers and Leverage Points'%}


There are many techniques to combat this, such as regularization techniques (Ridge, Lasso regression) and more empirical methods.

[Regularization](https://towardsdatascience.com/regularization-in-machine-learning-76441ddcf99a) significantly reduces model variance without substantial impact on bias. But their implementation is more complex and will be covered in a future post.

For now, we can focus on the more pragmatical methods:

+ **Tukey's Fence**
+ **Standardized Residuals**
+ **Outliers Removal Strategy**
+ **Unsupervised Outlier Detection**:
    - *Isolation Forest*
    - *Local Outlier Factor*
+ **Deleted Residuals**:
    - *Difference in FITS*
    - *Cook's Distance*

<br>
### 1.1) Tukey's Fence
Applied on predictors individually. Typically represented in a Box-Plot, values more extreme than the lower and upper boundaries of Tukey's fence are considered outliers and should be further investigated:
1. Calculate the first and third quartiles, $$Q_1$$ and $$Q_3$$
2. Calculate the Inter-Quartile Range, $$IQR = Q_3 - Q_1$$
3. Remove values more extreme than upper and lower boundaries:
    - **Upper**: $$Q_3 + 1.5 \times IQR$$
    - **Lower**: $$Q_1 - 1.5 \times IQR$$
    
>**Note**: it is important to note this method is commonly used, but only captures extreme values for univariate distribution. It is important to investigate multivariate distribution as well.


```python
def outlier_Tukeys_Fence(df, variable, factor=1.5, drop_outliers=False):
    Q3 = df[variable].quantile(0.75)
    Q1 = df[variable].quantile(0.25)
    IQR = Q3 - Q1
    ...

df2 = df.copy()
len(df2)
```
*6028*


```python
outlier_Tukeys_Fence(df2, 'area', drop_outliers=True)
len(df2)
```
*Boundaries: [-1,631.0 | 5,985.0]*<br>
*Points outside Upper Fence: 26*<br>
*Points outside Lower Fence: 0*


*6002*

{%include figure image_path='images\Linear-Regression-Part2\outlier - tukeys fence.png'
caption="Plot highlighting area outliers identified by Tukey's Fence method"%}

<br>
### 1.2) Standardized Residuals
Applied on the final model. A slight twist on the typical plot of Residuals by Fitted values. This method uses the standardized residual instead - which is obtained by dividing the error by the standard deviation of errors, $$\frac{e}{std(e)}$$
+ **Outliers**: Values more extreme than $$\pm$$ **3 standard deviation**, since they represent **99.7%** of data.


```python
def outliers_Standard_Error(df, fitted_model=fitted_model):
    # Standardized Error: absolute values > 3 means more extreme than 99.7%
    df['error'] = fitted_model.resid
    df['fitted_value'] = fitted_model.fittedvalues
    df['standard_error'] = df.error/np.std(df.error)
    ...

outliers_Standard_Error(df)
```
*Points > 3 Std: 24*<br>
*Points < -3 Std: 28*<br>
*Total of possible outliers: 52*

{%include figure image_path='images\Linear-Regression-Part2\outlier - standard error.png'
caption='Plot of Standardized Residuals, highlighting possible outliers (std > 3)'%}

<br>
### 1.3) Outliers Removal Strategy
This more generalized strategy can be implemented in any model and repeated multiple times, until the best model fit is found:

1. Train the model over all data points from the Train dataset
2. Remove 10% points with largest residual error
3. Re-train the model and evaluate against the Test dataset
4. Repeat from step 2, until best model fit is found

<br>
### 1.4) Unsupervised Outlier Detection
Two of the most used unsupervised methods are [Isolation Forest & Local Outlier Factor](https://scikit-learn.org/stable/auto_examples/plot_anomaly_comparison.html#sphx-glr-auto-examples-plot-anomaly-comparison-py). *See linked documentation for detailed explanation*.

The downside of such advanced multivariate methods (random tree and density based) is the requirement to arbitrarily specify the **contamination factor** - that is, the percentage of dataset believed to be "bad".

It should be implemented iteratively using the **Outliers Removal Strategy** (evaluating model performance against a test set).


```python
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor

# percentage of dataset 'contaminated' with outliers
contamination_factor = 0.1

algorithm_list = [
("Isolation Forest", IsolationForest(behaviour='new',contamination=contamination_factor,random_state=42), 'Iso_Forest'),
("Local Outlier Factor", LocalOutlierFactor(n_neighbors=35, contamination=contamination_factor), 'LOF')]

# X = df[features_list]
X = df[['price', 'area', 'neighb_B', 'area_neighb']]
counter = 0

fig, ax = plt.subplots(1,2, figsize=(10,4))

for name, model, tag in algorithm_list:
    # Apply unsupervised method and store classification: 1 = Inliner; -1 or 0 = Outlier
    df[f'{tag}'] = model.fit_predict(X)
    
    # Rename for legend purpose
    df[f'{tag}'] = np.where(df[f'{tag}'] == 1, 'Inlier', 'Outlier')
    
    sns.scatterplot(df.index, df.fitted_value, hue=df[f'{tag}'], alpha=0.25, palette=['k','r'], ax=ax[counter], edgecolor=None)

    ax[counter].set_title(f'{name}');
    
    # Amount of Outliers (dataset size * contamination_factor)
    outliers = len(df.query(f'{tag} == "Outlier"'))
    print(f'{name} | Outliers detected = {outliers:,}')
    
    counter += 1
    
plt.tight_layout()
fig.suptitle('Unsupervised Outlier Detection', size=16, weight='bold', y=1.1);
```
*Isolation Forest | Outliers detected = 602*<br>
*Local Outlier Factor | Outliers detected = 603*
    

{%include figure image_path='images\Linear-Regression-Part2\outlier - unsupervised method.png'
caption='Plot of Unsupervised Outlier Detection: Isolation Forest (random forest) and Local Outlier Factor (density based)'%}

<br>
### 1.5) [Deleted Residuals](https://newonlinecourses.science.psu.edu/stat501/node/340/)
The previous methods can be biased, since the model has been fitted using the entire dataset (extreme values included - if any). The alternative is a systematic approach of withholding one observation, fitting the model and calculating a specific metric - repeating the process for the entire data set. 

The interpretation follow each metric specific "guideline":
+ **Difference in Fits (DFFITS)**: values that stick out from the others are potentially influential extreme data points
+ **Cook's Distance**:
    - $$d > 0.5$$: might be influential points. Further investigate
    - $$d > 1$$: data point likely to be influential

>**Influential**: data points extreme enough to significantly impact the fitted model results, such as coefficients, p-value, and Confidence Intervals.

<br>
### 1.5.1) Difference in Fits (DFFITS)
Values that stick out from the others are potentially influential extreme data points:


```python
def outliers_DFFITS(df, fitted_model=fitted_model):
    df['dffits'] = fitted_model.get_influence().dffits[0]
    ...

outliers_DFFITS(df)
```

{%include figure image_path='images\Linear-Regression-Part2\outlier - DFFITS.png'
caption='Plot of Difference in FITS method, highlighting most extreme values (min and max)'%}


In this example, there are no points sticking out significantly more than the others, suggesting there are no influential extreme data points.

<br>
### 1.5.2) Cook's Distance
+ $$d > 0.5$$: might be influential points. Further investigate
+ $$d > 1.0$$: data point likely to be influential


```python
def outliers_Cooks_Distance(df, fitted_model=fitted_model):
    df['cooks_distance'] = fitted_model.get_influence().cooks_distance[0]

    # Plotting
    df.cooks_distance.plot(style='o', alpha=0.5, label='Not Influential')
    try:
        df.query('cooks_distance > 0.5').cooks_distance.plot(style='r',mfc='none',label='Possible Influential')
        plt.axhline(0.5, color='r', ls=':')

        df.query('cooks_distance > 1').cooks_distance.plot(style='r',label='Influential')
        plt.axhline(1, color='darkred', ls='--')
    except:
        pass
    ...

outliers_Cooks_Distance(df)
```

{%include figure image_path='images\Linear-Regression-Part2\outlier - cooks distance.png'
caption="Plot of Cook's Distance"%}


In this example, there are no points with **Cook's distance** greater than 0.5, suggesting there are no influential extreme data points.

<br>
## 2. Multicollinearity
One of the major 4 assumptions of Linear Regression, it assumes the (linear) **independence** between predictors. That is, it is not possible to infer one predictor based on the others.

Ultimately, we want the explanatory variables (predictors) to be correlated to the response, not between each other.

In the case of correlated features being used simultaneously as predictors, the model can yield:
+ Inverted coefficient sign from what we would expect
+ Unreliable p-value for the Hypothesis Test of variable coefficient being significantly $$\neq$$ 0

To investigate and identify multicollinearity between explanatory variables, we can use two approaches:
+ **Visual** 
+ **Metric**:
    - **Correlation Coefficient**, $$R$$
    - **Variance Inflation Factor (VIF)**
    
>**Hint**: Multicollinearity should be checked while choosing predictors: the modeling goal is to <u>identify relevant features</u> to the response, that are <u>independent from each other</u>.

<br>
### 2.1) Bivariate Plot (between Predictors)
Scatter plot between two variables looking for any relationship (positive or negative):

```python
# Visually investigating relationship between features
chart = sns.pairplot(data=df[['area', 'bedrooms', 'bathrooms', 'price']], kind='reg')
chart.fig.suptitle('Bivariate Plot: correlation between variables', y=1.02);
```
{%include figure image_path='images\Linear-Regression-Part2\Bivariate Plot.png'
caption="Bivariate plot between all variables"%}


It is possible to see the quantitative variables are correlated to each other, having a positive linear relationship.

<br>
### 2.2) Pearson's Correlation Matrix (Heatmap)

```python
def Correlation_Heatmap(df, variables_list=None, figsize=(10,8)):
    # Generate mask to hide the upper triangle/diagonal
    mask = np.zeros_like(corr, dtype=np.bool)
    mask[np.triu_indices_from(mask)] = True

    # Draw heatmap with mask
    sns.heatmap(corr, mask=mask, cmap=cmap, vmin=-1,vmax=1, center=0,square=True, annot=np.where(abs(corr)>0.1, corr,0),linewidths=0.5, cbar_kws={"shrink": 0.75})
    ...

Correlation_Heatmap(df, ['area','bedrooms','bathrooms', 'price'], figsize=(8,5))
```

{%include figure image_path='images\Linear-Regression-Part2\Correlation Heatmap.png'
caption="Pearson's Correlation Heatmap (lower diagonal only)"%}
<br>

According to the Pearson Correlation coefficient, the variable **area** is strongly correlated to **bedrooms** (0.9) and **bathrooms** (0.89), meaning we should use only 1 of these variables in the model.

The response **price** is strongly positively correlated to all quantitative variables, with **area** (0.82) stronger than **bedrooms** and **bathrooms** (0.74). 

$$\implies$$ Which suggests we should choose **area** as our predictor.

<br>
### 2.3) Variance Inflation Factor (VIF)
The general rule of thumb is that VIFs exceeding 4 warrants further investigation, while VIFs exceeding 10 indicates serious multicollinearity, requiring correction:

```python
# Metric investigation of correlation: VIF
def VIF(df="df", model_formula="y ~ x1 + x2 + x3"):
    from statsmodels.stats.outliers_influence import variance_inflation_factor

    y,X = dmatrices(model_formula, df, return_type='dataframe')
    vif = DataFrame()
    ...

model_formula = 'price ~ area + bedrooms + bathrooms'
VIF(df, model_formula)
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
<table border="0" class="dataframe">
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

<br>
Looks like there is a serious sign of multicollinearity between the explanatory variables. Let's remove the highest one, **bedrooms** and recalculate the VIF:


```python
# Removing correlated features: bedrooms
model_formula = 'price ~ area + bathrooms'
VIF(df, model_formula)
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
<table border="0" class="dataframe">
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

<br>
Removing **bedrooms** has improved the multicollinearity condition, but we still have **bathrooms** with a value greater than 4 - which warrants further investigation.

Since our previous analysis using the Pearson correlation coefficient showed **bathroom** to be strongly correlated to **area**, let's remove it and calculate the VIF one more time:


```python
# Removing correlated features
model_formula = 'price ~ area'
VIF(df, model_formula)
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
<table border="0" class="dataframe">
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

<br>
As expected, since we are only using one explanatory variable, the VIF reaches it's lowest possible value of 1. In this case, amongst all quantitative variables, we should regress our model using only the **area**.

<br>
## 3. Linearity
The main assumption of Linear Regression, there should be a linear relationship that truly exists between the response $$y$$ and the predictor variables $$X$$. If this isn't the case, the predictions will not be accurate and the interpretation of coefficients won't be useful.

We can verify if a linear relationship is reasonable by analyzing two plots:
+ **Pair Plot**
+ **Residual Plot**

>*Residuals* can be understood as the model error.<br>
<u>Ideally</u>, the error would be randomly scattered around 0, as seen in the <u>left-most case</u> in the image below.

{%include figure image_path='images\Linear-Regression-Part2\Linearity.png'
caption="Examples of Residual Plots regarding Linearity."%}

<br>
### 3.1) Pair Plot (Response vs Predictor)
Bivariate plot between the response and each individual predictor. There should be some kind of linear relationship, either positive or negative.


```python
# Pairplot of y vs. X
chart = sns.pairplot(df, x_vars=['area', 'neighb_B', 'area_neighb'], y_vars='price', kind='reg', height=3)
chart.fig.suptitle('Pair Plot: Response vs Predictor', y=1.1);
```

{%include figure image_path='images\Linear-Regression-Part2\Pair plot y-X.png'
caption="Pair plot between Response and Predictors"%}

<br>
### 3.2) Residual Plot
Plot of **residuals**, $$(y-\hat y)$$ by the predicted values, $$\hat y$$. Should be randomly scattered around 0. 

If there is any kind of structure/pattern, it suggests there is some other relationship between the response and the predictors, violating the Linearity assumption and compromising the model.


```python
def residual_plot(df, fitted_model=fitted_model):
    df['error'] = fitted_model.resid
    df['fitted_value'] = fitted_model.fittedvalues

    sns.regplot(y=df.error, x=df.fitted_value, lowess=False,line_kws={'color':'red'}, scatter_kws={'alpha':0.5})
    plt.axhline(0, color='gray', ls=':')
    plt.title('Residual Plot');

plt.figure(figsize=(8,4))
residual_plot(df, fitted_model)
```

{%include figure image_path='images\Linear-Regression-Part2\Residual Plot.png'
caption="Plot of Residuals by Fitted Values: useful for Linearity and Error Variance"%}
<br>

The residuals are not randomly scattered around 0. Indeed, there is some kind of pattern/structure. Hence, it violates the Linearity assumption, compromising the model.

One could make use of **data transformation** ([predictor](https://newonlinecourses.science.psu.edu/stat501/node/319/), [response](https://newonlinecourses.science.psu.edu/stat501/node/320/), or [both](https://newonlinecourses.science.psu.edu/stat501/node/321/)) or add new predictors (higher order, interaction term) in order to satisfy the Linearity assumption:

+ **Natural Log transformation**: only for positive values, spread out small values and bring large values closer. Appropriate for skewed distribution.

+ **Reciprocal transformation**: applied on non-zero variable, appropriate for highly skewed data, approximating it to a normal distribution

+ **Square Root**: only for positive values, appropriate for slightly skewed data.

+ **Higher Order**: applied for negative values, appropriate for highly left-skewed distribution.

+ **[Box-Cox technique](https://www.statisticshowto.datasciencecentral.com/box-cox-transformation/)**: this technique allows to systematically test the best parameter $$\lambda$$ in the power transformation. Ultimately, it identifies the best type of transformation to be applied in order to approximate the variable to a normal distribution.

<div text-align:center>
  $$y(\lambda) = \begin{cases}
    \frac{y^{\lambda} - 1}{\lambda},\space \text{if $\lambda \neq 0$}\\
    ln(y), \space \text{if $\lambda = 0$}
  \end{cases}$$
</div>

> **Hint**: when dealing with negative values, add a constant to the variable in order to transform it into positive values before using Box-Cox technique.

It is important to highlight that, after transforming a variable, it is necessary to "transform it back" when calculating Confidence Intervals, and the interpretation of coefficients might change from additive ($$y$$ units change per 1 $$X_i$$ unit increase) to multiplicative ($$y$$ % change due to 1% $$X_i$$ increase).

As a matter of fact, you should not move forward to verify other assumptions before making sure Linearity holds. Otherwise, the model will not yield useful results and will be limited to the scope of data in which it was trained - no extrapolation.

>**Attention**: this post focuses on the **Potential Modeling Problems**. Fixing the Linearity issue is not trivial and we will not address it at this time. <br>
$$\implies$$ A full project will be covered in another post, implementing Data Wrangling, Exploratory Data Analysis (EDA), Feature Engineering, and Model Regression & Performance Evaluation.

<br>
## 4. Correlated Residuals
It frequently occurs if the data is collected over time (stock market) or over a spatial area (flood, traffic jam, car accidents).

The main problem here is failing to account for correlated errors since it allows to improve the model performance by incorporating information from past data points (time) or the points nearby (space).

In case of **serial correlation**, we can implement [ARMA or ARIMA](http://www.statsref.com/HTML/index.html?arima.html) models to leverage correlated errors to make better predictions.

We can check for autocorrelation using two methods:
+ **Durbin-Watson Test**
+ **Residual vs. Order Plot**
+ **Autocorrelation Plot**

<br>
### 4.1) Durbin-Watson Test
We can verify if there are correlated errors by interpreting the [Durbin-Watson test](https://en.wikipedia.org/wiki/Durbin%E2%80%93Watson_statistic) - which is available in the `statsmodels` result summary:
+ verifies if there is serial correlation between one data point ($$t$$) and its previous ($$t$$-1) - called **lag** - over all residuals ($$res.^t$$, $$res.^{t-1}$$)
+ Test statistic varies between [1 - 4]
+ **Test = 2 $$\pm$$ 0.6**: no correlated errors
+ **Test < 1.4**: positive correlation between errors 
+ **Test > 2.6**: negative correlation between errors

>$$\implies$$ In our example, the *Durbin-Watson* test value of **1.979** is quite close to **2**, suggesting there are <u>no auto-correlated errors</u>.

<br>
### 4.2) Residual vs. Order Plot
If the **order**/**sequence** in which the data were collected is known, it is possible to visually assess the plot of Residuals vs. Order (of data collection). 

+ **No Correlation**: residuals are randomly scattered around 0.
+ **Time Trend**: residuals tend to form a positive linear pattern (like a diagonal). One should incorporate "time" as a predictor - and move to the realm of Time Series Analysis.
+ **Positive Correlation**: residuals tend to be followed by an error of the same sign and about the same magnitude, clearly forming a sequential pattern of smoothed ups and downs (like a sine curve). Once again, in this case one should move to Time Series.
+ **Negative Correlation**: residuals of one sign tend to be followed by residuals of the opposite sign, forming a sharp zig-zag line (like a W shape). Just as well, move to Time Series Analysis.

{%include figure image_path='images\Linear-Regression-Part2\Serial Correlation cases.png'
caption="Examples of Autocorrelation between Residuals"%}

<br>
### 4.3) Autocorrelation Plot
If most of residual autocorrelation falls within the 95% Confidence Interval band around 0, no serial correlation.

`from statsmodels.graphics.tsaplots import plot_acf`


```python
from statsmodels.graphics.tsaplots import plot_acf
plot_acf(df.error, lags=30, title='Autocorrelation Plot')
plt.xlabel('Lag value')
plt.ylabel('Correlation');
```

{%include figure image_path='images\Linear-Regression-Part2\autocorrelation plot.png'
caption="Autocorrelation plot showing most lags within 95% CI band, suggesting no serial correlation"%}


In this example, the data is not autocorrelated.

<br>
## 5. Normality of Residuals
Another assumption of Linear Regression, the residuals should be normally distributed around 0. This is important to achieve reliable **Confidence Intervals**.

>This particular problem doesn't significantly impact the prediction accuracy. But it leads to wrong p-values and confidence intervals when checking if coefficients are significantly different than 0. This <u>compromises the model inference capability</u>.

We can verify if normality of residuals is reasonable by analyzing it visually or using statistical tests:
+ **Visually**: 
    - **Histogram**
    - **Probability Plot**
    
Another possibility is that there are two or more subsets of the data having peculiar statistical properties, in which case separate models should be built.

It is also important to highlight this assumption might be violated due to the violation of Linearity - which would call for **transformation**. 

You would transform the response, individual predictors, or both, until the **Normality of Residuals** is satisfied. Note the variables *per se* don't need to be normally distributed: the focus here is on the distribution of **Residuals**.

>**Note**: Afterwards, it is necessary to go back and check if Linearity with the new transformations still holds true.

In some cases, the problem with the error distribution is mainly due to few very large errors (influential data points). Such data points should be further investigated and appropriately addressed, as covered in the **Outliers** and **Leverage Point** section.

<br>
### 5.1) Histogram of Residuals
Visually check if the data is normally distributed (bell shape), skewed, or bi-modal:


```python
sns.distplot(df.error, hist_kws={'edgecolor':'gray'})
plt.ylabel('Density')
plt.title('Histogram');
```

{%include figure image_path='images\Linear-Regression-Part2\residuals histogram.png'
caption="Density Histogram of Residuals, suggesting a Non-Normal distribution"%}


The histogram does not present the well-behaved Bell shape. Hence, this is not a normal distribution. More like bimodal.

<br>
### 5.2) Probability Plot of Residuals
Also known as QQ plot, this approach is better for small sample size.
Different than Histogram, this plot allows to clearly identify values far from the Normal.

Ideally, the points will fall over the diagonal line.

The image below illustrate some of the typical trends of a Probability Plot under specific situations: Normal, Bimodal, Light Tail, Heavy Tail, Left Skew, and Right Skew:

{%include figure image_path='images\Linear-Regression-Part2\Probability Plot - Typical Scenarios.png'
caption="Typical situations for a Probability Plot"%}
<br>


```python
def ProbPlot(df, fitted_model, ax=None):
    from scipy.stats import probplot
    df['error'] = fitted_model.resid
    ...

ProbPlot(df, fitted_model);
```

{%include figure image_path='images\Linear-Regression-Part2\probability plot residuals.png'
caption="Probability plot of Residuals, suggesting a Non-Normal distribution"%}


The points do not follow the diagonal line. As a matter of fact, it is possible to make out an stretched **"S" form** with gaps on the both extremities, suggesting the distribution to be **bimodal**. 

Which seems reasonable when looking at the previous histogram plot.

<br>
### 5.3) Statistical Tests
Some methods testing the Null Hypotheses of data being Normal. If low p-value, suggests data does not likely come from a normal distribution.
+ **Shapiro-Wilk**: `from scipy.stats import shapiro`
+ **D'Agostino**: `from scipy.stats import normaltest`
+ **Jarque-Bera**: `from scipy.stats import jarque_bera`. Recommended only for datasets larger than 2,000 observations. *Already included in* `statmodels` *result summary.*
    
>**Note**: Statistical tests are <u>not recommended</u> for datasets <u>larger than 5,000 observations</u>.

<br>
```python
def Normality_Tests(variable, alpha=0.05):
    from scipy.stats import shapiro, normaltest, jarque_bera
    
    normality_tests = {'Shapiro-Wilk': shapiro, 'D\'Agostino': normaltest,'Jarque-Bera': jarque_bera}
    
    for name,test in normality_tests.items():
        stat, p = test(variable)
        print(f'{name} Test: {stat:.3f} | p-value: {p:.3f}')
        ...

Normality_Tests(df.error)
```
*Shapiro-Wilk Test: 0.978 | p-value: 0.000*<br>
  *--> Reject the Null: not likely Normal Distribution*


*D'Agostino Test: 214.393 | p-value: 0.000*<br>
  *--> Reject the Null: not likely Normal Distribution*
    

*Jarque-Bera Test: 398.418 | p-value: 0.000*<br>
  **Recommended for dataset > 2k observations*<br>
  *--> Reject the Null: not likely Normal Distribution*
  

Since our dataset has over 6,000 observations, the function throws a warning message: statistical tests for normality are not suitable for such large dataset (> 5,000 rows). 

As explained in a previous post, hypothesis testing on large sample sizes leads to even the smallest difference being considered significant. Hence, being unreliable.

<br>
## 6. Constant Variance of Residuals
Another Linear Regression assumption: the variance of errors should not change based on the predicted value. We refer to non-constant variance of errors as **heteroscedastic**. 

>This particular problem doesn't significantly impact the prediction accuracy. But it leads to wrong p-values and confidence intervals, invalidating the basic assumption for Linear Models and <u>compromising the model inference capability</u>.

This can be verified by looking again at the plot of residuals by predicted values. Ideally, we want an unbiased model with **homoscedastic** residuals (consistent across the range of predicted values), as seen in the image of Linearity.

Below, we can see examples of **heteroscedasticity**:

{%include figure image_path='images\Linear-Regression-Part2\Heteroscedastic.png'
caption="Examples of Heteroscedasticity - Non-constant error variance"%}
<br>

To address *heteroscedastic* models, it is typical to apply some transformation on the response, $$y$$ to mitigate the non-constant variance of errors:
+ **Natural Log transformation**: only for positive values, spread out small values and bring large values closer. Appropriate if the variance increases in proportion to the mean.

+ **Reciprocal transformation**: applied on non-zero variable, appropriate for very skewed data, approximating it to a normal distribution.

+ **Square Root**: only for positive values, appropriate if variance changes proportionately to the square root of the mean.

+ **Higher Order**: appropriate if variance increases in proportion to the square or higher of the mean.

+ **[Box-Cox technique](https://www.statisticshowto.datasciencecentral.com/box-cox-transformation/)**: this technique allows to systematically test the best parameter $$\lambda$$ in the power transformation. Ultimately, it identifies the best type of transformation to be applied in order to approximate the variable to a normal distribution.

<div tex-align:center>
  $$y(\lambda) = \begin{cases}
    \frac{y^{\lambda} - 1}{\lambda},\space \text{if $$\lambda \neq 0$$}\\
    ln(y), \space \text{if $$\lambda = 0$$}
  \end{cases}$$
</div>

> **Hint**: when dealing with negative values, add a constant to the variable in order to transform it into positive values before using Box-Cox technique.

<br>
```python
residual_plot(df, fitted_model)
```


{%include figure image_path='images\Linear-Regression-Part2\Residual Plot.png'
caption="Plot of Residuals by Fitted Values: useful for Linearity and Error Variance"%}
<br>

Because the lack of linearity dominates the plot, we cannot use this **Residual Plot** to evaluate whether or not the error variances are equal. 

It would be necessary to fix the non-linearity problem before verifying the assumption of **homoscedasticity**. Since this falls outside the scope of this post, it will not be covered.

<hr>
# Conclusion
In this post we reviewed some of the potential modeling problems regarding the basic assumptions required for Regression models, covering some strategies on how to identify and address them. Namely, 1) Outliers and Leverage Points, 2) Multicollinearity, 3) Linearity, 4) Correlated Errors, 5) Normality of Residuals, and 6) Constant Variance of Errors.

In the next post (Part 3), we will introduce the basics of Logistic Regression (when to use it, how to interpret the results) and how to evaluate Model Performance (accuracy, precision, recall, AUROC *etc.*).
