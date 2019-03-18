---
layout: single
title: 'Weather Trend: Global and Local'
excerpt: >-
  Explores the Weather dataset analyzing trends and patterns in the Global and
  Local average temperature for the past century.
tags: [Data Analysis]
categories: [Project]
date: 2019-02-16T12:38:23.535Z
thumbnail: /images/global-warming.jpg
# classes: wide
author_profile: true
---

## Abstract
This project analyzes the Weather dataset, exploring and identifying patterns for the Local and Global average temperature. In particular, London, UK; Houston, USA; and Rio de Janeiro, Brazil. After some data cleaning, it was possible to evaluate the yearly average temperature for each city, individually, but also compare them against the Global average. There is an overall trend of actual warming after 1920. This pattern becomes quite clear after 1970, where the Global average temperature pretty much grows linearly, <u>suggesting the Global Warming is indeed a real event</u>.

## Introduction
This project was completed as part of the course requirements of Udacity's [Data Analyst Nanodegree](https://www.udacity.com/course/data-analyst-nanodegree--nd002) certification.

The complete code and analysis rationale can be found in the [jupyter notebook](https://github.com/marcellovictorino/DAND-Project0/blob/master/Project%200%20-%20Weather%20Trend.ipynb).

>**Limitation**: the dataset analyzed is simple. It only contains data on the yearly average temperature, in Celsius. Further weather-related data was not made available, such as: humidity, wind, and precipitation. 


## Data Wrangling
The first step was to query the database using SQL to extract weather data for the cities I have lived in, and the globe:
```SQL
SELECT * FROM city_data WHERE city = 'London' AND country = 'United Kingdom';
SELECT * FROM city_data WHERE city = 'Houston' AND country = 'United States';
SELECT * FROM city_data WHERE city = 'Rio De Janeiro' AND country = 'Brazil';
SELECT * FROM global_data
```
### Dealing with noisy data
The Global weather data has information ever since 1750! But this data, as expected,has an awful lot of variation (noise), mostly due to ups and downs in the average temperature along the time.

To smooth out the noise, but still keeping it sensitive enough to capture underlying nuances, I implemented the moving average of 10 years:

```python
df_global['Global | MA 10 years'] = df_global.Global_Temperature.rolling(window=10).mean()
```

<div class="row">
  <div class="col-md-6">
    <img src="{{ site.url }}{{ site.baseurl}}\images\weather-trend\global-weather-noisy.png" alt="global weather - noisy data" style="width:100%">
  </div>
  <div class="col-md-6">
    <img src="{{ site.url }}{{ site.baseurl }}\images\weather-trend\global-weather-moving-average.png" alt="Forest" style="width:100%">
  </div>  
</div>

### Missing Data



## Technologies Used

+ **Python**:
  + Pandas, Numpy, Matplotlib, Statmodels, Jupyter Notebook
+ **SQL**

## Key Findings

+ **London, UK** is consistently hotter than the Global average, with a mean temperature of 9.5 and 8.3 degrees Celsius, respectively
+ **London** presents the same trend as the Global average temperature: they both seem to be getting warmer recently. 
+ There is a pattern of ups and downs in the average temperature every 10 years... up until around 1920. After this, the average temperature presents an overall trend of actual warming. This becomes quite clear after 1970, when the Global average temperature pretty much grows linearly!
+ A simple Linear Regression using only data after 1920 yielded a good model (R-value of 94%) that can be used to predict **London's** temperature. With a slope of 1.1 and an intercept of only 0.09, it basically means that London is usually 10% hotter than the Global yearly average temperature
+ The same warming trend holds true for **Houston** and **Rio de Janeiro**. As expected, Rio and "Hell-stoun" are significantly warmer than London and the Global average, with 23.8 and 20.2 mean degrees Celsius, respectively.