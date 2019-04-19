---
title: 'Weather Trend: Global and Local'
excerpt: >-
  Explores the Weather dataset analyzing trends and patterns in the Global and
  Local average temperature for the past century.
tags: [Data Analysis]
categories: [Project]
date: 2019-02-16T12:38:23.535Z
thumbnail: \images\Project0-Weather-Trends\global-warming.jpg
toc: true
classes: wide
---
# Introduction

This project was completed as part of the course requirements of Udacity's [Data Analyst Nanodegree](https://www.udacity.com/course/data-analyst-nanodegree--nd002) certification.

The complete code and analysis rationale can be found in this [browser friendly version](https://github.com/marcellovictorino/DAND-Project0/blob/master/Project%200%20-%20Weather%20Trend.html) or the actual [jupyter notebook](https://github.com/marcellovictorino/DAND-Project0/blob/master/Project%200%20-%20Weather%20Trend.ipynb) or this .

## Overview

The global and local weather data were made available in a database, requiring some SQL commands to select and extract data for the desired locations. This project explores the correlation of local and global weather, as well as their trend over the past century.



## Technologies

+ **Python**:
  + Pandas, Numpy, Matplotlib, Statmodels, Jupyter Notebook
+ **SQL**



## Key Findings

+ **London, UK** is consistently hotter than the Global average, with a mean temperature of 9.5 and 8.3 degrees Celsius, respectively
+ **London** presents the same trend as the Global average temperature: they both seem to be getting warmer recently. 
+ There is a pattern of ups and downs in the average temperature every 10 years... up until around 1920. After this, the average temperature presents an overall trend of actual warming. This becomes quite clear after 1970, when the Global average temperature pretty much grows linearly!
+ A simple Linear Regression using only data after 1920 yielded a good model (R-value of 94%) that can be used to predict **London's** temperature. With a slope of 1.1 and an intercept of only 0.09, it basically means that London is usually 10% hotter than the Global yearly average temperature
+ The same warming trend holds true for **Houston** and **Rio de Janeiro**. As expected, Rio and "Hell-stoun" are significantly warmer than London and the Global average, with 23.8 and 20.2 mean degrees Celsius, respectively.
