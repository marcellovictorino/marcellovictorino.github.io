---
title: Project 5 - Ford GoBike 2017
excerpt: Data wrangling and systematic Exploratory Data Analysis of the Bike-share system, Ford GoBike (+3 million observations). Univariate analysis to identify outliers, as well as performing transformations on highly skewed data. Bi-variate and Multivariate analysis investigating the relationship between all features and uncovering interesting insights from the data.
tags: [Data Analysis]
categories: [Project]
date: 2019-07-01 00:00:00 +0000
thumbnail: \images\Project5-FordGoBike\Thumbnail.jpg
---

$$\implies$$ <a href="\images\Project5-FordGoBike\2_Presentation.slides.html" target="_blank"><b>Summary Presentation</b></a>

## Dataset

The Ford Go Bike is the Bike-share system operating in the Bay Area, San Francisco (USA). With around 6.400 bikes in more than 360 stations across San Francisco, East Bay and San Jose, there are basically two types of subscription:

1. Subscriber: Membership can be Monthly ($15/mo) or Annual ($150/yr, equivalent to $12.50/mo). It grants unlimited 45-minutes trips
2. Customer: 
   + Single ride ($2 per trip) and 
   + Access Pass ($10), granting unlimited 30-minute rides within 24 hours.

The [dataset](<https://www.fordgobike.com/system-data>) is open to the public, containing station and user data for each trip. At the time of this project, is was possible to download historic data from January, 2017 up to April, 2019.

The raw data was spread out into 17 files. After appending all into a single data frame, I performed some Data Wrangling, summarized in the following:
### Data Wrangling
+ **Missing values**: there were around 220 thousand observations missing data for user gender and birth year. Instead of applying imputation techniques (median and mode for numerical and categorical variables, respectively), I decided to drop these rows.
+ **Bike Share For All Trip**: this variable is not present in the 2017 data. I filled all 2017 missing data as 'No', since the special pricing program for qualifying low-income users in the Bay area wasn't in place at the time.
+ **Station Data:** to improve performance, I created a new table holding unique values for the station ID and their name, removing the Start and End station name from the main dataset. It is possible to retrieve this information afterward by merging the data on the unique station ID.
+ **Data Type**: 
  + 'station_id' stored as string instead of float
  + 'member birth year' stored as integer instead of float
  + end and start time stored as date-time

### Feature Engineering

After dealing with data integrity issues, I transformed some variables into a more meaningful format and extracted the components of the start time variable:

+ Transformed member gender into binary variable, where 1: Male, 0: Female.
+ Transformed User Type into binary, where 1:Subscriber (Annual Pass), 0: Customer (Single Ride or Access Pass)
+ Transformed Low Income Trip into binary, where 1: Yes, 0: No.
+ Transformed trip duration from seconds to minutes.
+ Calculated user age at time of bike rental.
+ Extracted from Start Time of rental: year, month, week, day, day of week, and hour

After this process, the final dataset has over 3 million observations with 19 columns, containing information over the initial and final bike station, time of start of the trip, duration in minutes; as well as details over the user, such as gender, age, and member status (subscriber or not).


## Main Findings
+ **Correlation**: after calculating the Person R for each pair of variables, it was not possible to identify any strong correlation between them;

+ **Short Trips**: even though most trips are taken by subscribers - who have access to unlimited 45-minutes trips - the vast majority are short, taking between 5 and 15 minutes;

+ **Commuting Pattern**: most bike-share systems in the world are vastly used by tourist as a cheap albeit good alternative to travel around and get to know the city they are visiting. But in this case, it is possible to identify a clear commuting pattern, with trips concentrated around 8am and 5pm;

+ **Typical behavior for each type of user**: one would think that, since subscribers have access to unlimited 45-minutes trips, there would be many trips of longer duration. But in this case, it is possible to see that subscribers actually tend to use the system for commuting purposes, with the vast majority of trips taking place during weekdays, around 8am and 5pm.


## Key Insights for Presentation

Out of the four main insights identified during the EDA, I chose to focus the Explanatory Analysis on two, due to their relevance and counter-intuitive nature: **Usage Pattern** and **Typical behavior for each type of user**.

For the first one, since I wanted to highlight the concentration of trips during specific hours of the day, I opted for a simple bar chart, where the bar height represent the proportion of trips and the x-axis had all 24 hours of the day in ascending order. I emphasized the **comparison aspect** of the analysis by using different colors to represent Weekday and Weekend data. 

On the second one, I chose side-by-side plots where it would be easy to compare aspects of Subscribers versus Sporadic users. Since I wanted to display the distribution of a pair of numerical variables ([trip duration, hour] & [trip duration, age]), I first tried to implement a basic scatter plot. But since there were too many points overlapping each other, not even applying transparency or jittering improved. Hence I decided to implement the heatmap as a good alternative.
