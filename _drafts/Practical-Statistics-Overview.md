---
title: Practical Statistics Overview
excerpt: Brief Overview of Hypothesis testing, covering how to set up the Null and Alternative Hypothesis, difference between One-side versus Two-sided tail test, how to interpret the p-value and why you should be using Confidence Interval; the Central Limit Theorem; the Law of Large Numbers; the Danger of Multiple Tests; and why "size matters" (Practical instead of purely Statistical Significance).
tags: [Data Analysis]
categories: [Post]
date: 2019-03-16 00:00:00 +0000
thumbnail: "/images/ab-testing.jpg"


---
## Abstract
Brief Overview of Hypothesis testing, covering: 1) how to set up the Null and Alternative Hypothesis; 2) difference between One-sided versus Two-sided tail test; 3) how to interpret the p-value and why you should be using Confidence Interval; 4) the Central Limit Theorem and the Law of Large Numbers; 5) the Danger of Multiple Tests and why "size matters" (Practical instead of purely Statistical Significance); 7) Simpson's Paradox; 8) Bayes (Conditional Probability)

## Introduction
This ambitious "overview" will not focus on the mathematics, but rather the core concepts and the applications of "Practical Statistics".

The complete code and analysis rationale can be found in [this Jupyter notebook](add link here).

>**Limitation**: the summary overview here presented are my own take on tricky subjects, serving as a reference reading for statistics concepts. External reference sources will be specified.

# Keywords

+ **Inferential Statistics**: based on data from a sample (statistic), draw conclusions about a population (parameter)

+ **Confidence Interval**: based on a specific confidence level, represents the interval that captures the estimated parameter (Two-tailed, think $$\alpha/2$$)

+ **Law of Large Numbers**: bigger sample size will result in a *statistic* that more closely represents the Population *parameter*

+ **Central Limit Theorem**: with a large enough sample size and several samples, we can expect the Sampling Distribution to be Normally Distributed around the mean value.
This applies to mean, proportion, difference in means, and difference of proportions.

+ **Sampling Distribution**: represents the frequency of each observed statistic, for all samples. With a large enough amount of samples (draw from the same population)

+ **Bootstrapping**: simulates the creation of a sampling distribution. Which allows to estimate the Population *parameter*



## Bootstrapping + Confidence Interval = Practical Hypothesis Testing
Generates a synthetic sample, drawing S observations - with replacement - from the original sample. Repeating this process several times allows to simulate the creation of a sampling distribution. Which captures the "characteristics" of the original sample.

```python
df_older21 = df.query('age ">=21" ').height
df_less21 = df.query('age "<21" ').height

bootstrap_older21 = np.random.choice(df_older21, (1000, 200)).mean(axis=1)
bootstrap_less21 = np.random.choice(df_less21, (1000, 200)).mean(axis=1)

delta_height = bootstrap_older21 - bootstrap_less21
plt.hist(delta_height)
```
{% include figure image_path="" caption="Sampling distribution of difference of means" alt=caption %}

# insert image from sampling distribution

Therefore, after using the Bootstrap technique to obtain our normal-ish sampling distribution, we can simply find the interval where we would be 95% confident contains the investigated statistic.

Considering level of confidence of 95%, we have $$\alpha = 0.05$$. Since we want to find the Confidence Interval, we need to determine the lower and upper bounds. Which would be [ percentile($$\frac{\alpha}{2}$$) , percentile(1- $$\frac{\alpha}{2}$$) ]:

```python
lower, upper = np.percentile(delta_height, 2.5), np.percentile(delta_height, 97.5)
print(f'Confidence Interval: [{lower:.3f}, {upper:.3f}]')
```

In our example, we can interpret the results as:
>Since the Confidence Interval does not contain (overlap) zero, there is evidence that suggests there is indeed a difference in the population height of those sub-groups. We can say, with 95% statistical confidence, on average, coffee drinkers are taller.

### Decision Errors (False Positives & False Negatives)
Whenever making decisions without knowing the correct answer, we can face 4 different outcomes. Enters the **Confusion Matrix**:

{% include figure image_path="images\Statistics-Overview\confusion-matrix.PNG" caption="Confusion Matrix" alt=caption %}

+ **Type I error (False Positive):**
    - Reject the Null when it is actually true
    - The probability of committing a Type I error is represented by $$\alpha$$ (also know as Significance Level)
    - Not committing it ($$1 - \alpha$$) is know as **Confidence Level**

+ **Type II error (False Negative):**
    - Fail to reject the Null when it is indeed false
    - Probability of committing a Type II error is represented by $$\beta$$
    - Not committing it ($$1 - \beta$$) is known as the **Power** of the test

Common values used in practice are:
+ **Significance Level** ($$\alpha$$) = 5%
+ **Test Power** ($$1 - \beta$$)= 80%

Differentiating between Type I (False Positive) and Type II (False Negative) errors can be confusing... until you see this image:

{% include figure image_path="images\Statistics-Overview\decision-errors2.jpg" caption="Decision Errors explained" alt=caption %}



## Hypothesis Test
Help making better and data-informed decisions.
1. Translate the research question into 2 clear and competing hypothesis: $$H_0:$$ **Null** and $$H_1:$$ **Alternative**
2. Collect data (Experiment Design: ideal sample size, how long to conduct, consistency among control and experiment groups *etc.*) to evaluate both hypothesis.

+ We assume the Null, $$H_0$$, to be true before even collecting the data (prior belief)
+ $H_0$ always holds some sign of equality ($$\leq = \geq$$)
+ The Alternative, $$H_1$$, is what we would like to prove to be true
+ $$H_1$$ holds the opposite sign ($$> \neq < $$)

### Setting up
It is usually easier to start defining the Alternative Hypothesis and what sign it should hold (Is it bigger? Is it different? Is it smaller?). Ergo, the Null assumes the opposite sign.

### One vs Two-sided Tails




### Interpreting the p-value


### Danger of Multiple Tests


### Caution: Size matters
With a large sample size, hypothesis testing leads to even the smallest differences being statistically significant. But not necessarily practical significance.
It is important to always take into consideration extraneous factors to guide the decision making (cost, budget/time constraint *etc.*)
