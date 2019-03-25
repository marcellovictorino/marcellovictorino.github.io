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

+ Bootstraping: 
    - Aims at estimating a parameter of the population (specifically, mean and proportion).
    - Do not have the ability to infer/make decision on an individual level (use ML models instead)
+ Larger Sample Size: Reduce $$CI_{width}$$
+ Increasing Confidence Level (i.e. 95% -> 99%): Increase $$CI_{width}$$
+ Margin of Error: $$CI_{width} / 2$$
+ If the CI does not contain 0, evidence suggests statistically significance. 
+ The farther away from zero, the more relevant. 
+ But always bear in mind the **Practical Significance**. 


## Hypothesis Test
Help making better and data-informed decisions.
1. Translate the research question into 2 clear and competing hypothesis: $$H_0:$$ **Null** and $$H_1:$$ **Alternative**
2. Collect data (Experiment Design: ideal sample size, how long to conduct, consistency among control and experiment groups *etc.*) to evaluate both hypothesis.

+ We assume the Null, $$H_0$$, to be true before even collecting the data (prior belief)
+ $$H_0$$ always holds some sign of equality ($$\leq$$  $$=$$  $$\geq$$)
+ The Alternative, $$H_1$$, is what we would like to prove to be true
+ $$H_1$$ holds the opposite sign ($$>$$  $$\neq$$  $$<$$)

### Setting up
It is usually easier to start defining the Alternative Hypothesis and what sign it should hold (Is it bigger? Is it different? Is it smaller?). Ergo, the Null assumes the opposite sign.

### One vs Two-sided Tails




### Interpreting the p-value


### Danger of Multiple Tests
There is **always** the chance of randomly finding data supporting the Alternative. Even if the p-value is quite small!

For the usually accepted Significance Level of 5%, it basically means that if we perform the same experiment 20 times, we expect that one of them will result in a False Positive... and we are okay with it!

Whenever replicating the same experiment, or conducting multiple tests (i.e. A/B testing using more than one metric), we need to watch for Type I error compound!

Enter the Correction Methods for multiple tests:
+ **Bonferroni**: $$\alpha^* = \frac{\alpha}{n}$$
    - The Type I error rate should be the desired $$\alpha$$, divided by the number of tests, $$n$$.
    - Example: if replicating the same experiment 20 times, each experiment correct $$alpha$$ would be $$0.05 / 20$$ = 0.0025 (that means, 0.25% instead of the initial 5%)
    - As illustrated in the example above, this method is **very conservative**. While it does minimize the rate of False Positives (making the findings more meaningful), it also fails to recognize actual significant differences. Hence, increasing Type II error (False Negatives), leading to a low test **Power**.

+ **Holm-Bonferroni**: recommended approach
    - Less conservative, this adaptation from the Bonferroni method presents a better trade-off between Type I and II errors.
    - It consist of a simple but tricky to explain algorithm. See [this Wikipedia] for detailed explanation.
    - I personally created a Python function to implement this method. It receives a list of the identification of the test/metric/experiment, and another with the respective p-value of each individual test/experiment. The output is a list of all tests that have evidence to Reject the Null; and another one with the remaining, where we Fail to Reject the Null.
    - See [this specific post](TODO:add_link) for detailed code and example.

### Caution: Size matters
With a large sample size, hypothesis testing leads to even the smallest differences being statistically significant. But not necessarily practical significance.
It is important to always take into consideration extraneous factors to guide the decision making (cost, budget/time constraint *etc.*)



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
