---
title: A/B Testing Summary
excerpt: Summary of A/B testing with an application example, handling for multiple
  tests correction using Holm-Bonferroni.
tags:
- Data Analysis
date: 2019-03-16 00:00:00 +0000
thumbnail: "/images/ab-testing.jpg"
rating: 4

---
## Difficulties to bear in mind

* **Novelty Effect** and **Aversion Change** when existing users first experience a change
* In case of **multiple tests**, requires correction (i.e. _Holm-Bonferroni_)
* Sufficient `traffic`and `conversions`  to have significant and repeatable results
* Best metric choice for making the ultimate decision (i.e. measuring $ revenue, not # clicks)
* Long enough runtime to account for changes in user behavior
* **Practical significance** of results (Benefit/Cost analysis)
* Consistency among test subjects in **Control** and **Experiment** groups (imablance in population represented in each group can lead to _Simpsons's Paradox_ situations)

Adding an image example:
<img src="{{ site.url }}{{ site.baseurl }}/images/logo_victorino3_77x88.png" alt="logo_image">

```python
import pandas as pd
import numpy as np
```

Inline code: `np.random.normal()`

Math example: $$H_0: \mu = 0$$

## Header2

### Header 3

#### Header 4