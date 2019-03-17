---
layout: archive
permalink: /projects/
title: "Data Science Projects"
author_profile: true
header:
    image: "/images/banner_equations.jpg"
---

{% include group-by-array collection=site.posts field="tags" %}

{% for tag in group_names %}
  {% assign posts = group_items[forloop.index0] %}
  <h3 id="{{ tag | slugify }}" class="archive__subtitle">{{ tag }}</h3>
  {% for post in posts %}
    {% include archive-single.html %}
  {% endfor %}
{% endfor %}

