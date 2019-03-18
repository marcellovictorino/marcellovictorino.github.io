---
title: Portfolio
layout: collection
permalink: /portfolio/
collection: portfolio
entries_layout: grid
classes: wide
---

{% assign loopindex = 0 %}
{% for post in site.posts %}                       
  {% if post.categories contains 'Project' %}
    {% assign loopindex = loopindex | plus: 1 %}
    {% assign rowfinder = loopindex | modulo: 3 %}
    {% if rowfinder == 1 %}
<div class="row">
    <div class="col-md-4">
		<a href="{{ post.url }}"><img src="{{ post.thumbnail }}"/></a>
		<a href="{{ post.url }}"> <h4>{{ post.title}}</h4></a> 
		<small>{{ post.excerpt }}</small>
	</div>
    {% elsif rowfinder == 0 %}
    <div class="col-md-4">
		<a href="{{ post.url }}"><img src="{{ post.thumbnail }}"/></a>
		<a href="{{ post.url }}"> <h4>{{ post.title}}</h4></a> 
		<small>{{ post.excerpt }}</small>
	 </div>
</div>
    {% else %}
    <div class="col-md-4">
		<a href="{{ post.url }}"><img src="{{ post.thumbnail }}"/></a>
		<a href="{{ post.url }}"> <h4>{{ post.title}}</h4></a> 
		<small>{{ post.excerpt }}</small>
	</div>
    {% endif %}
  {% endif %}
{% endfor %}

{% if rowfinder != 0 %}
</div>
{% endif %}