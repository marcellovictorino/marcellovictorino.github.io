---
title: Portfolio
layout: collection
permalink: /portfolio/
collection: portfolio
entries_layout: grid
classes: wide
---

<ul>
	{% for post in site.posts %}
	<li>
		<a href="{{ post.url }}">{{ post.title}}</a> 
		<br>
		<small>{{ post.excerpt }}</small>
	</li>
	{% endfor %}
</ul>


