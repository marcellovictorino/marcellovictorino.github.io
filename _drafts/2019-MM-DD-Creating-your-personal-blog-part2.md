---
title: 'Create your own Personal Blog - Part2'
excerpt: >-
  More advanced functionalities for your own personal blog using Jekyll on Windows, free hosting on Github Pages, and Content Manager from Netlify.
tags: [Programming] # Programming, Data Analysis, Data Science, Transportation
categories: [Post] # Post, Project
#date: 2019-03-20T08:20:00.535Z
thumbnail: /images/global-warming.jpg # Get new image
toc: true # generate Table of Content based on Headings
classes: wide # use the reserved right-sidebar for content
---
## Abstract
Two-parts tutorial on how to create your own personal <u>responsive</u> website (blog) using a <u>static site generator</u>. This post covers *Jekyll* (site generator), *Minimal Mistakes* (amazing Jekyll Theme), *Github Pages* (free hosting) and *Netlify* (Content Manager). Two approaches are covered: 1) **Simplistic**, no installation required; or 2) **Developer Mode**, allowing to preview changes by deploying the website on the localhost. Both provide amazing results, thanks to how the listed technologies work together like a charm! Special thanks to Michael Rose, developer behind the awesome open source Jekyll theme used in this tutorial (and in this website!).

LINKS
[Minimal Mistakes Documentation](https://mmistakes.github.io/minimal-mistakes/docs/quick-start-guide/)
[Font Awesome (free icons)](https://fontawesome.com/icons?d=gallery&s=solid&m=free)
[Awesome Youtube Tutorial (by Data Optimal)](https://www.youtube.com/watch?v=qWrcgHwSG8M)
[Repository: Minimal Mistakes](https://github.com/mmistakes/minimal-mistakes)
[Ruby Installer (go for the most stable DevKit version for Windows)](https://rubyinstaller.org/downloads/)
[Jekyll Documentation](https://jekyllrb.com/docs/)

## Introduction
This is Part 2 of the Tutorial on "How to Create your own personal website". It builds on top of the previous one, showing more advanced functionalities for those willing to dig deeper into the code.

#TODO: continue from here
This tutorial will cover two approaches to acomplish the same goal. Feel free to pick one and follow along:
1. **Simplistic**: doesn't require any installation. Just copy-and-paste, following along this tutorial instructions. If you are not interested in understanding what is going on and just wants your own personal blog right away.
2. **Developer Mode**: a little bit painfull, requiring to install ruby and all necessary gems (packages) in your own local machine. But allows for easy deploy on localhost, so you can preview your changes, testing them multiple times until happy with the results. Then you would push the changes so Github performs its magic and auto deploy your site.

This tutorial builds on top of this [awesome Youtube Tutorial (by Data Optimal)](https://www.youtube.com/watch?v=qWrcgHwSG8M). Feel free to use both resources!

## Basic Steps
Similar to both approaches, you will need to:
+ 1.A) Have Git installed in you localmachine
+ 1.B) Create your own Github repository
+ 1.C) Copy all files from the Minimal Mistakes Theme repository
+ 1.D) Make some changes and keep only necessary files
+ 1.E) Test if it works (basic working website)
+ 2.A) Customize your Website - well, some of it
+ 2.B) Create Pages
+ 2.C) Create Posts
+ 2.D) Get ready to deploy it!


### 1.A) Install Git





---
## Technologies Used
+ **Ruby**: technology behind Jekyll. Uses "gems", similar to packages/libraries
+ **Jekyll**: framework for generating static website based on a set project structure
    - *Minimal Mistakes*: minimalist theme built on top of Jekyll (thanks, Michael Rose!!!)
+ **Netlify**: for Content Manager (and hosting)
+ **Markdown**: use your favorite Markdown editor for writing content
+ **Github**: repository to keep all the website files
    - **Github Pages**: auto deploy your static website, giving the correct settings (covered in this post)