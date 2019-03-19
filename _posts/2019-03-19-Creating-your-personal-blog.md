---
title: 'Create your own Personal Blog!'
excerpt: >-
  Summary of main steps to build your own personal blog using Jekyll on Windows, free hosting on Github Pages, and Content Manager from Netlify.
tags: [Programming] # Programming, Data Analysis, Data Science, Transportation
categories: [Post] # Post, Project
date: 2019-03-19T08:20:00.535Z
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
>"Teach someone and you will learn twice!" (proverb)

Sharing the knowledge - especially about tricky subjects you have recently put some thought on - is always a good thing. 
Recommended by many in the Data Science field, it's a way of increasing your online visibility. Creating your own "brand", so future employers might have a way to know the quality of your work and personality.
But it also helps your future self, in case you ever need to tackle a similar challenge again.

Well, creating this website was one of those things. After an exciting weekend researching about the topic, getting inspiration from other people's website, and watching/reading some awesome tutorials... I decided to write up my own take on how to get this done. Especially for Windows users!

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

<hr>

### 1.A) Install Git
I am not going to insult your intelligence. Just go ahead and install [Git](https://git-scm.com/downloads) in your local machine - even if you have no idea what it is or what it does.

<br>

### 1.B) Create your own Github repository
You will need to create a Github account. <br>
Create a new repository using this specific nomenclature: <br>`your-github-username.github.io` <br>
Don't worry about Readme.md or Gitignore at this step. <br>
Click on ```Clone or Download``` button; copying the url link to this repository. <br>
Create a new Folder on your local machine, where you will keep all the project files.<br>
Open the Git bash in your project folder (many ways to do this). 
>**Pro tip**: right-click inside your project folder, and select `Git Bash Here`

Enter the command (you can just paste the repository link):
```
> git clone <your-repository-link-here>
```
Bam! You just created a connection between your project folder and your online Github repository. Later, Github Pages will host your website based on this repository content!

### 1.C) Get Minimal Mistakes Theme
Similar to the previous step, but this time all you want is the project files. <br>
Go to the [Minimal Mistakes Repository](https://github.com/mmistakes/minimal-mistakes) and download the ZIP containing all files. <br>
Extract everything into your project folder. <br>

### 1.D) Clean the House
Let's remove the unecessary files and folders. Here is a list of everything to delete (by the time this post was written. Might change in the future):
+ **Folders**: **.github**; docs; test
+ **Files**: .editorconfig; .gitattribute; .travis.yml; CHANGELOG.md; screenshot-layouts.png; screenshot.png

Extra attention on deleting the folder **.github**, since you want to keep the connection to your own repository, as accomplished on the step **1.B**. <br>
Now simply copy all the remaining files and paste them into your project folder.

### 1.E) Test It!
At this stage you should have everything ready for a working prototype of the website. Let's save all these new files into your own Github repository. Enter these following commands (remember how to get to the Git Bash?):
```
> git add .
> git commit -m "website prototype"
> git push
```

> Note: for those who just installed Git... these three commands represent the very basic workflow of version control. Basically, you are *staging* everything you have been working so far (git keeps track of **changes**. Think of new files/folders or adding/removing text from files). <br>
Then you *commit*, grouping all changes so far and giving it a meaningful but brief description (what you see under quotes) <br>
Finally you *push* all these changes, applying them to your repository.

Okay. Give it a minute. Github Pages auto deploy the website (if using this specific repository nomenclature) after a new *push* has been made.<br>
You should be able to open your favorite browser and navigate to `your-github-username.github.io` to check whether a very simple website loads or not.

### 2.A) General Customization
Alright, if you wanted to build your own website, you must have know this time would come! Here is where things start to get *technical*. If you want to follow along the more **Simplistic** approach, just suck it up and follow the next instructions. <br>
But if you want to stay in wonderland and see how far the rabbit hole goes... bear with me. More advanced customizations and template logic will be covered in the next post (wait for Part 2).

The `_config.yml` file holds general information used to build your personal website. Open it with your favorite text editor (i.e. VS Code). Most fields are self-explanatory or have some sort of instruction. <br>
Here is the very minimal you should change:
+ **Site Settings**: locale; title; name; description; url; repository
+ **Site Author**: name; avatar (path/to/profile-picture); bio; location. I recommend adding your github and linkedin links as well
+ **Outputting**: timezone

And here is what you should <u>ADD</u>:
+ **Under Reading Files / include**: - _pages
<img src="{{site.url}}\images\creating-personal-blog\config-look-for-pages.PNG">
+ **Under Defaults / defaults**: # _pages
<img src="{{site.url}}\images\creating-personal-blog\config-pages-default.PNG">

Feel free to play around with the other fields: if something breaks, just undo it!

### 2.B) Creating Pages
Now go ahead and create a new folder named "_pages" (inside your project folder).
This folder will hold the markdown (or html) files containing: 1) metadata; and 2) content of each page.

There are many cool things you can accomplish using some theme logic (called *Liquid*) to auto generate the content of your pages. But for Part 1 of this tutorial, let's K.I.S.S (*keep it simple, stupid*).

#### About page
Under the newly created folder `_pages`, create the file `about.md` and edit it with your favorite text editor (the extension stands for markdown format).

```
---
title: "About"
permalink: /about/
header:
    image: "/images/banner_about.jpg"
classes: wide
---

Enter your bio description here. Who you are, your interests... afterall, this is your personal blog. Just keep it short: abround 3 paragraphs should be enough.
```

The first section, separated by the `---`, represent the metadata. There are many keywords with special functionality here. Feel free to read the [Minimal Mistakes Documentation](https://mmistakes.github.io/minimal-mistakes/docs/quick-start-guide/) for further options.

The second section, after the closing `---`, is the page content. This is very similar to the post structure. You can add Markdown (or html tags) for the content.
>Obs: the **permalink** defined here is where the page will "live" in your site.

#### Adding Page to the Navigation Bar
In order to be able to navigate to your new page, you must add the url via a link... or place it in the Navigation Bar - which is way cooler!

Under the folder `_data`, edit the file `navigation.yml`.
> You are probably starting to understand how things work behind the scenes. It is all about knowing <u>what</u> to change and <u>where</u> to find <u>which</u> file.

Just uncomment the code by removing the `#`. Make sure to use the exact same **permalink** as specified in the metadata of the page file.


### 2.C) Creating your first Post!
Similar to the last step, let's first create a folder named `_posts` under your project folder. <br>
This is where all your posts will be stored. Then, create a file with the specific format `YYYY-MM-DD-Post-Title-Here.md` and edit it:
```
---
title: 'My first post!'
excerpt: >-
  Concise description of your post, making sure to add keywords to help Search Engines like Google find your post.
date: 2019-03-19T08:20:00.535Z
thumbnail: /images/post-image-here.jpg # 
toc: true # generate Table of Content based on Headings
classes: wide # use the reserved right-sidebar for content
---
# Main Title (AKA Heading 1 or <H1>)
Some text here. Lorem Ipsum Bacon Muitom Flavourfum.

Consectetur adipiscing elit, sed do **eiusmod tempor** incididunt ut labore et dolore magna aliqua. Non arcu risus quis varius. Id leo in vitae turpis massa sed elementum. 
+ Mauris cursus mattis molestie a iaculis at erat pellentesque adipiscing.
+ Ac odio tempor orci dapibus ultrices in iaculis nunc.

> Viverra justo nec ultrices dui sapien eget mi. Pellentesque diam volutpat commodo sed. Viverra nam libero justo laoreet sit amet cursus sit. In fermentum et sollicitudin ac.
```
The snippet above contain some examples for the metadata (dont worry about tags/categories for now) and the content markdown. You can find more in the [Markdown Documentation](https://www.markdownguide.org/cheat-sheet/) page.

### 2.D) Customize your Landing Page
Just so your website starts to look flashier, let's also add some bonus costumization to your landing page. <br>
Edit the file `index.html` found in you project folder:
```
---
layout: home
title: "My Personal Blog"
author_profile: true
header:
    image: "/images/cool-good-looking-image.jpg"      
classes: wide
---

<H1>Welcome to my Personal Blog!</H1>
```
As you can see, the **layout** is specific for the homepage. We are also adding a nice header image, to make your site look sharp. I recommend using images that are banner-like (small, but wide) and with 1280 px resolution.

### 2.E) Deploy it!
Now that you have made some significant changes in your project structure, it is time to `stage, commit, push` them to your repository. If everything works like it hsould, Github Pages will auto deploy your personal website.<br>
Head to `your-github-username.github.io` to see what it looks like!


## Technologies Used
+ Ruby: technology behind Jekyll. Uses "gems", similar to packages/libraries
+ Jekyll: framework for generating static website based on a set project structure
    - Minimal Mistakes: minimalistic theme built on top of Jekyll (thanks, Michael Rose!!!)
+ Netlify: for Content Manager (and hosting)
+ Markdown: use your favorite Markdown editor for writing content
+ Github: repository to keep all the website files
    - Github Pages: auto deploy your static website, giving the correct settings (covered in this post)