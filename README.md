# How-I-Analyzed-Your-Mother
## Capstone Project

## Introduction

The purpose of this project is to do a thorough exploratory data analysis of the characters and scripts of the TV show How I Met Your Mother, as well as create a model that can accurately classify who said which line.

Following the Data Science process of OSEMN, outlined below, a multi-class classifier with an accuracy of 32% was built. While this is not great, due to time constraints this was the final model choice; details about ways to improve this if there was more time are included below.

## The Data Science Process: OSEMN

### Obtain

The data is from Forever Dreaming Transcripts and contains over 22,000 lines from each character throughout the series.

### Scrub

Text preprocessing was done here, as this was a Natural Language Processing project. The dataset, once cleaned, was remotely turned into an SQL database and reuploaded to the notebook for exploration.

### Explore

There were so many ways to look at this data, here are a few of the things I looked at:

  * Total Lines during Series
  * Largest Vocabulary
  * Character Mentions
  * Common Phrases
  * Most Used Terms
  * Sentiment Analysis

I also created visualizations for each of these avenues I explored:

* Total Lines during Series

<img src="https://raw.githubusercontent.com/srobz/How-I-Analyzed-Your-Mother/main/Visuals/TotalLines.png" height="300">

* Largest Vocabulary

<img src="https://raw.githubusercontent.com/srobz/How-I-Analyzed-Your-Mother/main/Visuals/Vocabulary.png" height="300">

* Character Mentions - Total and by Nickname

<img src="https://raw.githubusercontent.com/srobz/How-I-Analyzed-Your-Mother/main/Visuals/CharMentions.png" height="300"> <img src="https://raw.githubusercontent.com/srobz/How-I-Analyzed-Your-Mother/main/Visuals/Nicknames.png" height="300">

* Common Three Word Phrases: Three word phrases are where differences started to become more apparent among the characters.

<img src="https://raw.githubusercontent.com/srobz/How-I-Analyzed-Your-Mother/main/Visuals/ThreePhrase.png" height="600">

### Word Clouds

* Total Corpus

<img src="https://raw.githubusercontent.com/srobz/How-I-Analyzed-Your-Mother/main/Visuals/CorpusWC.png" height="300">

* Ted - Marshall - Barney - Lily - Robin

<img src="https://raw.githubusercontent.com/srobz/How-I-Analyzed-Your-Mother/main/Visuals/TedCloud.png" height="300"> <img src="https://raw.githubusercontent.com/srobz/How-I-Analyzed-Your-Mother/main/Visuals/MarshCloud.png" height="300"> <img src="https://raw.githubusercontent.com/srobz/How-I-Analyzed-Your-Mother/main/Visuals/BarnCloud.png" height="300"> <img src="https://raw.githubusercontent.com/srobz/How-I-Analyzed-Your-Mother/main/Visuals/LilCloud.png" height="300"> <img src="https://raw.githubusercontent.com/srobz/How-I-Analyzed-Your-Mother/main/Visuals/RobCloud.png" height="300">













