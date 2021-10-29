# How-I-Analyzed-Your-Mother
## Capstone Project

<img src="https://raw.githubusercontent.com/srobz/How-I-Analyzed-Your-Mother/main/Visuals/himymcast.png" height="500">

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

* Total Lines during Series - Largest Vocabulary

<img src="https://raw.githubusercontent.com/srobz/How-I-Analyzed-Your-Mother/main/Visuals/TotalLines.png" height="200"> <img src="https://raw.githubusercontent.com/srobz/How-I-Analyzed-Your-Mother/main/Visuals/Vocabulary.png" height="200">

* Character Mentions - Total and by Nickname

<img src="https://raw.githubusercontent.com/srobz/How-I-Analyzed-Your-Mother/main/Visuals/CharMentions.png" height="200"> <img src="https://raw.githubusercontent.com/srobz/How-I-Analyzed-Your-Mother/main/Visuals/Nicknames.png" height="200">

* Common Three Word Phrases: Three word phrases are where differences started to become more apparent among the characters.

<img src="https://raw.githubusercontent.com/srobz/How-I-Analyzed-Your-Mother/main/Visuals/ThreePhrase.png" height="600">

#### Word Clouds

* Total Corpus - Ted - Marshall - Barney - Lily - Robin

<img src="https://raw.githubusercontent.com/srobz/How-I-Analyzed-Your-Mother/main/Visuals/CorpusWC.png" height="200"> <img src="https://raw.githubusercontent.com/srobz/How-I-Analyzed-Your-Mother/main/Visuals/TedCloud.png" height="200"> <img src="https://raw.githubusercontent.com/srobz/How-I-Analyzed-Your-Mother/main/Visuals/MarshCloud.png" height="200"> <img src="https://raw.githubusercontent.com/srobz/How-I-Analyzed-Your-Mother/main/Visuals/BarnCloud.png" height="200"> <img src="https://raw.githubusercontent.com/srobz/How-I-Analyzed-Your-Mother/main/Visuals/LilCloud.png" height="200"> <img src="https://raw.githubusercontent.com/srobz/How-I-Analyzed-Your-Mother/main/Visuals/RobCloud.png" height="200">

#### Sentiment Analysis

I looked at how Marshall and Lily's polarity and subjectivity evolved over seasons 1 and 2 as their relationship evolved, I also did the same for Ted and Robin with season 1, 2, and 3, as well as Robin and Barney's relationship across seasons 3, 4, and 5. I've only included a few of the sentiment analysis graphs here just to show how their sentiment ebbs and flows throughout the season and their storylines with each other. If you want to see further you can look through ['Capstone Project - 3 - Exploratory Data Analysis.ipynb'](https://github.com/srobz/How-I-Analyzed-Your-Mother/blob/main/Capstone%20Project%20-%203%20-%20Exploratory%20Data%20Analysis.ipynb).

* Marshall and Lily Sentiment Spread - Season 2 // Barney and Robin Sentiment Spread - Season 5

<img src="https://raw.githubusercontent.com/srobz/How-I-Analyzed-Your-Mother/main/Visuals/MLSent2.png" height="200"> <img src="https://raw.githubusercontent.com/srobz/How-I-Analyzed-Your-Mother/main/Visuals/BRSent5.png" height="200">


### Model

Using a brute force approach I tried six different models with two vectorizers each, resulting in twelve baseline models. I evaluated these models using Cohen's Kappa score, as it is a metric used on multi-class classifiers in imbalanced datasets. I then tuned the top five models using a brute force approach with parameter tuning, resulting in 23 tuned models.


### iNterpret

My final model was Logistic Regression using Count Vectorizer with C set to 0.5. This model yielded an accuracy of 0.32 with Cohen's Kappa score at 0.125. This was not a great model at all and even after hyperparameter tuning of this model, this was still the highest it got. In general, the model was able to accurately classify Barney's lines as Barney's 34% of the time, Lily's as Lily's 34% of the time, Marshall's as Marshall's 31% of the time, Robin's as Robin's 27% of the time, and Ted as Ted's 33% of the time. Due to how imbalanced the data was and how similar the language was between each character, this does not surprise me.

<img src="https://raw.githubusercontent.com/srobz/How-I-Analyzed-Your-Mother/main/Visuals/ConfMat.png" height="300">

Regarding our precision recall curve: none of these would be considered good classifiers, as they all perform worse than random guessing (0.5). The highest AUC score is Ted's, at 0.36, followed by Barney with 0.34, then Marshall at 0.28, Lily at 0.26, and Robin with 0.24.

<img src="https://raw.githubusercontent.com/srobz/How-I-Analyzed-Your-Mother/main/Visuals/PRC.png" height="300">


## Conclusion

The EDA dug into the depths of the scripts as well as the characters. We learned a lot, but there is still so much more that could have been done, such as exploring who used which common phrases, common words/phrases in each season, deeper analysis into themes across seasons and characters, etc. We didn't even bring in any other characters! Can you imagine what that would look like?

In addition, the modeling process can be improved by perhaps trying to classify the characters' lines in a specific season, rather than across the entire series. It could also be improved by classifying by season rather than character, as there are specific terms used in specific seasons, and those differences are much more apparent. This data was heavily imbalanced as well, so going back and seeing if there are ways to balance the data prior to modeling might help yield a higher accuracy.

While our models were not the best, we did create a robust exploratory data analysis that I feel encompasses the heart of the show. You can see the rise and fall of character relationships as well as how some of our favorite lines and catchphrases are used; all of this adds to the ambience of the How I Met Your Mother experience, and as an avid fan I'm glad I got to re-experience the show through this avenue.



## Repository Organization
- ['Capstone Project - 1 - Web Scraping and Initial DF.ipynb'](https://github.com/srobz/How-I-Analyzed-Your-Mother/blob/main/Capstone%20Project%20-%201%20-%20Web%20Scraping%20and%20Initial%20DF.ipynb) : Jupyter notebook of Web Scraping with code and comments
- ['Capstone Project - 2 - Preprocessing and Database Creation.ipynb'](https://github.com/srobz/How-I-Analyzed-Your-Mother/blob/main/Capstone%20Project%20-%202%20-%20Preprocessing%20and%20Database%20Creation.ipynb) : Jupyter notebook of preprocessing with code and comments
- ['Capstone Project - 3 - Exploratory Data Analysis.ipynb'](https://github.com/srobz/How-I-Analyzed-Your-Mother/blob/main/Capstone%20Project%20-%203%20-%20Exploratory%20Data%20Analysis.ipynb) : Jupyter notebook of Exploratory Data Analysis with code and comments
- ['Capstone Project - 4 - Modeling.ipynb'](https://github.com/srobz/How-I-Analyzed-Your-Mother/blob/main/Capstone%20Project%20-%204%20-%20Modeling.ipynb) : Jupyter notebook of Modeling with code and comments
- ['Capstone Presentation.pdf'](https://github.com/srobz/How-I-Analyzed-Your-Mother/blob/main/Capstone%20Presentation.pdf) : PDF of Capstone project Powerpoint presentation
- ['Characters'](https://github.com/srobz/How-I-Analyzed-Your-Mother/tree/main/Characters) : Folder containing images of characters
- ['Images'](https://github.com/srobz/How-I-Analyzed-Your-Mother/tree/main/Images) : Folder containing font and miscellaneous images used in notebooks
- ['Visuals'](https://github.com/srobz/How-I-Analyzed-Your-Mother/tree/main/Visuals) : Folder containing visuals for ReadMe
- ['DFforSQL.csv'](https://github.com/srobz/How-I-Analyzed-Your-Mother/blob/main/DFforSQL.csv) : CSV file created in notebook 2 used to create SQL database
- ['HIMYM.db'](https://github.com/srobz/How-I-Analyzed-Your-Mother/blob/main/HIMYM.db) : SQL database for project created on remote desktop
- ['ModelDF.csv'](https://github.com/srobz/How-I-Analyzed-Your-Mother/blob/main/ModelDF.csv) : CSV file created in notebook 3 used in notebook 4 for modeling
- ['project_functions.py'](https://github.com/srobz/How-I-Analyzed-Your-Mother/blob/main/project_functions.py) : File containing all functions used in notebooks
