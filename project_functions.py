#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.collocations import *
from wordcloud import WordCloud
from sklearn import metrics
from sklearn.metrics import classification_report, precision_recall_curve, auc, roc_curve
from sklearn.metrics import plot_confusion_matrix, cohen_kappa_score, matthews_corrcoef

nltk_stopwords = stopwords.words('english') #Pulling up NLTK stopwords

new_stopwords = ['a', 'about', 'above', 'after', 'again', 'against', 'ain', 'all', 'am', 'an', 'and', 'any', 
                 'are', 'aren', "arent", 'as', 'at', 'be', 'because', 'been', 'before', 'being', 'below', 
                 'between', 'both', 'but', 'by', 'can', 'couldn', "couldnt", 'd', 'did', 'didn', "didnt", 'do', 
                 'does', 'doesn', "doesnt", 'doing', 'don', "dont", 'down', 'during', 'each', 'few', 'for', 'from', 
                 'further', 'had', 'hadn', "hadnt", 'has', 'hasn', "hasnt", 'have', 'haven', "havent", 'having', 
                 'he', 'her', 'here', 'hers', 'herself', 'him', 'himself', 'his', 'how', 'i', 'if', 'in', 'into', 
                 'is', 'im', 'isn', "isnt", 'it', "its", 'its', 'itself', 'just', 'll', 'm', 'ma', 'me', 'mightn', 
                 "mightnt", 'more', 'most', 'mustn', "mustnt", 'my', 'myself', 'needn', "neednt", 'no', 'nor', 'ive', 
                 'not', 'now', 'o', 'of', 'off', 'on', 'once', 'only', 'or', 'other', 'our', 'ours', 'ourselves', 
                 'out', 'over', 'own', 're', 's', 'same', 'shan', "shant", 'she', "shes", 'should', "shouldve", 
                 'shouldn', "shouldnt", 'so', 'some', 'such', 't', 'than', 'that', 'thats', "thatll", 'the', 'their', 
                 'theirs', 'them', 'themselves', 'then', 'there', 'these', 'they', 'this', 'those', 'through', 
                 'to', 'too', 'under', 'until', 'up', 've', 'very', 'was', 'wasn', "wasnt", 'we', 'were', 
                 'weren', "werent", 'what', 'when', 'where', 'which', 'while', 'who', 'whom', 'why', 'will', 
                 'with', 'won', "wont", 'wouldn', "wouldnt", 'y', 'you', "youd", "youll", "youre", "youve", 'your', 
                 'yours', 'yourself', 'yourselves', 'would', 'go', 'yeah', 'yes', 'well', 'get', 'okay', 
                'got', 'oh', 'like', 'wan', 'na', 'gon', 'theyre'] #New list

def remstopandtok(text): #function to remove stopwords and tokenize line
    '''
    Function that takes the word in text and tokenizes it if not in stopword list
    '''
    return [word for word in word_tokenize(text) if not word in new_stopwords] #returns tokens without stopwords

def make_wordcloud(tokens, colormap, mask = None): #Defining function to make word clouds
    """
    Function takes in list of tokens, color, and a mask for an image shape 
    if desired and creates a word cloud of the token list; if mask is set the wordcloud
    will be in the shape of the mask.
    """
    plt.figure(figsize = (15, 10)) #Instantiate plot and set figure size
    
    wc = WordCloud(font_path = 'Images/Dax.ttf', max_words = 200, stopwords = new_stopwords, collocations = False, 
                   mask = mask, width = 1500, height = 1000, 
                   colormap = colormap).generate(" ".join(tokens)) #Instantiate wordcloud
    
    plt.imshow(wc) #Show word cloud
    
    plt.axis('off') #Turn off xy axis
    
tokens = [] #Instantiating total token list
ted_tokens = [] #Instantiation ted token list
marshall_tokens = [] #Instantiation marshall token list
barney_tokens = [] #Instantiation barney token list
robin_tokens = [] #Instantiation robin token list
lily_tokens = [] #Instantiation lily token list
    
def make_ngram(i, tokens = tokens, ted_tokens = ted_tokens, marshall_tokens = marshall_tokens, 
              barney_tokens = barney_tokens, robin_tokens = robin_tokens, lily_tokens = lily_tokens): #define function
    """
    Function takes in a number that will be the amount of tokens in the phrase and
    displays the 10 most common phrases of that number of tokens for each token list.
    """
    
    n_gram_tot = (pd.Series(nltk.ngrams(tokens, i)).value_counts())[:10] #Setting up total ngrams
    n_gram_ted = (pd.Series(nltk.ngrams(ted_tokens, i)).value_counts())[:10] #Setting up ted ngrams
    n_gram_marshall = (pd.Series(nltk.ngrams(marshall_tokens, i)).value_counts())[:10] #Setting up marshall ngrams
    n_gram_barney = (pd.Series(nltk.ngrams(barney_tokens, i)).value_counts())[:10] #Setting up barney ngrams
    n_gram_robin = (pd.Series(nltk.ngrams(robin_tokens, i)).value_counts())[:10] #Setting up robin ngrams
    n_gram_lily = (pd.Series(nltk.ngrams(lily_tokens, i)).value_counts())[:10] #Setting up lily ngrams
    
    n_gram_df_tot = pd.DataFrame(n_gram_tot) #Creating total ngram df
    n_gram_df_ted = pd.DataFrame(n_gram_ted) #Creating ted ngram df
    n_gram_df_marshall = pd.DataFrame(n_gram_marshall) #Creating marshall ngram df
    n_gram_df_barney = pd.DataFrame(n_gram_barney) #Creating barney ngram df
    n_gram_df_robin = pd.DataFrame(n_gram_robin) #Creating robin ngram df
    n_gram_df_lily = pd.DataFrame(n_gram_lily) #Creating lily ngram df
    
    n_gram_df_tot = n_gram_df_tot.reset_index() #Resetting index
    n_gram_df_ted = n_gram_df_ted.reset_index() #Resetting index
    n_gram_df_marshall = n_gram_df_marshall.reset_index() #Resetting index
    n_gram_df_barney = n_gram_df_barney.reset_index() #Resetting index
    n_gram_df_robin = n_gram_df_robin.reset_index() #Resetting index
    n_gram_df_lily = n_gram_df_lily.reset_index() #Resetting index
    
    n_gram_df_tot = n_gram_df_tot.rename(columns = {'index': 'Phrase', 0: 'Count'}) #Renaming total plot
    n_gram_df_ted = n_gram_df_ted.rename(columns = {'index': 'Phrase', 0: 'Count'}) #Renaming ted plot
    n_gram_df_marshall = n_gram_df_marshall.rename(columns = {'index': 'Phrase', 0: 'Count'}) #Renaming marshall plot
    n_gram_df_barney = n_gram_df_barney.rename(columns = {'index': 'Phrase', 0: 'Count'}) #Renaming barney plot
    n_gram_df_robin = n_gram_df_robin.rename(columns = {'index': 'Phrase', 0: 'Count'}) #Renaming robin plot
    n_gram_df_lily = n_gram_df_lily.rename(columns = {'index': 'Phrase', 0: 'Count'}) #Renaming lily plot
    
    with sns.axes_style('darkgrid'): #Setting seaborn to darkgrid style
        
        sns.set_theme(font = 'Dax')
        fig = plt.figure(figsize = (10, 15)) #Setting figsize
        ax1 = fig.add_subplot(611) #Stacking first figure
        ax2 = fig.add_subplot(612) #Stacking second figure
        ax3 = fig.add_subplot(613) #Stacking third figure
        ax4 = fig.add_subplot(614) #Stacking fourth figure
        ax5 = fig.add_subplot(615) #Stacking fifth figure
        ax6 = fig.add_subplot(616) #Stacking sixth figure
        
        sns.barplot(ax = ax1, x = 'Count', y = 'Phrase', data = n_gram_df_tot, 
                    palette = 'Dark2').set(title = 'Total Ngrams') #Assigning barplot to total ngrams
        sns.barplot(ax = ax2, x = 'Count', y = 'Phrase', data = n_gram_df_ted, 
                    palette = 'Wistia').set(title = 'Ted Ngrams') #Assigning barplot to ted ngrams
        sns.barplot(ax = ax3, x = 'Count', y = 'Phrase', data = n_gram_df_marshall, 
                    palette = 'Greens').set(title = 'Marshall Ngrams') #Assigning barplot to marshall ngrams
        sns.barplot(ax = ax4, x = 'Count', y = 'Phrase', data = n_gram_df_barney, 
                    palette = 'Blues').set(title = 'Barney Ngrams') #Assigning barplot to barney ngrams
        sns.barplot(ax = ax5, x = 'Count', y = 'Phrase', data = n_gram_df_robin, 
                    palette = 'Purples').set(title = 'Robin Ngrams') #Assigning barplot to robin ngrams
        sns.barplot(ax = ax6, x = 'Count', y = 'Phrase', data = n_gram_df_lily, 
                    palette = 'RdPu').set(title = 'Lily Ngrams') #Assigning barplot to lily ngrams
    
    plt.tight_layout() #Make plot layouts tight
    
def calc_bigram(tokens, filter_count): #Defining function to show and calculate mutual information scores
    """
    Function takes in list of tokens and minimum number of times pair of tokens needs to
    show up for a mutual information score to be calculated.
    """
    
    bigram_measures = nltk.collocations.BigramAssocMeasures() #Instantiating Bigram Association Measures
    
    tokens_pmi_finder = BigramCollocationFinder.from_words(tokens) #Instantiating Bigram Collocation Finder
    
    tokens_pmi_finder.apply_freq_filter(filter_count) #Setting minimum amount of times bigram must appear
    
    tokens_pmi_scored = tokens_pmi_finder.score_ngrams(bigram_measures.pmi) #Scoring tokens from bigrams
    
    return tokens_pmi_scored #Show scores

def passthrough(doc): #Function to passthrough our pipelines
    '''
    Function to passthrough parameters of other functions
    '''
    return doc #returns same

def confmat_and_classrep(estimator, X, y, labels, set_name): #Function that prints confusion matrix and class report
    '''
    Function to create confusion matrix and classification report: it takes in an estimator, value for X,
    value for y, class labels, and a name for the set (should be a string) and returns the confusion matrix
    and classification report, as well as Cohen's Kappa score and Matthew's Correlation Coefficient
    '''
    predictions = estimator.predict(X) #predicts from estimators
    print(f'Classification Report for {set_name} Set') #print classification report name
    print(classification_report(y, predictions, target_names = labels)) #print classification report numbers
    fig, axes = plt.subplots(nrows = 1, ncols = 2, figsize = (10, 5)) #Making subplots for each matrix
    matrix = plot_confusion_matrix(estimator, X, y, display_labels = labels, cmap = plt.cm.Spectral, 
                                  xticks_rotation = 70, values_format = 'd', ax = axes[0]) #plot non-normal matrix
    matrix.ax_.set_title(f'{set_name} Set Confusion Matrix, \n without Normalization') #setting title of non-normal
    matrix = plot_confusion_matrix(estimator, X, y, display_labels = labels, cmap = plt.cm.Spectral, 
                                  xticks_rotation = 70, normalize = 'true', ax = axes[1]) #plot normal matrix
    matrix.ax_.set_title(f'{set_name} Set Confusion Matrix, \n with Normalization') #setting title of normal matrix
    plt.subplots_adjust(wspace = 0.5) #Adding space between graphs
    plt.show() #Showing matrix
    
    print(f"Cohen's Kappa Score for {set_name} Set:") #Setting title for cohen's kappa score
    print(round(cohen_kappa_score(y, predictions), 3)) #printing cohen's kappa score
    
    print(f"Matthew's Correlation Coefficient for {set_name} Set:") #Setting title for matt's corr coef
    print(round(matthews_corrcoef(y, predictions), 3)) #printing matt's corr coef
    
    Report = classification_report(y, predictions, target_names = labels, output_dict = True)
    Report['Cohen'] = round(cohen_kappa_score(y, predictions), 3)
    Report['MCC'] = round(matthews_corrcoef(y, predictions), 3)
    
    return(Report)
    
def pr_curves(y_test_multi, y_hat_test_multi, classes): #Function that prints precision recall curves
    '''
    Function that takes in y test and y hat test and the classes and returns a graph of all
    the classes' precision recall curves as well as a legend with their auc scores
    '''
    precision = dict() #creating dict of precision scores
    recall = dict() #creating dict of recall scores
    pr_auc = dict() #creating dict of auc
    for i in range(5): #for loop
        precision[i], recall[i], _ = precision_recall_curve(y_test_multi[:, i], y_hat_test_multi[:, i]) #calling
        pr_auc[i] = auc(recall[i], precision[i]) #setting
    
    plt.figure()
    lw = 2
    colors = ['royalblue', 'deeppink', 'green', 'darkviolet', 'yellow']
    
    for i, class_ in enumerate(classes):
        plt.plot(recall[i], precision[i], color = colors[i], lw = lw, 
                 label = f'{classes[i]}, PR Curve AUC: {round(pr_auc[i], 2)}')
    
    plt.plot([0, 1], [1, 0], 'k--') #setting plot with label
    plt.xlim([0.0, 1.0]) #setting limits of x
    plt.ylim([0.0, 1.05]) #setting limits of y
    plt.xlabel('Recall') #labeling x
    plt.ylabel('Precision') #labeling y
    plt.title('Precision Recall Curve') #setting title
    plt.legend(loc="lower left") #setting legend
    plt.show() #Showing graph
        
def roc_curves(y_test_multi, y_hat_test_multi, classes): #Function that prints roc curves
    '''
    Function that takes in y test and y hat test as well as classes and returns a graph of the roc
    curves as well as a legend containing the auc scores
    '''
    fpr = dict() #creating dict of fpr
    tpr = dict() #creating dict of tpr
    roc_auc = dict() #Creating dict of auc
    for i in range(5): #for loop
        fpr[i], tpr[i], _ = roc_curve(y_test_multi[:, i], y_hat_test_multi[:, i]) #calling
        roc_auc[i] = auc(fpr[i], tpr[i]) #setting
    
    plt.figure()
    lw = 2
    colors = ['royalblue', 'deeppink', 'green', 'darkviolet', 'yellow']
    
    for i, class_ in enumerate(classes): #for loop to create singular graphs
        plt.plot(fpr[i], tpr[i], color = colors[i], lw = lw, 
                 label = f'{classes[i]}, ROC Curve AUC: {round(roc_auc[i], 2)}') #setting plot with label
        
    plt.plot([0, 1], [0, 1], 'k--') #setting plot with label
    plt.xlim([0.0, 1.0]) #setting limits of x
    plt.ylim([0.0, 1.05]) #setting limits of y
    plt.xlabel('False Positive Rate') #labeling x
    plt.ylabel('True Positive Rate') #labeling y
    plt.title('Receiver Operating Characteristic Curve') #setting title
    plt.legend(loc="lower right") #setting legend
    plt.show() #showing graph

