#!/usr/bin/env python
# coding: utf-8

# ## Rahul Zalkikar
# ### Custom Natural Language Processing on Real-Time Bitcoin Tweets for Sentiment Analysis and Algorthmic Trading

# In[88]:


# Imports / Helper Classes
import os
import time
t0 = time.time()
os.chdir('C:\\Users\\rayzc\\Downloads')
from Zalk_TweetDownloader import tweetscraper
from Zalk_TweetCleaner import tweetcleaner
from Zalk_Preprocessing1 import tweetpreprocess


# ### Zalk_TweetDownloader
# 
# #### Tweet Scraping
# 
# I can scrape a number of Bitcoin related tweets using the tweepy API and live Bitcoin price data using the coinbase API. To add keywords besides Bitcoin and other metrics like BTC volume modify the helper class.

# In[2]:


num_tweets = 200 #2000 # number of tweets to be scraped on each run - must be >100

tweet_scraper = tweetscraper(num_tweets) # scrape tweets 

btc_df = tweet_scraper.get_btc_df() # RETURNS A DATAFRAME 


# In[95]:


btc_price = float(btc_df.btc_price.iloc[0])
btc_price


# ### Zalk_TweetCleaner
# 
# #### Noise Tweets
# 
# Inspired largely by (Colianni et al.), I found that most BOT tweets could be filtered by removing duplicate tweets. 
# 
# From anecdotal experience, I have found these tweets to encompass anywhere between 90-98% of tweets scraped during a given run.
# 
# In order to ensure that automated posts were removed, I compute the Levenshtein edit-distance from each tweet to every other tweet and form a distance matrix, removing tweets that are atleast 50% similar.
# 
# *The Levenshtein distance is the edit distance between two strings. This value can be leveraged to prove that any pair of tweets is dissimilar by at least some threshold value.*

# In[3]:


tweet_cleaner = tweetcleaner(btc_df) # clean bot tweets

btc_df_cleaned = tweet_cleaner.get_btc_df() # RETURNS A DATAFRAME 


# ### Zalk_Preprocessing1 
# *Note these processes are not listed in their order of execution in class code*
# 
# #### Standard:
# 
# * Unfortunately, there is no right way to clean tweets perfectly via simple regular expression. 
# 
# * An optimal method, as mentioned by (Bakliwal et al., 2012), would be using a precognitive self-learning algorithm since twitter has no standard tweet format.
# 
# * Here, I:
#     1. convert words to lowercase
#     2. remove words less than two character lengths
#     3. remove digits (that are not contained in words)
#     4. remove special characters (non alphanumerics) , hashtags, urls, targets(@s). 
# 
# However, there is a lot more to consider.
# 
# #### Spell Correction:
# 
# I was inspired to write my own spell correction algorithm similar to the one used in (Bora, 2012).
# 
# Tweets have no standard form, but aren't necessarily random either. For a project dealing with user-generated content, preprocessing without any focus given to spell correction can harm results. 
# 
# Users type certain characters an arbitrary number of times to put more emphasis on the word. This reflects a different sentiment than the word itself. 
# 
# In my algorithm I replace a word with a character repeating three times or more with two words, one in which the repeated character is placed once and second in which the repeated character is placed twice. 
# 
# Examples: 
# 
# 1. ‘swwweeeetttt’ is replaced with 8 words: ‘swet’, ‘swwet’, ‘sweet’, ‘swett’, ‘swweet’, 'sweett', 'swwett', 'swweett'
# 
# 2. 'cooooolll' is replaced with 4 words: 'col', 'coll', 'cool', 'cooll'
# 
# 3. 'dddduuuuuude' is replaced with 4 words: 'dude', 'ddude', 'duude', 'dduude'
# 
# *First Note: You might be thinking: Don't most common spelling mistakes occur because of missing characters? Yes, absolutely. These spelling mistakes are not currently handled by my code. I propose a phonetic level spell correction method in future.*
# 
# *Second Note: I do not use an acronym dictionary referenced by (Agarwal et al., 2011) in their research. That dictionary has translations for 5,184 acronyms (e.g. lol is translated to laughing out loud). However, acronym checking is also not employed in the research paper that I am (mainly) basing my preprocessing on.*
# 
# ##### Punctuation:
# 
# * Exclamation marks (!) and question marks (?) tend to carry some sentiment. 
# 
#     1. ‘!’ is often used when we have to emphasize on a positive word.
#     
#     2. ‘?’ is often used to highlight the state of confusion or disagreement. 
#     
# I replace all the occurrences of ‘!' with ‘zzexclaimzz’ and of ‘?’ with ‘zzquestzz’. 
# 
# ##### Other Cases:
# 
# To incorporate common examples of character/letter replacement, I replace the number '3', when contained in words, with 'e', and '5' in words with 's'.

# In[5]:


def clean(str):
    processed_tweet = tweetpreprocess(str) # clean tweet string
    clean_text = processed_tweet.get_clean_tweet() # RETURNS A CLEANED STRING
    return clean_text


# In[10]:


example_tweet = clean("I realllyyy love Bitcoin!")

example_tweet # Spell Correction in action!


# In[47]:


btc_df_cleaned['cleaned_text'] = btc_df_cleaned.raw_text.apply(clean) # Applying tweet cleaning across raw tweet text


# #### Emojis:
# 
# At first, I will be using my best judgement as an avid twitter and emoji user, factoring in wikipedia emoji definitions and published research, to assign sentiment in a binary way (+1, -1) to emojis scraped in a given run.
# 
# After enough tweet data is collected and matched with corresponding Bitcoin price changes, an autonomous sentiment polarity dictionary is (re)created for each emoji.
# 
# This dictionary assigns each emoji that was scraped a +/- score between 0-1 depending on the emoji count during Bitcoin price rises and declines.
# 
# I replace all the emoticons which I consider positive or extremely positive with ‘zzhappyzz’ and the emoticons I consider negative or extremely negative with ‘zzsadzz’. These cutoffs are >0.6 for positive emojis and <0.3 for negative emojis.
# 
# * I append and prepend ‘zz’ to markers in order to prevent them from mixing into tweet text.
# 
# * Cutoffs of 0.6 and 0.3 are formed by trial and error.

# In[18]:


import emoji
import numpy as np
import pandas as pd
from nltk.tokenize import sent_tokenize, word_tokenize


# In[17]:


emoji_pol_dict_path = 'C:/Users/rayzc/OneDrive/Pictures/Documents/emojidict11.csv'


# In[16]:


emoji_chain = [] # emoji list for tweets
def form_emoji_dict(s):
    emo = emoji.demojize(' '.join(c for c in s if c in emoji.UNICODE_EMOJI))
    emoji_chain.append(emo)

btc_df['raw_text'].apply(form_emoji_dict) # We don't want our RAW text altered and instead deal with emojis as separate entities

e = np.array(emoji_chain).astype(np.dtype).sum()
e = e.replace(':', ' ')
e = np.unique(word_tokenize(e))
print(e) # All Unique Emojis in tweet set!


# In[19]:


# If our emoji polarity dictionary has been formed, use it instead of the hardcoded one.
if os.path.isfile(emoji_pol_dict_path):
    emojidict = pd.read_csv(emoji_pol_dict_path)
    emoji_dict = dict(zip(emojidict.Emoji, emojidict.Polarity))
    
# hardcoded emoji polarity dict
else:
    emoji_dict = {'rocket': 1, 'white_heavy_check_mark': 1, 'heavy_check_mark': 1, 'thumbs_up': 1,
                'grinning_face': 1, 'grinning_face_with_smiling_eyes': 1,
                'beaming_face_with_smiling_eyes': 1, 'TOP_arrow': 1, 'dollar_banknote': 1,
                'money_bag': 1, 'fire': 1, 'upwards_button': 1, 'kissing_face_with_smiling_eyes': 1,
                'kissing_face_with_closed_eyes': 1, 'face_throwing_a_kiss': 1, 'smiling_face_with_smiling_eyes': 1,
                'money_with_wings': 1, 'trophy': 1, 'face_with_tears_of_joy': 1, 'red_heart': 1, 
                'party_popper': 1, 'chart_increasing': 1, 'loudly_crying_face': -1,
                'no_entry_sign': -1, 'confused_face': -1, 'unamused_face': -1, 'weary_face': -1, 
                'frowning_face_with_open_mounth': -1, 'angry_face': -1, 'expressionless_face': -1, 
                'pensive_face': -1 ,'crying_face': -1, 'disappointed_face': -1, 'downwards_button': -1, 'thumbs_down': -1,
                'DOWN_arrow': -1}


# In[20]:


def extract_clean_emojis(s):
    
    # Possible overlaying: the presence of ':c' in emojis replaced with ':clapping_hands:' would lead unintended results
    # Check/filter emojis that have a specific unicode after replacing emojis that contain simple characters like :-)
    
    # This can also be avoided by ensuring there are no spaces before / after ':c'. However, in reality, people often tweet
    # these emojis conjoined with words ie. "makes me sad:c". I don't want to overlook this.
    
    pos = [':)', ':-)', ':o)', ':]', ':3', ':c)',':D', 'C:', ':-D', ';)', ';p', ":')'"]
    neg = [':(', '>:(', ':-(', ':c', ':[', '>:(', '>:[', '>:[', ":'("]
    counter = 0
    
    for e in pos: 
        if e in s: 
            s = s.replace(e,' zzhappyzz ')
            counter+=1
    for e in neg:
        if e in s:
            s = s.replace(e,' zzsadzz ')
            counter+=1
            
    emo = emoji.demojize(' '.join(c for c in s if c in emoji.UNICODE_EMOJI))
    s = s.replace(' '.join(c for c in s if c in emoji.UNICODE_EMOJI), emo)
    
    for e,v in emoji_dict.items(): 
        if e in s:
            if v > 0.6:
                s = s.replace(e,' zzhappyzz ')
                counter+=1
            elif v < 0.3:
                s = s.replace(e,' zzsadzz ')
                counter+=1
            else: s = s.replace(e,' ') #Don't care about neutral emojis
    #print(str(counter)+' emojis replaced by markers.')
    
    return s


# ### Modified Stemming
# 
# Here I stem all of the cleaned text and remove stopwords.
# 
# As done in (Bakliwal et al., 2012), I modified the traditional porter stemmer by restricting it to step 1. Step 1 gets rid of plurals and -ed or -ing suffixes.
# 

# In[45]:


import nltk
import re
from nltk.corpus import stopwords


# In[48]:


# Clean all Tweet text

myStemmer = nltk.stem.porter.PorterStemmer()

clean_stemmed_text = [''.join(sentence) for sentence in btc_df_cleaned['cleaned_text']]
clean_stemmed_text = [[myStemmer._step1b(myStemmer._step1a(word)) for word in sentence.split(" ")] for sentence in clean_stemmed_text]
clean_stemmed_text = [' '.join(sentence) for sentence in clean_stemmed_text]
clean_stemmed_text = [[word for word in sentence.split(" ") if word not in stopwords.words('english')] for sentence in clean_stemmed_text] # I remove stopwords here


# In[49]:


clean_stemmed_text = [' '.join(sentence) for sentence in clean_stemmed_text]
clean_stemmed_text = [re.sub(' +', ' ',sentence) for sentence in clean_stemmed_text] # Don't need extra spaces


# In[50]:


btc_df_cleaned['processed_text'] = clean_stemmed_text # Add to df


# In[51]:


btc_df_cleaned = btc_df_cleaned[btc_df_cleaned['processed_text'] != ' '] # We only deal with non-empty cleaned tweets


# In[53]:


(btc_df_cleaned.shape)


# ### Unigram Formation
# 
# I form positive and negative unigrams from high confidence positive and negative tweets.
# 
# Since there is no provided confidence measure independent of my own approach, I leverage the VADER Sentiment package and its corresponding SentimentIntensityAnalyzer(). 
# 
# VADER (Valence Aware Dictionary and sEntiment Reasoner) is a lexicon and rule-based sentiment analysis tool that is specifically attuned to sentiments expressed in social media. In other words, VADER maps lexical features to emotion intensity, and combines this with five simple heuristics, which encode how contextual elements increment, decrement, or negate the sentiment of text.
# 
# Its performance and speed when dealing with streaming/real-time data and its independence from training data (it's constructed from a generalizable, valence-based, human-curated gold standard sentiment lexicon) make it appealing for this project.
# 
# *Note: It is also worth attempting bigram, trigram, etc. approaches instead of (or in addition to) the unigram approach in the future.*

# In[65]:


from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import seaborn as sns


# In[57]:


analyser = SentimentIntensityAnalyzer()

def sentiment_analyzer_scores(sentence):
    score = analyser.polarity_scores(sentence)
    return score


# In[58]:


def prob_pos(prob_vect): return prob_vect[0]
def prob_neg(prob_vect): return prob_vect[1]
def prob_neu(prob_vect): return prob_vect[2]
def prob_compound_dict(dict): return dict['compound']


# In[60]:


btc_df_cleaned['VADER_compound_raw'] = btc_df_cleaned['raw_text'].apply(sentiment_analyzer_scores).apply(prob_compound_dict)
btc_df_cleaned['VADER_compound_cleaned'] = btc_df_cleaned['cleaned_text'].apply(sentiment_analyzer_scores).apply(prob_compound_dict)
btc_df_cleaned['VADER_compound_processed'] = btc_df_cleaned['processed_text'].apply(sentiment_analyzer_scores).apply(prob_compound_dict)


# In[67]:


corr = btc_df_cleaned[['VADER_compound_raw','VADER_compound_cleaned','VADER_compound_processed']].corr(method="pearson")
#corr 

# VADER Sentiment scores between raw, cleaned, and processed text should be HIGHLY correlated, which is what we see here.


# In[70]:


non_neu_txts = btc_df_cleaned[((btc_df_cleaned['VADER_compound_raw'] > 0.05) |
                      (btc_df_cleaned['VADER_compound_raw'] < -0.05)) &
                      ((btc_df_cleaned['VADER_compound_cleaned'] > 0.05) |  
                      (btc_df_cleaned['VADER_compound_cleaned'] < -0.05)) & 
                      ((btc_df_cleaned['VADER_compound_processed'] > 0.05) |
                      (btc_df_cleaned['VADER_compound_processed'] < -0.05))] # We don't care about neutral tweets
neg_inds = non_neu_txts['processed_text'][btc_df_cleaned['VADER_compound_processed']>0.05].index
pos_inds = non_neu_txts['processed_text'][btc_df_cleaned['VADER_compound_processed']<-0.05].index


# In[71]:


pos_docs = non_neu_txts['processed_text'][pos_inds.values]
neg_docs = non_neu_txts['processed_text'][neg_inds.values]

pos_stemmed_words_per_doc = [[myStemmer._step1b(myStemmer._step1a(word)) for word in sentence.split(" ")] for sentence in pos_docs]
pos_stemmed_unis = np.array(pos_stemmed_words_per_doc).sum()

neg_stemmed_words_per_doc = [[myStemmer._step1b(myStemmer._step1a(word)) for word in sentence.split(" ")] for sentence in neg_docs]
neg_stemmed_unis = np.array(neg_stemmed_words_per_doc).sum()

filtered_pos_unis = [word for word in pos_stemmed_unis if word not in stopwords.words('english')] # I remove stopwords here
filtered_pos_unis = list(filter(lambda a: a != '', filtered_pos_unis)) # '' is not a uni. remove it with functional approach since .remove() only acts on first appearance of ''

filtered_neg_unis = [word for word in neg_stemmed_unis if word not in stopwords.words('english')] # I remove stopwords here
filtered_neg_unis = list(filter(lambda b: b != '', filtered_neg_unis))


# In[72]:


unq_filtered_pos_unis = list(set(filtered_pos_unis))
unq_filtered_neg_unis = list(set(filtered_neg_unis))

# All Unique Unigrams in this tweet set
all_unq_unis = list(set(unq_filtered_pos_unis + unq_filtered_neg_unis))


# ### Unigram/Tweet Scoring
# 
# I form unigram probability scores prior to scoring tweets.
# 
# First, I look at the reduced word and identify it as a noun by looking at its part of speech tag in English using WordNet(Miller, 1995). If the majority sense (most commonly used sense) of that word is Noun, I discard the word while scoring. 
# 
# Nouns inherently don’t carry sentiment and are of no use in scoring.
# 
# Second, I reduce the effect of nonsentiment bearing words in scoring. I do this by boosting the scores of sentiment bearing words by looking for each token in a pre-defined list of positive and negative words. 
# 
# When I come across a token in this list, instead of scoring it using the Naive Bayes formula detailed below, I score the token +/- 1 depending on the list in which it exists.
# 
# Naive Bayes Formula
# 
# * Pf = Frequency in Positive Training Set
# 
# * Nf = Frequency in Negative Training Set
# 
# * Pp = Positive Probability of the token
# 
# * Np = Negative Probability of the token 
# 
# $$ Pp = Pf / (Pf + Nf)$$
# 
# $$ Np = Nf / (Pf + Nf )$$
# 
# Third, I account for emojis / punctuation as follows:
# 
# * +1 to the total tweet score for each ‘zzhappyzz’ (positive / extremely positive emoji)
# * -1 from the total tweet score for each ‘zzsadzz’ (negative / extremely negative emoji)
# * +0.1 to the total tweet score for each ‘zzexclaimzz’ (exclamation mark)
# * -0.1 from the total tweet score for each ‘zzquestzz’ (question mark)
# 
# 1 and 0.1 degrees are chosen by trial and error methods (Bakliwal et al., 2012).
# 
# Fourth, to boost the scores of the most commonly used words, which are domain specific, I multiply a word's popularity factor (pF) to the score of each unigram token determined by the probability difference Pp - Np. I assign popularity factors to words based on the absolute difference between their Pf and Nf. Those thresholds are chosen by trail and error methods.
# 
# $$ UNIscore= (Pp - Np) * pF $$
# 
# I sum the scores of all the constituent unigrams to find an overall score measure of the tweet. 
# 
# If tweet score is > 0 then it is positive (otherwise negative). 
# 
# *Note: I attempted this weighting strategy, as used in (Bakliwal et al., 2012), to deviate from the traditional TF-IDF method. In practice, different datasets require tailored weighting schemes for best performance.*

# In[84]:


from nltk.corpus import wordnet as wn
from itertools import repeat


# In[74]:


def noun_identification(s):
    part_of_speech = []
    if s == 'bitcoin': return False # Bitcoin goes unregistered as a noun
    for s1 in wn.synsets(s):
        s1 = str(s1)
        part_of_speech.append(s1[s1.index('.')+1:s1.index(')')-4])
    if len(part_of_speech) > 0:
        if part_of_speech.count('n')/len(part_of_speech) > 0.5:
            return False
        else: return True
    else: return True 


# In[78]:


def positiveWords():
    with open(r'C:\Users\rayzc\OneDrive\Pictures\Documents\positive-words.txt') as word_file:
        return set(word.strip().lower() for word in word_file)  
positive_word_list = positiveWords()


# In[79]:


def negativeWords():
    with open(r'C:\Users\rayzc\OneDrive\Pictures\Documents\negative-words.txt') as word_file:
        return set(word.strip().lower() for word in word_file)  
negative_word_list = negativeWords()


# In[80]:


def check_pos_neg(s):
    if s in negative_word_list: return -1
    elif s in positive_word_list: return 1
    else: return 0


# In[81]:


# Form Unigram metric table for later scoring
uni_df = pd.DataFrame(columns=['uni','pf','nf','pp','np','pp+np','pop_score','score'])
uni_df['uni'] = all_unq_unis
for uni in all_unq_unis:
    if noun_identification(uni) is True:
        if (uni not in ['zzexclaimzz', 'zzquestzz', 'zzhappyzz', 'zzsadzz']): # Emojis/Punctuations will be taken care of later
            psore = 0
            uni_df['pf'][uni_df.uni == uni] = filtered_pos_unis.count(uni) 
            uni_df['nf'][uni_df.uni == uni] = filtered_neg_unis.count(uni)
            uni_df['pp'][uni_df.uni == uni] = uni_df['pf'][uni_df.uni == uni] / (uni_df['pf'][uni_df.uni == uni] + uni_df['nf'][uni_df.uni == uni])
            uni_df['np'][uni_df.uni == uni] = uni_df['nf'][uni_df.uni == uni] / (uni_df['pf'][uni_df.uni == uni] + uni_df['nf'][uni_df.uni == uni])
            uni_df['pp+np'][uni_df.uni == uni] = uni_df['pp'][uni_df.uni == uni] - uni_df['np'][uni_df.uni == uni]
            pscore = abs(filtered_pos_unis.count(uni) - filtered_neg_unis.count(uni))
            if pscore > .75*(len(non_neu_txts['processed_text'])):
                uni_df['pop_score'][uni_df.uni == uni] = 0.9
            elif pscore > .5*(len(non_neu_txts['processed_text'])):
                uni_df['pop_score'][uni_df.uni == uni] = 0.7
            elif pscore > .25*(len(non_neu_txts['processed_text'])):
                uni_df['pop_score'][uni_df.uni == uni] = 0.5
            elif pscore <= .25*(len(non_neu_txts['processed_text'])):
                uni_df['pop_score'][uni_df.uni == uni] = 0.1
            uni_df['score'][uni_df.uni == uni] = uni_df['pp+np'][uni_df.uni == uni]*uni_df['pop_score'][uni_df.uni == uni]
            if check_pos_neg(uni) == -1: uni_df['score'][uni_df.uni == uni] == -1
            elif check_pos_neg(uni) == 1: uni_df['score'][uni_df.uni == uni] == 1
        else:
            uni_df['score'][uni_df.uni == uni] = 0  
    else: 
        uni_df['score'][uni_df.uni == uni] = 0


# In[82]:


uni_score_dict = dict(zip(uni_df['uni'], uni_df['score'])) # Unigram Score Dictionary


# In[85]:


tweet_scores = []
scored_unis = []
txt_df = pd.DataFrame()
for tweet in non_neu_txts['processed_text']:
    tweet_score = 0
    relevant_unis = []
    
    # Emojis / Punctuation Scoring
    if 'zzexclaimzz' in tweet: 
        tweet_score += (0.1*tweet.count('zzexclaimzz')) # for each occurance
        relevant_unis.extend(repeat('zzexclaimzz',tweet.count('zzexclaimzz')))
    if 'zzquestzz' in tweet: 
        tweet_score -= (0.1*tweet.count('zzquestzz'))
        relevant_unis.extend(repeat('zzquestzz',tweet.count('zzquestzz')))
    if 'zzhappyzz' in tweet: 
        tweet_score += (1*tweet.count('zzhappyzz'))
        relevant_unis.extend(repeat('zzhappyzz',tweet.count('zzhappyzz')))
    if 'zzsadzz' in tweet: 
        tweet_score -= (1*tweet.count('zzsadzz'))
        relevant_unis.extend(repeat('zzsadzz',tweet.count('zzsadzz')))
        
    # Unigram Scoring
    for uni, score in uni_score_dict.items():
        if uni in tweet:
            tweet_score += (score*tweet.count(uni))
            relevant_unis.extend(repeat(uni,tweet.count(uni)))
    
    tweet_scores.append(tweet_score)

    if len(relevant_unis) == 0: 
        relevant_unis.append('')
        
    scored_unis.append(' '.join(relevant_unis))
    
txt_df['tweet_score'] = tweet_scores
txt_df['relevant_unis'] = scored_unis


# In[86]:


if txt_df['tweet_score'].mean() < 0: verdict = "Mean Sentiment is Negative"
elif txt_df['tweet_score'].mean() > 0: verdict = "Mean Sentiment is Positive"
else: verdict = "HODL"


# In[87]:


verdict


# ### Store Data

# In[102]:


import csv
from datetime import datetime
t1 = time.time()
form = '%Y-%m-%d %H:%M:%S.%f'
date = datetime.strftime(datetime.now(), form)


# In[100]:


# Store Emoji Data
emojistr = ', '.join(e) 
with open(emoji_pol_dict_path,'a', newline='', encoding='utf-8') as fd:
    writer=csv.writer(fd, delimiter=',')
    writer.writerow([emojistr])
    print("Emojis Stored.")


# In[103]:


# For storing data and signals
DataList = [bytes(date.encode()),
            str(txt_df['tweet_score'].count()),
            str(txt_df['tweet_score'].mean()),
            str(txt_df['tweet_score'].sum()),
            verdict,
            btc_price]


# In[104]:


with open(r'C:\Users\rayzc\OneDrive\Pictures\Documents\cryptomodel11.csv','a', newline='', encoding='utf-8') as fd:
    writer=csv.writer(fd, delimiter=',')
    writer.writerow(DataList)
    print("Data Stored.")


# In[105]:


with open(r'C:\Users\rayzc\OneDrive\Pictures\Documents\cryptotext11.csv','a', newline='', encoding='utf-8') as ft:
    txt_df.to_csv(ft, index=False)
    print("Text Stored.")

