#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Imports 
from nltk.metrics.distance import edit_distance
import itertools

class tweetcleaner():
    
    def __init__(self, btc_df):
        self.btc_df = btc_df
        self.cleanduplicates()
        self.cleanbots()
        self.get_btc_df()
        
    def cleanbots(self):
        w, h = len(self.btc_df)+1, len(self.btc_df)+1
        Matrix = [[0 for x in range(w)] for y in range(h)] 
        x1 = range(0,len(self.btc_df)-1)
        x2 = range(0,len(self.btc_df))
        for i, j in itertools.product(x1, x2): Matrix[i][j] = edit_distance(self.btc_df.raw_text[i],self.btc_df.raw_text[j])
        
        counter = 0
        sim_txt_ind = []
        for i in range(0,len(self.btc_df)-1):
            sent_len = len(self.btc_df.raw_text[i])
            for j in range(0,len(self.btc_df)):
                if (Matrix[i][j] <= sent_len/2): # If the levenshtein distance is less than sentence length/2
                    # Matrix contains the minimum number of single-character edits 
                    # (i.e. insertions, deletions or substitutions) required to change one word into the other
                    # Ie. sentences that are atleast 50% similar 
                    counter += 1
            #print("# Similar Sentences found:",counter)
            if (counter > 1):#len(btc_df.raw_text)):
                #print("Similar text:",btc_df.raw_text[i])
                sim_txt_ind.append(i)
            counter = 0   
            
        for i in sim_txt_ind: self.btc_df = self.btc_df.drop(i)
        self.btc_df = self.btc_df.reset_index(drop=True)
        print(len(sim_txt_ind)," similar Tweets (Levenshtein edit-distance) found.")
        print(len(self.btc_df.raw_text)," unique Tweets found.")
        
    def cleanduplicates(self):
        print(len(self.btc_df['raw_text'][self.btc_df.duplicated('raw_text') == True])," duplicate Tweets found.")
        self.btc_df = self.btc_df.drop_duplicates('raw_text')
        self.btc_df = self.btc_df.reset_index(drop=True)
        
    def get_btc_df(self):
        return self.btc_df

