#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#Imports
import pandas as pd
import cryptonator
import tweepy
import cbpro
from datetime import datetime, timezone

class tweetscraper():
    
    def __init__(self, num_tweets):
        self.num_tweets = num_tweets
        self.since_id = 0 # 0 starting tweet id
        self.twitter_api, self.auth_client = self.apiauth()
        self.btc_df = self.batchdownload()
        self.get_btc_df()
        
    def apiauth(self):
        #Auth / APIs
        auth = tweepy.OAuthHandler('xxXXXXXXXXXXXxx', 'xxXXXXXXXXXXXxx')
        auth.set_access_token('xxXXXXXXXXXXXxx', 'xxXXXXXXXXXXXxx')
        twitter_api = tweepy.API(auth) # Twitter
        crypto_api = cryptonator.Cryptonator() # crypto price API
        b64secret = 'xxXXXXXXXXXXXxx'
        key = 'xxXXXXXXXXXXXxx'
        passphrase = 'xxXXXXXXXXXXXxx'
        auth_client = cbpro.AuthenticatedClient(key, b64secret, passphrase) # CoinbasePro
        return twitter_api, auth_client
    
    def batchdownload(self):
        
        def utc_to_local(utc_dt): # timezone adjustment (if needed)
            return utc_dt.replace(tzinfo=timezone.utc).astimezone(tz=None)
        
        def get_tweet_df(query, count, sinceid):
            # empty list to store parsed tweets
            btc_df = pd.DataFrame(columns = ['tweet_ID','raw_text','time_stamp','user_ID','btc_price']) 
            # in the future add more (ex. user location might be useful)
            # call twitter api to fetch tweets
            q=str(query)
            a=str(q+" bitcoin")
            #b=str(q+" sarcastic") # Add more keywords here based on research, perhaps bitcoin when mentioned with other altcoins
            fetched_tweets = self.twitter_api.search(a, count = count, since_id = sinceid)
            # parsing tweets batch by batch, one by one
            sinceid = fetched_tweets.max_id
            btc_df['tweet_ID'] = [t.id for t in fetched_tweets]
            btc_df['raw_text'] = [t.text for t in fetched_tweets]
            btc_df['user_ID'] = [t.user.id for t in fetched_tweets]
            btc_df['time_stamp'] = [t.created_at for t in fetched_tweets]
            btc_df['time_stamp'].apply(utc_to_local)
            btc_df['btc_price'] = float(self.auth_client.get_product_ticker('BTC-USD')['price'])
            return btc_df, sinceid
        
        btc_df = pd.DataFrame()
        
        for i in range(0,int(self.num_tweets/100)): # batch run of 100 tweets - 100 is max limit on twitter api
            btcdf, sinceid = (get_tweet_df(query ="", count =100, sinceid = self.since_id)) # keep query empty here for now
            btc_df = btc_df.append(btcdf)
            since_id = sinceid
            print(len(btc_df)," tweets scraped!")
        
        return btc_df
    
    def get_btc_df(self):
        return self.btc_df
    

