# BTC_TwitterScrape_helpers

Research has shown twitter is the most reliable social media platform for live sentiment analysis because tweets are small and limited in ambiguity, easily stream-able with an API, and are unbiased (ex. A ‘like’ on Facebook is a biased reaction since there is no way to tell how individual users are feeling, whereas the words expressed by an individual on twitter are their own words).

Process (specific script runs and data not featured in this repo)
* I scrape about 2 thousand Bitcoin-related tweets in about 1 minute and clean the data. As expected, I found that 90-95% of tweets collected are bots. After filtering those out I also removed retweets that have little to no sentimental value based on an edit-distance value between sentences (number of single character deletions, insertions, or substitutions required to transform one string of text to another).

* After this, I introduce some nuance by using custom preprocessing techniques (from using unigram features instead of bigrams or trigrams, to writing a spell corrector algorithm that accounts for different sentiments when characters are repeated over 3 times in a word).

* I also build an autonomously updating emoji polarity dictionary for bitcoin. In other words, as bitcoin’s price fluctuates, tweets are aggregated and labeled according to the price change during that interval. These tweets contain emojis that are then scored +1/-1 based on their counts during price rises/falls – leading to an ever-changing polarity score for each emoji that quantifies its relationship to bitcoin price changes. This is my way of going beyond standard lexicon dictionaries and specializing my analysis for bitcoin / crypto, since a substantial portion of ‘crypto language’ (ex. rocket emoji) should be differentiated from standard English.
