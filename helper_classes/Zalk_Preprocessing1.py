#!/usr/bin/env python
# coding: utf-8

# In[5]:


import re
import numpy as np
from langdetect import detect

class tweetpreprocess():
    
    def __init__(self, tweet_string):
        self.s = tweet_string
        self.cleantweet = self.clean()
        self.get_clean_tweet()
            
    def clean(self):
        def spell_correction_algo(x):
            sample1 = re.sub(r'(\w)\1(\1+)',r'\1\1',x) # Every sequence of characters >= 3 in a word is replaced by 2 of those chars
            sample2 = re.sub(r'(\w)\1(\1+)',r'\1',x) # The unique characters in a word, in order of appearance

            if (sample1 != sample2): # If the word HAS a sequence of characters >=3, execute
                #print('algo run on ' + x)
                word_algo = [] 
                word = ''
                marker = 0
                word_algo.append(sample1) # Add the first sample string 

                for char in list(sample2): # For every unique character in the word
                    if sample1.count(char) == 2: # If it appears twice (3+ times in original string)
                        word_algo.append(sample1.replace(char,'',1)) # Remove one of its occurances and add resulting string

                while (sample2 not in word_algo): # while the unique character string hasn't been added by the algorithm yet
                    repeat_char = ''
                    for char in list(sample2):
                        word += char # Add unique character
                        if (marker == 0):
                            if sample1.count(char) == 2: # If it appears twice (3+ times in original string)
                                marker += 1 # Increment marker to prevent any more characters being added twice to our word
                                word = word + char # Add it again to our word
                                repeat_char = char # Mark it has a repeated character
                    word_algo.append(word) 
                    sample1 = sample1.replace(repeat_char,'') # remove the repeated char from sample1 string
                    # this is so the algorithm doesn't run again with the same repeated char
                    marker = 0 # Reset marker
                    word = '' # Reset word

                word_algo = np.unique(word_algo) # Saftey check
                return ' '.join(word_algo)

            else:
                return sample1
            
        s = str(self.s)
        
        s = s.replace("'", "") # Remove any single quotes

        # Uppercase -> Lowercase. This might or might not be a good idea but is necessary for stemming using 
        # typical packages such as porterstemmer. Capitalization could be indicative of sentiment. 
        s = s.lower()

        try:
            lang = detect(s) # If the string is not (most likely) english
        # Don't isolate only english because a tweet can be numbers + url and should throw an exception
        except: 
            #print("Non english text: "+s)
            return ' '

        if(lang == 'en'):
            # Delete all digits from a string, but keep the digits contained in a word
            s = re.sub(r'^\d+\s|\s\d+\s|\s\d+$',' ',s) 
            s = s.replace('3','e') # People use '3' as 'e' a lot. 
            s = s.replace('5','s') # Same with '5' as 's'

            # Exclamation marks (!) and question marks (?) also carry some sentiment. 
            s = s.replace('!',' zzexclaimzz ') # Note that there is also an emoji exclamation mark
            s = s.replace('?',' zzquestzz ') 

            # Spell Correction
            s = spell_correction_algo(s)

            # Replace URL (literal text: 'http', and therefore 'https', until next whitespace) - this could be edited more to account for more use cases
            s = re.sub(r'http\S+',' ',s)
            # Replace URL (literal text: 'www' until next whitespace)
            s = re.sub(r"www\S+",' ',s)
            # Replace Targets (@s)
            s = re.sub(r'@\S+',' ',s)
            # Replace Hashtags (#s)
            s = re.sub(r'#\S+',' ',s)
            # Remove special characters and numbers (things that aren't numbers)
            # Although these might be present (accidently or not) in the middle of word strings, we can ignore this case as it shouldn't necessarily mess things up
            # I sub with a space here because it should work for most use cases
            # I also subbed quotes (', '') with no space at the beginning (e.g. 'can't' is already 'cant')
            s = re.sub('[^a-z]+',' ',s)

            s = re.sub(r'\W*\b\w{1,2}\b',' ',s) # Remove words less than or equal to two characters long

            return s

        else: return ' '
        
    def get_clean_tweet(self):
        #print(self.cleantweet)
        return self.cleantweet

