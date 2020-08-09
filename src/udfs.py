import re
import emoji
import string
from nltk.tokenize import word_tokenize
from pyspark.sql import functions, types
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

@functions.udf(returnType=types.ArrayType(types.StringType()))
def get_urls(input_string):
    return re.findall('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+',\
                     str(input_string))

@functions.udf(returnType=types.ArrayType(types.StringType()))
def get_user_mentions(input_string):
    return re.findall('@([^\s:]+)', str(input_string))

@functions.udf(returnType=types.ArrayType(types.StringType()))
def get_hashtags(input_string):
    return re.findall('#(\w+)', str(input_string))

@functions.udf(returnType=types.ArrayType(types.StringType()))
def get_capitalized_words(input_string):
    return re.findall('([A-Z]+(?:(?!\s?[A-Z][a-z])\s?[A-Z])+)', str(input_string))

@functions.udf(returnType=types.ArrayType(types.StringType()))
def get_emojis(input_string):
    list_of_emojis = list()
    for char in str(input_string):
        if char in emoji.UNICODE_EMOJI:
            list_of_emojis.append(char)
    return list_of_emojis

@functions.udf(returnType=types.ArrayType(types.StringType()))
def get_emoji_sequence(input_string):
    # This function will return ocurrences of emoji sequences of length greater than 1
    list_of_emojis = list()
    emote_seq = str()
    flag = 0
    for idx, char in enumerate(str(input_string)):
        if char in emoji.UNICODE_EMOJI:
            flag = 1
            emote_seq += char
            if idx == len(input_string)-1:
                list_of_emojis.append(emote_seq)
        else:
            if flag == 1 and emote_seq != '':
                if len(emote_seq) > 1:
                    list_of_emojis.append(emote_seq)
                flag = 0
                emote_seq = str()
            else:
                continue
    return list_of_emojis

@functions.udf(returnType=types.FloatType())
def get_sentiment_score(input_string):
    analyzer = SentimentIntensityAnalyzer()
    return analyzer.polarity_scores(str(input_string))['compound']

@functions.udf(returnType=types.StringType())
def clean_tweets(input_string):
    from nltk.corpus import stopwords
    
    if str(input_string) is None:
        input_string = ''
    input_string = str(input_string).lower()
    input_string = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+',\
                         '', str(input_string))
    
    #Emoji patterns
    emoji_pattern = re.compile("["
         u"\U0001F600-\U0001F64F"  
         u"\U0001F300-\U0001F5FF"  
         u"\U0001F680-\U0001F6FF"  
         u"\U0001F1E0-\U0001F1FF"  
         u"\U00002702-\U000027B0"
         u"\U000024C2-\U0001F251"
         "]+", flags=re.UNICODE)

    # Remove emojis from tweet
    input_string = emoji_pattern.sub(r'', str(input_string))
    stop_words = set(stopwords.words('english'))
    
    # You may choose to add custom stopwords to this list if you would like them to be filtered out
    custom_stop = ['covid19', "''", "’", "``", '”', "covid-19"]
    
    for _ in custom_stop:
        stop_words.add(_)
    word_tokens = word_tokenize(str(input_string))
    input_string = re.sub(r':', '', str(input_string))
    input_string = re.sub(r'‚Ä¶', '', str(input_string))
    
    # Replace consecutive non-ASCII characters with a space
    input_string = re.sub(r'[^\x00-\x7F]+',' ', str(input_string))
    
    # Filter using NLTK library append it to a string
    filtered_tweet = [w for w in word_tokens if not w in stop_words]
    filtered_tweet = []
    
    for w in word_tokens:
    # Check tokens against stop words and punctuations
        if w not in stop_words and w not in string.punctuation and w != "'s":
            filtered_tweet.append(w)
    return ' '.join(filtered_tweet)
