import os
import udfs as u
import pandas as pd
import seaborn as sns
from pyspark.ml import feature
import matplotlib.pyplot as plt
from pyspark import SparkContext
from bots import get_botometer_score
from pyspark.sql import SQLContext, SparkSession, functions

data_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', 'data')

def spark_init():
    ''' Initializes Spark and returns spark context, sql context and spark session.
    '''
    sc = SparkContext.getOrCreate()
    sqlContext = SQLContext(sc)
    spark = SparkSession.builder.appName('covid-twitter').getOrCreate()
    return sc, sqlContext, spark

def extract_basics(df):
    '''
    This function takes in a dataframe containing tweets and extracts items such as URLs, user mentions, hashtags, 
    capitalized words and emojis using the user defined functions declared in udfs.py.
    '''
    df = df.withColumn('URLs', u.get_urls(df.text))
    df = df.withColumn('user_mentions', u.get_user_mentions(df.text))
    df = df.withColumn('hashtags', u.get_hashtags(df.text))
    df = df.withColumn('caps_words', u.get_capitalized_words(df.text))
    df = df.withColumn('emojis', u.get_emojis(df.text))
    df = df.withColumn('emoji_sequence', u.get_emoji_sequence(df.text))
    df = df.withColumn('sentiment_score', u.get_sentiment_score(df.text))
    df = df.withColumn('clean_text', u.clean_tweets(df.text))
    df = df.withColumn('words', u.functions.split(df.clean_text, ' '))
    return df

def get_topN_of_col(df, col_name, n):
    '''
    This function takes in a pySpark dataframe along with its column name as string
    and returns a pandas dataframe consisting of top n occurrances of the column along
    with its count.
    '''
    col_name_df = df.select(functions.explode(df[col_name]).alias(col_name))
    aggregated = col_name_df.groupBy('{}'.format(col_name)).count().sort(functions.desc('count'))
    return pd.DataFrame(aggregated.take(n), columns=[str(col_name), 'count'])

def get_frequent_ngrams_containing(search_term, ngram_df, col_name, n):
    '''
    This function will take a string, search_term, which is the word of interest, ngram 
    Spark dataframe along with column name containing the ngrams. Function will return 
    top N frequent ngrams that has the search_term in it.
    '''
    exploded_df = ngram_df.select(functions.explode(ngram_df['{}'.format(col_name)]).alias('exploded'))
    # Note: The rlike matches with Java regex
    exploded_df = exploded_df.withColumn('contains',\
                              exploded_df.exploded.rlike('.*\\b{}\\b.*'.format(search_term)))
    filtered_df = exploded_df.filter(exploded_df['contains'] == 'true')
    aggregated = filtered_df.groupBy('exploded').count().sort(functions.desc('count')) 
    return pd.DataFrame(aggregated.take(n), columns=[str(col_name), 'count'])

def main():
    sc, sqlContext, spark = spark_init()
    df = spark.read.csv(os.path.join(data_dir, 'tcat*.csv'), header=True)
    # Removing the retweet indicator RT from tweets
    df = df.withColumn('text', functions.regexp_replace('text', 'RT', ''))
    df = extract_basics(df)
   
    from_users_df = pd.DataFrame(df.groupBy('from_user_name').count()\
                    .sort(functions.desc('count')).take(1000), columns = ['from_user_name', 'count'])
    from_users_df.to_csv(os.path.join(data_dir, 'frequent_users.csv'), index=False)

    from_users_df['botometer_scores'] = from_users_df['from_user_name'].apply(get_botometer_score)
    bots_df = from_users_df[from_users_df['botometer_scores'] >= 3]
    bots_df.to_csv(os.path.join(data_dir, 'bots.csv'), index=False)

    user_mentions_df = get_topN_of_col(df, 'user_mentions', 50)
    user_mentions_df.to_csv(os.path.join(data_dir, 'frequent_user_mentions.csv'), index=False)

    bot_set = set(bots_df['from_user_name'].to_list())
    df = df.where(functions.col('from_user_name').isin(bot_set))

    hashtags_df = get_topN_of_col(df, 'hashtags', 50)
    hashtags_df.to_csv(os.path.join(data_dir, 'frequent_hashtags.csv'), index=False)

    emojis_df = get_topN_of_col(df, 'emojis', 50)
    emojis_df.to_csv(os.path.join(data_dir, 'frequent_emojis.csv'), index=False)

    emoji_sequence_df = get_topN_of_col(df, 'emoji_sequence', 50)
    emoji_sequence_df.to_csv(os.path.join(data_dir, 'frequent_emoji_sequences.csv'), index=False)

    words_df = get_topN_of_col(df, 'words', 50)
    words_df.to_csv(os.path.join(data_dir, 'frequent_words.csv'), index=False)

    bigram = feature.NGram(n=2, inputCol='words', outputCol='BiGrams').transform(df.select(df.words))
    trigram = feature.NGram(n=3, inputCol='words', outputCol='TriGrams').transform(df.select(df.words))
    bigram_pd_df = get_topN_of_col(bigram, 'BiGrams', 20)
    trigram_pd_df = get_topN_of_col(trigram, 'TriGrams', 20)
    bigram_pd_df.to_csv(os.path.join(data_dir, 'frequent_bigrams.csv'), index=False)
    trigram_pd_df.to_csv(os.path.join(data_dir, 'frequent_trigrams.csv'), index=False)

    # To retrieve ngrams containing specific search term
    get_frequent_ngrams_containing('home', bigram, 'BiGrams', 20)
    get_frequent_ngrams_containing('safe', trigram, 'TriGrams', 20)

    # Descriptive statistics of sentiment scores
    stats_df = df.select(functions.mean(df.sentiment_score).alias('mean'),\
                     functions.min(df.sentiment_score).alias('min'),\
                     functions.max(df.sentiment_score).alias('max'),\
                     functions.stddev(df.sentiment_score).alias('stddev'),\
                     functions.variance(df.sentiment_score).alias('variance'))
    stats_pd_df = stats_df.toPandas()
    stats_pd_df.to_csv(os.path.join(data_dir, 'sentiment_stats.csv'), index=False)

    quantiles = df.approxQuantile(col='sentiment_score', probabilities=[0.0, 0.25, 0.5, 0.75, 1.0],\
                                 relativeError=0.05)
    sns.set(style='darkgrid', palette='pastel')
    plt.figure(figsize=(16, 6))
    sns.boxplot(palette=['m'], data=quantiles, orient='h')
    plt.savefig(os.path.join(data_dir, 'sentiment_score_boxplot.png'))
    plt.close()


if __name__ == '__main__':
    main()
