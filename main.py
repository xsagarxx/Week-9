import sys
import threading
import time

from mpi4py import MPI
import pandas as pd
from textblob import TextBlob
from pyspark import SparkConf, SparkContext
from pyspark.sql import SparkSession


def get_sentiment(text):
    """
    Returns the sentiment polarity of the given text.
    """
    blob = TextBlob(text)
    return blob.sentiment.polarity


def process_data(data):
    """
    Process the given data by computing the sentiment polarity for each tweet.
    """
    sentiments = []
    for tweet in data:
        sentiment = get_sentiment(tweet)
        sentiments.append(sentiment)
    return sentiments


def compute_sentiments_parallel(data):
    """
    Compute the sentiment polarity for each tweet in parallel using threads.
    """
    num_threads = 4
    chunk_size = int(len(data) / num_threads)

    threads = []
    results = []

    # Create and start threads
    for i in range(num_threads):
        start_idx = i * chunk_size
        end_idx = start_idx + chunk_size
        if i == num_threads - 1:
            end_idx = len(data)
        thread = threading.Thread(target=process_data, args=(data[start_idx:end_idx],))
        threads.append(thread)
        thread.start()

    # Wait for threads to finish
    for thread in threads:
        thread.join()

    # Retrieve results from threads
    for thread in threads:
        results.extend(thread.result)

    return results


def compute_sentiments_mpi(data):
    """
    Compute the sentiment polarity for each tweet in parallel using MPI.
    """
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # Split data into chunks
    chunk_size = int(len(data) / size)
    start_idx = rank * chunk_size
    end_idx = start_idx + chunk_size
    if rank == size - 1:
        end_idx = len(data)

    # Process data chunk
    sentiments = process_data(data[start_idx:end_idx])

    # Gather results from all processes
    results = comm.gather(sentiments, root=0)

    # Flatten the results
    if rank == 0:
        all_results = []
        for result in results:
            all_results.extend(result)
        return all_results


def compute_sentiments_spark(data):
    """
    Compute the sentiment polarity for each tweet in parallel using Spark.
    """
    conf = SparkConf().setAppName("SentimentAnalysis").setMaster("local[*]")
    sc = SparkContext(conf=conf)
    spark = SparkSession.builder.appName("SentimentAnalysis").getOrCreate()

    # Convert data to RDD
    rdd = sc.parallelize(data)

    # Process data using Spark transformations and actions
    sentiments = rdd.map(get_sentiment).collect()

    # Stop Spark context
    sc.stop()

    return sentiments


if __name__ == "__main__":
    # Load social media data
    data = pd.read_csv("social_media_data.csv")["text"].tolist()

    # Compute sentiments using threads
    start_time = time.time()
    results_thread = compute_sentiments_parallel(data)
    end_time = time.time()
    print(f"Time taken using threads: {end_time - start_time:.2f} seconds")

    # Compute sentiments using MPI
    start_time = time.time()
    results_mpi = compute_sentiments_mpi(data)
    end_time = time.time()
    print(f"Time taken using MPI: {end_time - start_time:.2f} seconds")

    # Compute sentiments using Spark
    start_time = time.time
if __name__ == '__main__':
    # initialize SparkContext
    sc = SparkContext(appName="SentimentAnalysis")

    # read input data from HDFS
    input_file = sys.argv[1]
    input_rdd = sc.textFile(input_file)

    # preprocess the data
    preprocessed_rdd = input_rdd.map(preprocess)

    # calculate sentiment scores using Vader
    sentiment_scores_rdd = preprocessed_rdd.map(lambda x: (x[0], calculate_sentiment_score(x[1])))

    # calculate overall sentiment for each tweet
    overall_sentiment_rdd = sentiment_scores_rdd.map(lambda x: (x[0], calculate_overall_sentiment(x[1])))

    # write output to HDFS
    output_file = sys.argv[2]
    overall_sentiment_rdd.saveAsTextFile(output_file)

    # stop SparkContext
    sc.stop()
