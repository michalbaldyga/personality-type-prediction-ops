import os
import csv

from clear_tweets import clear_data

INPUT_DIR = '../labeling/input_data'
TRAIN_FILE_PATH = 'output_data/training_data.csv'
TEST_FILE_PATH = 'output_data/testing_data.csv'

def get_tweets(directory):
    """
    Gets tweets from all files in given directory
    :param directory: directory with input files
    :type directory: str
    :return: list of tweets
    :rtype: list[str]
    """
    tweets_list = {}

    for file in os.listdir(directory):
        file_path = os.path.join(directory, file)
        tweets_list[file_path] = []

        with open(file_path, encoding="utf-8") as tweets_file:
            for line in tweets_file.readlines()[1:]:
                cleared_tweet = clear_data(line[:-1])
                if len(cleared_tweet) > 0 and cleared_tweet != ' ':
                    tweets_list[file_path].append(cleared_tweet)

    return tweets_list


def save_labeled_tweets(tweets_dict):
    """
    Save tweets with labels to output csv file
    :param tweets_dict: dictionary with tweets assigned to analyzed types
    :type tweets_dict: dict
    :return: None
    """
    training_data = []
    testing_data = []

    for ops in tweets_dict.keys():
        idx = 0
        partition_idx = int(len(tweets_dict[ops]) * 0.8)

        for account in tweets_dict[ops].keys():
            merged_tweets = ' '.join(tweets_dict[ops][account])

            if idx < partition_idx:
                training_data.append(f"{ops.upper()}|{merged_tweets}")
            else:
                testing_data.append(f"{ops.upper()}|{merged_tweets}")

            idx += 1

    with open(TRAIN_FILE_PATH, 'w', encoding='utf-8') as f:
        f.write("label|text\n")
        for tweet in training_data:
            f.write(f"{tweet}\n")
    with open(TEST_FILE_PATH, 'w', encoding='utf-8') as f:
        f.write("label|text\n")
        for tweet in testing_data:
            f.write(f"{tweet}\n")


if __name__ == '__main__':
    output_dict = {}

    for subdir in [f.path for f in os.scandir(INPUT_DIR) if f.is_dir()]:
        tweets = get_tweets(subdir)
        output_dict[os.path.split(subdir)[-1]] = tweets

    save_labeled_tweets(output_dict)

