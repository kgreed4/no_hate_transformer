import pandas as pd
import re

# Here we strip all characters that may be unnecessary for the model in the Twitter data
def strip_all_entities(text):
    return ' '.join(re.sub(r"(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)", " ", text).split())

def prepare_data(file_path, clean_tweets=False):
    """
    Purpose: Load the data, clean the tweets, and save the cleaned data to a new CSV file
    Input:
        file_path: str, path to the CSV file containing the raw data
        clean_tweets: bool, whether to clean the tweets or not
    """
    # Here we load in the data from the CSV file
    df = pd.read_csv(file_path)
    # Print the number of training sentences for better understanding and clarity 
    print(f'Number of training sentences: {df.shape[0]:,}\n')

    # If clean_tweets is True, we will clean the tweets
    if clean_tweets:
        # Here we clean the tweets by removing any Twitter entities using the function defined above
        df['tweet'] = df['tweet'].apply(strip_all_entities)

    # Now we want to combine classes 0 and 1 into class 0, rename class 2 to class 1
    df['class'] = df['class'].replace(1, 0)
    df['class'] = df['class'].replace(2, 1)

    # Save the modified DataFrame to a new CSV file
    df.to_csv('cleaned_labeled_data.csv', index=False)
    return df

if __name__ == "__main__":
    prepare_data('/Users/sveerisetti/Desktop/Duke_Spring/Deep_Learning/Projects/Project_2/Data/raw_data.csv', clean_tweets=False)
