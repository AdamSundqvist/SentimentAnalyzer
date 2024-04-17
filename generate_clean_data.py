import pandas as pd
import re
from sklearn.model_selection import train_test_split
from bs4 import BeautifulSoup

#   Remove html elements
def remove_htmltag(text):
    soup = BeautifulSoup(text, "html.parser")
    return soup.get_text(separator=" ")

#   Functions name says it all... 
def remove_special_characters(text):
    # Remove double quotes
    text = text.replace('"', '')
    # Remove special characters
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    return text

#Choose how many reviews to analyse
data_length = 100

#Load IMDB dataset
data = pd.read_csv("IMDB Dataset.csv")

#Make data for exploring shorter
train_df, _ = train_test_split(data, train_size=data_length, stratify=data['sentiment'])

# Save the train set to a new CSV file
train_df.to_csv("balanced_subset.csv", index=False)

#   Read shorted down data
short_data = pd.read_csv("balanced_subset.csv")

#   Apply cleaning methods
short_data["review"] = short_data["review"].apply(remove_htmltag)
short_data["review"] = short_data["review"].apply(remove_special_characters)

#   Write to new document
short_data.to_csv("cleaned_data.csv", index=False)