# Sentiment Analyzer tool for Movie Reviews

## Description

This project implements a sentiment analyzer for movie reviews using OpenAI's GPT-3.5 and a dataset from IMDb. It preprocesses the data, requests the openai API for a sentiment based on a prompt, and evaluates its performance using metrics like accuracy, precision, recall, and F1-score.

## How to use it

1. Clone this repository
2. Install needed libraries

```bash
pip install openai pandas scikit-learn
```

3. Install the dataset from https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews and keep in same directory as generate_clean_data.py
4. First run the generate_clean_data.py script to clean the data and choose how many reviews to analyse
5. Secondly enter your OpenAI private key to access the model in main.py and run the script.
6. main.py will analyse the reviews and give a short summary of the results.
