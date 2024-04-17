import pandas as pd
import openai
from sklearn.metrics import confusion_matrix

#   OpenAI key
mykey = "ENTER PRIVATE KEY HERE"
openai.api_key = mykey

#   Read cleaned data
data = pd.read_csv("cleaned_data.csv")

#   Prompt gpt model
pred_sentiments = []
sentiments = data["sentiment"]

for idx, row in data.iterrows():
    txt = row["review"]
    response = openai.completions.create(
    model="gpt-3.5-turbo-instruct",
    prompt=f"Sentiment analysis on the text: '{txt}'\nSentiment score:",
    temperature=0,
    max_tokens=1,
    )
    pred_sentiments.append(response.choices[0].text.lower().strip())

#   Combine true and predicted values
sentiments = pd.concat([sentiments, pd.Series(pred_sentiments)], axis=1)

#   Rename the column
sentiments.columns = ["sentiment", "predicted"]

#   Drop all occurences where we might have a prediction that is not "positive" or "negative"
sentiments = sentiments[(sentiments['predicted'].isin(['positive', 'negative']))]

#   Metrics 
confmat = confusion_matrix(sentiments["sentiment"], sentiments["predicted"])

tn, fp, fn, tp = confmat.ravel()

accuracy = (tp + tn) / (tp + tn + fp + fn)
precision = tp / (tp + fp)
recall = tp / (tp + fn)
f1_score = 2 * (precision * recall) / (precision + recall)

print(f"Confusion Matrix: \n {confmat}")
print(f'Accuracy: {accuracy}')
print(f'Precision: {precision}')
print(f'Recall: {recall}')
print(f'F1-score: {f1_score}')