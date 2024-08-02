import pandas as pd
import numpy as np
import nltk
import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS
import string
from nltk.corpus import stopwords
import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import BernoulliNB

# Load dataset
file_path = "stress.csv"
data = pd.read_csv(file_path)
print(data.head())

# Check for null values
print(data.isnull().sum())

# Download stopwords
nltk.download('stopwords')

# Initialize stemmer and stopwords
stemmer = nltk.SnowballStemmer("english")
stopword = set(stopwords.words('english'))

# Clean text function
def clean(text):
    text = str(text).lower()
    text = re.sub('\[.*?\]', '', text)
    text = re.sub('https://\S+|www\.\S+', '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\n', '', text)
    text = re.sub('\w*\d\w*', '', text)
    text = [word for word in text.split(' ') if word not in stopword]
    text = " ".join(text)
    text = [stemmer.stem(word) for word in text.split(' ')]
    text = " ".join(text)
    return text

# Apply cleaning to the "text" column
data["text"] = data["text"].apply(clean)

# Generate WordCloud
cleaned_text = " ".join(data["text"])
stopwords = set(STOPWORDS)
wordcloud = WordCloud(stopwords=stopwords, background_color="white").generate(cleaned_text)
plt.figure(figsize=(15, 10))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()

# Map labels to "No Stress" and "Stress"
data["label"] = data["label"].map({0: "No Stress", 1: "Stress"})

# Select relevant columns
data = data[["text", "label"]]
print(data.head())

# Prepare data for training
x = np.array(data["text"])
y = np.array(data["label"])

# Vectorize the text using CountVectorizer
cv = CountVectorizer()
X = cv.fit_transform(x)

# Split the data into training and testing sets
xtrain, xtest, ytrain, ytest = train_test_split(X.toarray(), y, test_size=0.33, random_state=42)

# Train a Naive Bayes model
model = BernoulliNB()
model.fit(xtrain, ytrain)

# Process user input
def process_user_input(model, cv):
    user_input = input("Enter a Text: ")
    cleaned_input = clean(user_input)
    data_input = cv.transform([cleaned_input]).toarray()
    output = model.predict(data_input)
    print("Prediction:", output[0])

# Make predictions using the trained model
process_user_input(model, cv)

# Make predictions using the trained model
process_user_input(model, cv)
