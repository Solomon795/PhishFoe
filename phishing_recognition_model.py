import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import tensorflow as tf
import logging

tf.get_logger().setLevel(logging.ERROR)
print("TensorFlow version:", tf.__version__)
import pickle
from nltk.util import ngrams
import nltk

nltk.data.path.append('nltk_data')
import re
import numpy as np
import itertools
import argparse

from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer


# import logging

# 1. Load the Trained Model from the Pickle File:
# First, make sure you have your trained model saved in a pickle file.
# You can load it using the pickle module in Python.
# Assuming your model is named malicious_url_model.pkl, here’s how you can load it:


def load_model_email(model_path):
    # Load the email model from the specified path
    model = tf.keras.models.load_model(model_path)
    # Compile the model with default loss and optimizer to avoid the warning
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    with open('tokenizer.pickle', 'rb') as handle:
        tokenizer = pickle.load(handle)
    return model, tokenizer


def load_model_url(model_path):
    with open(model_path, 'rb') as url_model_file:
        model = pickle.load(url_model_file)
    return model


# 2. Parse Command-Line Arguments:
# Your standalone executable should accept command-line arguments for the list of URLs
# and the delimiter. You can use the argparse module to handle this.
# Here’s a basic example:
def parse_arguments():
    parser = argparse.ArgumentParser(description='Malicious URL Detection')
    parser.add_argument('--target', type=str, choices=['url', 'email'], required=True,
                        help='Target of the detection: "url" or "email"')
    parser.add_argument('input_list', type=str,
                        help='Path to the file containing URLs /email chunks or direct URLs /emails')
    parser.add_argument('--delimiter', type=str, default='\n', help='Delimiter for separating inputs')
    args = parser.parse_args()
    return args


# 3. Read URLs from the File: Read the URLs from the specified file
# (given by args.url_list). Split them based on the delimiter (defaulting to newline \n).

def read_inputs(input_list, delimiter='\n'):
    '''
    '''

    if os.path.isfile(input_list):
        with open(input_list, 'r') as input_file:
            inputs = input_file.read().split(delimiter)
    else:
        # Assume url_list is a direct argument containing URLs
        inputs = input_list.split(delimiter)
    return inputs


# # #
# np.set_printoptions(threshold=np.inf)
alphanum = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u',
            'v', 'w', 'x', 'y', 'z', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
permutations = itertools.product(alphanum, repeat=3)
featuresDict = {}
counter = 0
for perm in permutations:
    # print(perm)
    f = ''
    for char in perm:
        f = f + char
    featuresDict[(''.join(perm))] = counter
    counter = counter + 1


def generate_ngram_url(sentence):
    s = sentence.lower()
    s = ''.join(e for e in s if e.isalnum())  # replace spaces and slashes
    processedList = []
    for tup in list(ngrams(s, 3)):
        processedList.append((''.join(tup)))
    return processedList


def preprocess_sentences_url(url):
    X = np.zeros([1, 46656], dtype="int")
    url = url.strip().replace("https://", "")
    url = url.replace("http://", "")
    url = re.sub(r'\.[A-Za-z0-9]+/*', '', url)
    for gram in generate_ngram_url(url):
        try:
            X[0][featuresDict[gram]] = X[0][featuresDict[gram]] + 1
            # print('preprocess_sentences')
            # print(X[index][featuresDict[gram]])
        except:
            print(gram, "doesn't exist")
    return X


# # #
# 4. Predict Maliciousness for Each URL: Use your loaded model to predict whether
# each URL is malicious or not. You can assign a score between 1 and 10 or
# provide a percentage. Adjust the scoring logic based on your model’s output.
def predict_maliciousness_url(url, model):
    # Your model prediction logic here
    # Example: Replace this with your actual prediction code
    test = preprocess_sentences_url(url)
    pred = model.predict(test)
    probabilities = 1 / (1 + np.exp(-model.decision_function(test)))
    return pred, probabilities


def output_result_url(url_text, model):
    pred, probabilities = predict_maliciousness_url(url_text, model)
    if pred == 1:
        prediction = "Malicious"
    else:
        prediction = "Safe"
    print(f"URL: {url_text} | Prediction: {prediction} \n Malicious Score: {probabilities}\n")
    return url_text, probabilities


def analyze_urls(urls):
    url_results = []
    model_path = 'malicious_url_model.pkl'
    model = load_model_url(model_path)
    for url in urls:
        url_text, score = output_result_url(url, model)
        url_results.append((url_text, score))
    return url_results


"""Stemming and Tokenizer"""
# nltk.download('stopwords')
stop_words = stopwords.words('english')
stemmer = SnowballStemmer('english')

text_cleaning_re = "@\S+|https?:\S+|http?:\S+|[^A-Za-z0-9]:\S+|subject:\S+|nbsp"
""""""


def preprocess_email(text, stem=False):
    # Assuming 'preprocess' is your text cleaning function
    text = re.sub(text_cleaning_re, ' ', str(text).lower()).strip()
    tokens = []
    for token in text.split():
        if token not in stop_words:
            if stem:
                tokens.append(stemmer.stem(token))
            else:
                tokens.append(token)
    return " ".join(tokens)


def tokenize_and_pad_email(email_text, tokenizer, max_length=50):
    # Tokenize the email
    tokenized_text = tokenizer.texts_to_sequences([email_text])
    # Pad the tokenized text
    padded_text = tf.keras.utils.pad_sequences(tokenized_text, maxlen=max_length)
    return padded_text


def predict_email(email_text, model, tokenizer):
    # Preprocess the email text
    preprocessed_text = preprocess_email(email_text)
    # Tokenize and pad the email text
    padded_text = tokenize_and_pad_email(preprocessed_text, tokenizer)
    # Predict using the model
    prediction = model.predict(padded_text)
    return prediction


def extract_urls_from_email(email_text):
    # Use regex to extract URLs from email body
    url_pattern = r'\b((?:https?|ftp|file)://[-a-zA-Z0-9+&@#/%?=~_|!:,.;]*[-a-zA-Z0-9+&@#/%=~_|])'
    urls = re.findall(url_pattern, email_text)
    return urls


def analyze_emails(emails):
    email_results = []
    email_urls = []
    model_path = 'bi_lstm_phishing_email_TF_2.16.1.h5'
    model, tokenizer = load_model_email(model_path)
    for i, email in enumerate(emails):
        email_urls.extend(extract_urls_from_email(email))
        # Predict the email
        prediction = predict_email(email, model, tokenizer)
        email_results.append((i, prediction))
        # Output the prediction
        print(f"The probability of the {i} email being malicious is: {prediction[0][0]}")
    return email_results, email_urls


def format_as_csv(email_results=None, url_results=None, headers=True):
    output = []

    # Add headers if required
    if headers:
        output.append("ID,Type,Content,Malicious_Score")
    if email_results is not None:
        # Format email results
        for email_id, score in email_results:
            malicious_score = int(round(score.item() * 10))
            output.append(f"{email_id},email,{malicious_score}")

    if url_results is not None:
        # Format URL results
        for url, score in url_results:
            malicious_score = int(round(score.item() * 10))
            output.append(f",url,{url},{malicious_score}")

    return output


def main():
    print('''.--.
|__| .-------.
|=.| |.-----.|
|--| || KCK ||
|  | |'-----'|
|__|~')_____(''')
    print("\nPhishFoe is your digital shield, \nmeticulously analyzing and scoring the safety of emails and URLs\n")
    # step 1
    args = parse_arguments()
    # step 2
    inputs = read_inputs(args.input_list, delimiter=args.delimiter)
    # step 3
    if args.target == 'url':
        url_results = analyze_urls(inputs)
        csv_output = format_as_csv(url_results=url_results, headers=True)
    else:
        email_results, email_urls = analyze_emails(inputs)
        csv_output = format_as_csv(email_results=email_results, headers=True)
        if len(email_urls) > 0:
            print(f"Extracted URLs from emails:\n{email_urls}\n")
            # Ask user if they want to predict the maliciousness of these URLs
            user_response = input("Do you want to predict the maliciousness of these URLs? (yes/no): ")
            if user_response.lower() == 'yes':
                email_url_results = analyze_urls(email_urls)
                csv_output = format_as_csv(email_results=email_results, url_results=email_url_results, headers=True)
            else:
                print("\nThank you for using PhishGuard!")
    for line in csv_output:
        print(line)


if __name__ == "__main__":
    main()
