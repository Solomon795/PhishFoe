import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import tensorflow
import keras
import logging

tensorflow.get_logger().setLevel(logging.ERROR)
print("TensorFlow version:", tensorflow.__version__)
import pickle
from nltk.util import ngrams
import nltk
import json
nltk.data.path.append('nltk_data')
import re
import numpy as np
import itertools
import argparse
import prettytable as pt
from googletrans import Translator
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer

def load_model_email(model_path):
    # Load the email model from the specified path
    model = keras.models.load_model(model_path)
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
    parser = argparse.ArgumentParser(description='PhishFoe - Malicious Email & URL Detection',
                                     epilog="Example:\n"
                                            "python phishing_recognition_model.py -target email emails.txt\n\n"
                                            "For more information, visit our documentation at https://github.com/Solomon795/PhishFoe.git",
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('-t', '--target', type=str, choices=['url', 'email', 'email_with_url'], required=True,
                        help='Target of the detection: "url" or "email"')
    parser.add_argument('-i', '--input', type=str,
                        help='Path to the file containing URLs /email chunks or direct URLs /emails')
    parser.add_argument('-d', '--delimiter', type=str, default='\n', help='Delimiter for separating inputs')
    parser.add_argument('-l', '--language', type=str, choices=['english', 'german', 'hebrew'], default='english',
                        help='Language of the input data')
    args = parser.parse_args()
    return args


# 3. Read URLs from the File: Read the URLs from the specified file
# (given by args.url_list). Split them based on the delimiter (defaulting to newline \n).

def read_inputs(input_list, delimiter='\n'):
    if os.path.isfile(input_list):
        with open(input_list, 'r', encoding='utf-8') as input_file:
            inputs = input_file.read().split(delimiter)
    else:
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
# Load stopwords and stemmers for English, German, and Hebrew
stop_words = stopwords.words('english')
stemmer = SnowballStemmer('english')

text_cleaning_re = "@\S+|https?:\S+|http?:\S+|[^A-Za-z0-9]:\S+|subject:\S+|nbsp"
""""""

# Initialize the Google Translate API
translator = Translator()

def preprocess_email(text, language='english', stem=False):
    # Assuming 'preprocess' is your text cleaning function (clean the text)
    if language != 'english':
        # Translate text to English
        translated = translator.translate(text, src=language, dest='en')
        text = translated.text

    # Clean the text
    text = re.sub(text_cleaning_re, ' ', str(text).lower()).strip()

    # Tokenize text
    tokens = nltk.word_tokenize(text)

    processed_tokens = []
    for token in tokens:
        if token not in stop_words:
            if stem:
                processed_tokens.append(stemmer.stem(token))
            else:
                processed_tokens.append(token)

    return " ".join(processed_tokens)
    # tokens = []
    # for token in text.split():
    #     if token not in stop_words:
    #         if stem:
    #             tokens.append(stemmer.stem(token))
    #         else:
    #             tokens.append(token)
    # return " ".join(tokens)



def tokenize_and_pad_email(email_text, tokenizer, max_length=50):
    # Tokenize the email
    tokenized_text = tokenizer.texts_to_sequences([email_text])
    # Pad the tokenized text
    padded_text = keras.utils.pad_sequences(tokenized_text, maxlen=max_length)
    return padded_text


def predict_email(email_text, model, tokenizer, language='english'):
    # Preprocess the email text
    preprocessed_text = preprocess_email(email_text, language=language, stem=True)
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


def analyze_emails(emails, language='english'):
    email_results = []
    email_urls = []
    model_path = 'bi_lstm_phishing_email_TF_2.16.1.h5'
    model, tokenizer = load_model_email(model_path)
    for i, email in enumerate(emails):
        email_urls.extend(extract_urls_from_email(email))
        # Predict the email
        prediction = predict_email(email, model, tokenizer, language=language)
        email_results.append((i, emails[i], prediction))
        # Output the prediction
        print(f"The probability of the {i} email being malicious is: {prediction[0][0]}")
    return email_results, email_urls


def clip_text(text, max_length=50):
    if len(text) > max_length:
        return text[:max_length] + "..."
    return text


def format_as_table(email_results=None, url_results=None):
    table = pt.PrettyTable()
    table.field_names = ["ID", "Type", "Content", "Malicious_Score"]

    counter = 1

    if email_results is not None:
        for email_id, content, score in email_results:
            malicious_score = float(round(score.item() * 10, 1))
            clipped_content = clip_text(content)
            table.add_row([counter, "email", clipped_content, malicious_score])
            counter += 1

    if url_results is not None:
        for url_id, (url, score) in enumerate(url_results):
            malicious_score = float(round(score.item() * 10, 1))
            table.add_row([counter, "url", url, malicious_score])
            counter += 1

    return table

def format_as_json(email_results=None, url_results=None):
    output = []

    if email_results is not None:
        for email_id, content, score  in email_results:
            malicious_score = float(round(score.item() * 10))
            output.append({
                "ID": email_id,
                "Type": "email",
                "Content": content,
                "Malicious_Score": malicious_score
            })

    if url_results is not None:
        for url_id, (url, score) in enumerate(url_results):
            malicious_score = float(round(score.item() * 10))
            output.append({
                "ID": url_id,
                "Type": "url",
                "Content": url,
                "Malicious_Score": malicious_score
            })

    return json.dumps(output, indent=4)

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
    print(args)
    # step 2
    inputs = read_inputs(args.input, delimiter=args.delimiter)
    # step 3
    if args.target == 'url':
        url_results = analyze_urls(inputs)
        table_output = format_as_table(url_results=url_results)
        json_output = format_as_json(url_results=url_results)
    else:
        email_results, email_urls = analyze_emails(inputs, language=args.language)
        table_output = format_as_table(email_results=email_results)
        json_output = format_as_json(email_results=email_results)
        if args.target == 'email_with_url' and len(email_urls) > 0:
            print(f"Extracted URLs from emails:\n{email_urls}\n")
            # # Ask user if they want to predict the maliciousness of these URLs
            # user_response = input("Do you want to predict the maliciousness of these URLs? (yes/no): ")
            # if user_response.lower() == 'yes':
            email_url_results = analyze_urls(email_urls)
            table_output = format_as_table(email_results=email_results, url_results=email_url_results)
            json_output = format_as_json(email_results=email_results, url_results=email_url_results)

    print(table_output)
    with open('recognition_output.json', 'w', encoding='utf-8') as json_file:
        json_file.write(json_output)


if __name__ == "__main__":
    main()
