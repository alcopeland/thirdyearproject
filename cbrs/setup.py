from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
import string

import csv

from sklearn.pipeline import Pipeline
from datasets import DatasetDict
from pathlib import Path
from train_classifier import load_dataset_from_csv, train_naive_bayes

def clean_text(text: str) -> str:
    # Takes a string and cleans it
    # Tokenises the text provided
    tokenizedText = word_tokenize(text)
    ps = PorterStemmer()
    # Gets the list of stopwords for English from the nltk corpus
    stop_words = set(stopwords.words('english'))
    cleanText = []
    # Checks every word in the tokenised form to see if it is a stopword
    for word in tokenizedText:
        # If it is not a stopword, append the stem of the word to a list
        if word not in stop_words:
            cleanText.append(ps.stem(word))
    # Reform the new string in stemmed form
    cleanTextS = ' '.join(cleanText)
    # Remove any punctuation
    cleanTextS = cleanTextS.translate(str.maketrans('', '', string.punctuation))
    return cleanTextS

def write_to_csv(names: list, labels: list, texts: list, original_texts: list) -> None:
    # Creates a 2d array of all the data together
    combinedArray = list(zip(names,labels,texts,original_texts))
    # Opens a new file called "final_text.csv" and writes all the data into the file
    with open('final_text.csv', 'w+', encoding="utf-8", errors="ignore", newline='') as file:
        writer = csv.writer(file)        
        writer.writerow(["name","label","text","original"])
        writer.writerows(combinedArray)    

def read_booksdataset() -> tuple[list, list, list, list]:
    # Creates empty lsits to store the data from the dataset
    names, labels, texts, original_texts = [], [], [], []
    # Opens the dataset file in csv format
    with open("BooksDataSet.csv", "r", encoding="utf-8", errors="ignore") as file:
        # Creates a csv file reader
        csv_reader = csv.reader(file)
        # Skips the title row
        next(file)
        # For every other line in the file extract the book name, genre, original text summary and cleaned text summary
        for row in csv_reader:
            names.append(row[2])
            cleanText = clean_text(row[4])
            labels.append(row[3])
            texts.append(f"{row[2]} {row[3]} {cleanText}")
            original_texts.append(row[4])
    # Return the data
    return names, labels, texts, original_texts

def system_setup(run: int) -> tuple[Pipeline, DatasetDict]:
    # Checks to see if the file "final_text.csv" exists in the directory
    file = Path("final_text.csv")
    # If the file does not exist, or the debug tag is 1, create the clean dataset by reading the original file and writing to the new one
    if file.is_file() == False or run == 1:
        # Reads the original dataset and stores the desired information
        names, labels, texts, original_texts = read_booksdataset()
        # Writes the new data to a new file
        write_to_csv(names, labels, texts, original_texts)
    # Loads the dataset and splits it by the float value provided into train and test sets
    full_dataset, split_dataset = load_dataset_from_csv(size=0.15)
    # Creates a naive bayes model on the splot dataset
    model = train_naive_bayes(split_dataset)
    # Returns the model and full dataset
    return model, full_dataset