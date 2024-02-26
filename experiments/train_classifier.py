from datasets import DatasetDict, load_dataset

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline, make_pipeline    

from sklearn.metrics import accuracy_score

def load_dataset_from_csv(size: float) -> tuple[DatasetDict, DatasetDict]:
    # Method found from link below
    # https://medium.com/@lokaregns/fine-tuning-transformers-with-custom-dataset-classification-task-f261579ae068
    # Load the csv file into a Dataset type
    dataset = load_dataset('csv', data_files = 'final_text.csv')
    # The dataset is then split into train and test datasets based on the float value provided
    split_dataset = dataset['train'].train_test_split(test_size=size)
    return dataset, split_dataset

def train_naive_bayes(dataset: DatasetDict) -> Pipeline:
    # Method found from link below
    # https://www.turing.com/kb/document-classification-using-naive-bayes
    # Makes a pipeline for TF-IDF vectorisation and Naive Bayes
    model = make_pipeline(TfidfVectorizer(), MultinomialNB())
    # Trains the model on the train split
    model.fit(dataset["train"]["text"],dataset["train"]["label"])
    return model

def test_naive_bayes(dataset: DatasetDict) -> float:
    # Method found from link below
    # https://www.turing.com/kb/document-classification-using-naive-bayes
    # Makes a pipeline for TF-IDF vectorisation and Naive Bayes
    model = make_pipeline(TfidfVectorizer(), MultinomialNB())
    # Trains the model on the train split
    model.fit(dataset["train"]["text"],dataset["train"]["label"])
    # Predicts labels for the test split using the model trained
    predicted_labels = model.predict(dataset["test"]["text"])
    # Calculates the accuracy score of the model and returns it
    return accuracy_score(dataset["test"]["label"], predicted_labels)