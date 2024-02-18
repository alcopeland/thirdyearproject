def load_dataset_from_csv(size):
    # https://medium.com/@lokaregns/fine-tuning-transformers-with-custom-dataset-classification-task-f261579ae068 
    from datasets import load_dataset
    dataset = load_dataset('csv', data_files = 'final_text.csv')
    split_dataset = dataset['train'].train_test_split(test_size=size)
    return dataset, split_dataset

def train_naive_bayes(dataset):
    # https://www.turing.com/kb/document-classification-using-naive-bayes
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.naive_bayes import MultinomialNB
    from sklearn.pipeline import make_pipeline    
    model = make_pipeline(TfidfVectorizer(), MultinomialNB())
    model.fit(dataset["train"]["text"],dataset["train"]["label"])
    return model

def test_naive_bayes(dataset):
    # https://www.turing.com/kb/document-classification-using-naive-bayes
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.naive_bayes import MultinomialNB
    from sklearn.pipeline import make_pipeline    
    from sklearn.metrics import accuracy_score
    model = make_pipeline(TfidfVectorizer(), MultinomialNB())
    model.fit(dataset["train"]["text"],dataset["train"]["label"])
    predicted_labels = model.predict(dataset["test"]["text"])
    return accuracy_score(dataset["test"]["label"], predicted_labels)