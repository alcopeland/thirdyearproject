def clean_text(text):
    from nltk.tokenize import word_tokenize
    tokenizedText = word_tokenize(text)

    from nltk.corpus import stopwords
    stop_words = set(stopwords.words('english'))
    cleanText = []
    for word in tokenizedText:
        if word not in stop_words:
            cleanText.append(word)
    cleanTextS = ' '.join(cleanText)
    return cleanTextS

def read_booksdataset(names, texts, labels):
    print("reading booksdataset...")
    import csv
    with open("BooksDataSet.csv", "r", encoding="utf-8", errors="ignore") as file:
        csv_reader = csv.reader(file)
        next(file)
        for row in csv_reader:
            names.append(row[2])
            cleanText = clean_text(f"{row[2]} {row[4]}")
            texts.append(cleanText)
            labels.append(row[3]) # put square brackets around to make into list for multiclass classification
    return names, labels, texts

def read_datasets():
    print("reading datasets...")
    names, labels, texts = read_booksdataset([],[],[])
    # read_booksummaries(names, labels, texts)
    return names, labels, texts  

def write_to_csv(names, labels, texts):
    print("writing to csv...")
    combinedArray = list(zip(names,labels,texts))
    import csv
    with open('final_text.csv', 'w+', encoding="utf-8", errors="ignore", newline='') as file:
        writer = csv.writer(file)        
        writer.writerow(["name","label","text"])
        writer.writerows(combinedArray)    

def load_dataset_from_csv():
    print("loading dataset...")
    from datasets import load_dataset
    dataset = load_dataset('csv', data_files = 'final_text.csv')
    dataset = dataset.class_encode_column("label")
    dataset = dataset['train'].train_test_split(test_size=0.2)
    return dataset

def train_naive_bayes(dataset):
    print("training classifier...")
    # https://www.turing.com/kb/document-classification-using-naive-bayes
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.naive_bayes import MultinomialNB
    from sklearn.pipeline import make_pipeline    
    from sklearn.metrics import accuracy_score
    
    model = make_pipeline(TfidfVectorizer(), MultinomialNB())
    model.fit(dataset["train"]["text"],dataset["train"]["label"])
    predicted_labels = model.predict(dataset["test"]["text"])

    #print(predicted_labels)
    #print(accuracy_score(dataset["test"]["label"], predicted_labels))
    return model

def predict_user_input(model):
    print("Please enter a sentence: ")
    query = input()
    prediction = model.predict([query])
    return query, prediction[0]

def limit_dataset(prediction, dataset):
    print("limiting dataset...")
    refined_dataset = dataset.filter(lambda record: record["label"]==prediction)
    # print(refined_dataset["train"]["name"])
    print(len(refined_dataset["train"]["name"]))
    return refined_dataset

def calculate_bert_score(query, summary):
    from bert_score import BERTScorer
    scorer = BERTScorer(model_type='distilbert-base-uncased')
    P, R, F1  = scorer.score([query], [summary])
    return F1.detach().numpy()[0]

def calculate_bert_scores(query, dataset):
    print("calculating bert scores...")
    import numpy as np
    queries = np.repeat(query,len(dataset["train"]["text"]))
    scores = map(calculate_bert_score, queries, dataset["train"]["text"])
    scores = list(scores)
    print(scores)
    top_10_index = np.argsort(scores)[-10:]
    print(top_10_index)
    return top_10_index, scores

def main():
    print("System starting...")
    import sys
    if len(sys.argv)>1:
        names, labels, texts = read_datasets()
        write_to_csv(names, labels, texts)
    dataset = load_dataset_from_csv()
    model = train_naive_bayes(dataset)
    query, prediction = predict_user_input(model)
    refined_dataset = limit_dataset(prediction, dataset)
    top_10_index, all_scores = calculate_bert_scores(query, refined_dataset)
    import numpy as np
    top_10_names = list(np.array(refined_dataset["train"]["name"])[top_10_index])
    print("Here are some similar books you might like: ")
    for name in top_10_names:
        print(f"- {name}")

if __name__ == "__main__":
    main()