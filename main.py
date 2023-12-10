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

def trainNaiveBayes(dataset):
    print("training classifier...")
    from sklearn.naive_bayes import GaussianNB
    from sklearn.metrics import accuracy_score
    gnb = GaussianNB()
    # Train the classifier:
    import numpy as np
    model = gnb.fit(np.array(dataset["train"]["text"]).reshape(-1, 1), np.array(dataset["train"]["label"]))
    # Make predictions with the classifier:
    predictive_labels = gnb.predict(dataset["test"]["text"])
    print(predictive_labels)
    # Evaluate label (subsets) accuracy:
    print(accuracy_score(dataset["test"]["label"], predictive_labels))

def main():
    print("System starting...")
    import sys
    if len(sys.argv)>1:
        names, labels, texts = read_datasets()
        write_to_csv(names, labels, texts)
    dataset = load_dataset_from_csv()
    print(dataset)
    trainNaiveBayes(dataset)

if __name__ == "__main__":
    main()