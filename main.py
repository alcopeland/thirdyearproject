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

def read_booksummaries(names, labels, texts):
    print("reading booksummaries...")
    import re
    with open("datasets/uncompressed/booksummaries.txt", "r", encoding="utf-8") as file:
        sep1 = re.compile(r'"/m/\w{4,7}":')
        sep2 = re.compile(r'"/m/\w{4,7}"')
        sep3 = re.compile(r'/m/\w{4,7}')
        sep4 = re.compile(r'=====')
        for line in file.readlines():
            newLine = re.sub(sep1,'',line)
            newLine = re.sub(sep2,'',newLine)
            newLine = re.sub(sep3,'',newLine)
            newLine = re.sub(sep4,'',newLine)
            split = newLine.split("{")
            noNumber = split[0].split("\t\t")
            nameSplit = noNumber[1].split("\t")
            
            if(len(split)>1):
                genreSplit = split[1].split("}")
                text = ' '.join((noNumber[1] + genreSplit[1]).split())
                cleanText = clean_text(text)
                label = re.sub('"','',genreSplit[0])
                label = label.split(",  ")
                label[0] = label[0][1:]

                names.append(nameSplit[0])
                labels.append(label)
                texts.append(cleanText)
                #print(f"{nameSplit[0]} / {label} / {cleanText}")

def read_booksdataset(names, labels, texts):
    print("reading booksdataset...")
    import csv
    with open("BooksDataSet.csv", "r", encoding="utf-8", errors="ignore") as file:
        csv_reader = csv.reader(file)
        next(file)
        for row in csv_reader:
            names.append(row[2])
            cleanText = clean_text(row[4])
            texts.append(f"{row[2]} {row[3]} {cleanText}")
            labels.append(row[3]) # put square brackets around to make into list for multiclass classification
    return names, labels, texts

def read_datasets():
    print("reading datasets...")
    names, labels, texts = read_booksdataset([],[],[])
    #read_booksummaries(names, labels, texts)
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
    # https://medium.com/@lokaregns/fine-tuning-transformers-with-custom-dataset-classification-task-f261579ae068 
    from datasets import load_dataset
    dataset = load_dataset('csv', data_files = 'final_text.csv')
    #dataset = dataset.class_encode_column("label")
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
    print(accuracy_score(dataset["test"]["label"], predicted_labels))
    return model

def predict_user_input(model):
    print("Please enter a sentence: ")
    query = input()
    prediction = model.predict([query])
    return query, prediction[0]

def limit_dataset(prediction, dataset):
    print("limiting dataset...")
    print(prediction)
    refined_dataset = dataset.filter(lambda record: record["label"]==prediction)
    #print(refined_dataset["train"]["label"])
    print(refined_dataset["train"]["name"])
    #print(len(refined_dataset["train"]["name"]))
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
    top_10_names.reverse()
    print("Here are some similar books you might like: ")
    for name in top_10_names:
        print(f"- {name}")

if __name__ == "__main__":
    main()