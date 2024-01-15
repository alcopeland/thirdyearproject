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
                #TODO append original text here
                #print(f"{nameSplit[0]} / {label} / {cleanText}")

def read_booksdataset(names, labels, texts, original_texts):
    import csv
    with open("BooksDataSet.csv", "r", encoding="utf-8", errors="ignore") as file:
        csv_reader = csv.reader(file)
        next(file)
        for row in csv_reader:
            names.append(row[2])
            cleanText = clean_text(row[4])
            labels.append(row[3]) # put square brackets around to make into list for multiclass classification
            texts.append(f"{row[2]} {row[3]} {cleanText}")
            original_texts.append(row[4])
    return names, labels, texts, original_texts

def read_datasets():
    names, labels, texts, original_texts = read_booksdataset([],[],[],[])
    #read_booksummaries(names, labels, texts)
    return names, labels, texts  , original_texts

def write_to_csv(names, labels, texts, original_texts):
    combinedArray = list(zip(names,labels,texts,original_texts))
    import csv
    with open('final_text.csv', 'w+', encoding="utf-8", errors="ignore", newline='') as file:
        writer = csv.writer(file)        
        writer.writerow(["name","label","text","original"])
        writer.writerows(combinedArray)    

def load_dataset_from_csv():
    # https://medium.com/@lokaregns/fine-tuning-transformers-with-custom-dataset-classification-task-f261579ae068 
    from datasets import load_dataset
    dataset = load_dataset('csv', data_files = 'final_text.csv')
    dataset = dataset['train'].train_test_split(test_size=0.2)
    return dataset

def train_naive_bayes(dataset):
    # https://www.turing.com/kb/document-classification-using-naive-bayes
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.naive_bayes import MultinomialNB
    from sklearn.pipeline import make_pipeline    
    from sklearn.metrics import accuracy_score
    
    model = make_pipeline(TfidfVectorizer(), MultinomialNB())
    model.fit(dataset["train"]["text"],dataset["train"]["label"])
    predicted_labels = model.predict(dataset["test"]["text"])
    #print(accuracy_score(dataset["test"]["label"], predicted_labels))
    return model

def predict_user_input(model):
    print("Please enter a sentence: ")
    query = input()
    prediction = model.predict([query])
    # print(prediction)
    return query, prediction[0]

def limit_dataset(prediction, dataset):
    refined_dataset = dataset.filter(lambda record: record["label"]==prediction)
    return refined_dataset

def calculate_bert_score(summary, scorer, query):
    P, R, F1  = scorer.score([query], [summary])
    return F1.detach().numpy()[0]

def calculate_bert_scores(query, dataset):
    from bert_score import BERTScorer
    scorer = BERTScorer(model_type='distilbert-base-uncased')
    scores = []

    # import time
    # start = time.time()
    import concurrent.futures
    with concurrent.futures.ThreadPoolExecutor(max_workers=60) as executor:
        future_scores = [executor.submit(calculate_bert_score, summary, scorer, query) for summary in dataset['train']['text']]
        for score in future_scores:
            scores.append(score.result())
    # print(f'Took {time.time()-start} seconds')
    import numpy as np
    top_10_index = np.argsort(scores)[-10:]
    return top_10_index, scores

def make_summary(input_summary, summarizer):
    # https://www.turing.com/kb/5-powerful-text-summarization-techniques-in-python 
    # from transformers import AutoTokenizer, AutoModelWithLMHead
    # from transformers import T5ForConditionalGeneration, T5Tokenizer

    # https://thepythoncode.com/article/text-summarization-using-huggingface-transformers-python
    text = summarizer(input_summary[:1024], max_length=115, min_length=85, do_sample=False)[0]['summary_text']
    return text
    
def summarize_summary(input_summaries):
    input_summaries.reverse()
    output_summaries = []
    
    from transformers import pipeline
    # model_id = 'Falconsai/text_summarization'
    # model_id = 'pszemraj/led-large-book-summary'
    # model_id = 'pszemraj/long-t5-tglobal-base-16384-book-summary'
    model_id = 'facebook/bart-large-cnn'
    summarizer = pipeline('summarization', model=model_id)
    
    # import time
    # start = time.time()
    import concurrent.futures
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
        future_scores = [executor.submit(make_summary, summary, summarizer) for summary in input_summaries]
        for score in future_scores:
            output_summaries.append(score.result())
    # print(f'Took {time.time()-start} seconds')
    return output_summaries

def main():
    print("System starting...")
    import sys
    if len(sys.argv)>1:
        names, labels, texts, original_texts = read_datasets()
        write_to_csv(names, labels, texts, original_texts)

    dataset = load_dataset_from_csv()
    model = train_naive_bayes(dataset)
    query, prediction = predict_user_input(model)
    refined_dataset = limit_dataset(prediction, dataset)

    top_10_index, all_scores = calculate_bert_scores(query, refined_dataset)
    import numpy as np
    top_10_names = list(np.array(refined_dataset["train"]["name"])[top_10_index])
    top_10_names.reverse()

    top_10_summaries_short = summarize_summary(list(np.array(refined_dataset["train"]["original"])[top_10_index]))

    results = tuple(zip(top_10_names, top_10_summaries_short))
    print("\n\nHere are some similar books you might like: ")
    for result in results:
        print(f" -- {result[0]} -- ")
        print(f"{result[1]}")
        print("\n")

if __name__ == "__main__":
    main()