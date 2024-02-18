def clean_text(text):
    from nltk.tokenize import word_tokenize
    tokenizedText = word_tokenize(text)
    from nltk.stem import PorterStemmer
    ps = PorterStemmer()
    from nltk.corpus import stopwords
    stop_words = set(stopwords.words('english'))
    cleanText = []
    for word in tokenizedText:
        if word not in stop_words:
            cleanText.append(ps.stem(word))
    cleanTextS = ' '.join(cleanText)
    import string
    cleanTextS = cleanTextS.translate(str.maketrans('', '', string.punctuation))
    return cleanTextS

def write_to_csv(names, labels, texts, original_texts):
    combinedArray = list(zip(names,labels,texts,original_texts))
    import csv
    with open('final_text.csv', 'w+', encoding="utf-8", errors="ignore", newline='') as file:
        writer = csv.writer(file)        
        writer.writerow(["name","label","text","original"])
        writer.writerows(combinedArray)    

def read_booksdataset():
    names, labels, texts, original_texts = [], [], [], []
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

def system_setup(run):
    from pathlib import Path
    file = Path("final_text.csv")
    if file.is_file() == False or run == 1:
        names, labels, texts, original_texts = read_booksdataset()
        write_to_csv(names, labels, texts, original_texts)
    from train_classifier import load_dataset_from_csv, train_naive_bayes
    full_dataset, split_dataset = load_dataset_from_csv(size=0.15)
    model = train_naive_bayes(split_dataset)
    return model, full_dataset