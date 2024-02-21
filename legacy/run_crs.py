# An old appraoch to my project which I scrapped as I wanted to do it differently


import torch
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

def read_booksummaries(texts, labels):
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
            if(len(split)>1):
                genreSplit = split[1].split("}")
                text = ' '.join((noNumber[1] + genreSplit[1]).split())
                cleanText = clean_text(text)
                texts.append(cleanText)
                label = re.sub('"','',genreSplit[0])
                label = label.split(",  ")
                label[0] = label[0][1:]
                labels.append(label)

def read_booksdataset(texts, labels):
    print("reading booksdataset...")
    import csv
    with open("datasets/uncompressed/BooksDataSet.csv", "r", encoding="utf-8", errors="ignore") as file:
        csv_reader = csv.reader(file)
        next(file)
        for row in csv_reader:
            text = f"{row[2]} {row[4]}"
            cleanText = clean_text(text)
            texts.append(cleanText)
            labels.append([row[3]])

def read_datasets():
    texts = []
    labels = []
    print("reading datasets...")
    read_booksummaries(texts, labels)
    read_booksdataset(texts, labels)
    return texts, labels  

def write_to_csv(texts, labels):
    print("writing to csv...")
    combinedArray = list(zip(texts,labels))
    import csv
    with open('final_text.csv', 'w+', encoding="utf-8", errors="ignore", newline='') as file:
        writer = csv.writer(file)        
        writer.writerow(["text","label"])
        writer.writerows(combinedArray)

def train_bot():
    print("training model")    
    from datasets import load_dataset
    dataset = load_dataset('csv', data_files = 'final_text.csv')

    from sklearn.model_selection import train_test_split
    splitDataset = dataset['train'].train_test_split(test_size=0.2)
    splitDataset = splitDataset.map(tokenize, batched=True, batch_size=None)

    print(splitDataset["train"]["text"][0])
    print(type(splitDataset["train"]["text"][0]))
    print(splitDataset["train"]["label"][0])
    print(type(splitDataset["train"]["label"][0]))
    # print(splitDataset["train"]["label"][0][0])
    # print(type(splitDataset["train"]["label"][0][0]))

    import torch
    from transformers import TrainingArguments, Trainer, GPT2LMHeadModel, GPT2Tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2-medium')
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id

    training_args = TrainingArguments(
        output_dir="./models",
        num_train_epochs=3,              # total number of training epochs
        per_device_train_batch_size=16,  # batch size per device during training
        per_device_eval_batch_size=64,   # batch size for evaluation
        warmup_steps=500,                # number of warmup steps for learning rate scheduler
        weight_decay=0.01,               # strength of weight decay
        logging_dir='./logs',            # directory for storing logs
        logging_steps=10,
    )
    model = GPT2LMHeadModel.from_pretrained('gpt2-medium', pad_token_id = tokenizer.eos_token_id)
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=splitDataset["train"],         # training dataset
        eval_dataset=splitDataset["test"]            # evaluation dataset
    )
    print("training...")
    trainer.train()
    print("trained")

    trainer.save_model("models/")
    tokenizer.save_pretrained("models/")

def tokenize(batch):
        import torch
        from transformers import GPT2Tokenizer
        tokenizer = GPT2Tokenizer.from_pretrained('gpt2-medium')
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
        return tokenizer(batch["text"], padding=True, truncation=True)
    
def convert_labels(labels, listLabels):
    indexList = []
    for row in labels:
        # rowList = []
        # for word in row:
        #     rowList.append(listLabels.index(word))
        # indexList.append(rowList)
        indexList.append(listLabels.index(row[0]))
    return indexList

def unique_labels_list(labels):
    uniqueLabels = set(item for list in labels for item in list)
    listLabels = list(uniqueLabels)
    return listLabels

def generate_finetuned_response(query):
    import torch
    from transformers import GPT2LMHeadModel, GPT2Tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained('models/')
    model = GPT2LMHeadModel.from_pretrained('models/', pad_token_id = tokenizer.eos_token_id)

    inputs = tokenizer.encode(query, return_tensors='pt')
    outputs = model.generate(inputs,
        max_length=200,
        num_beams = 5,
        no_repeat_ngram_size = 2,
        early_stopping = True,
        do_sample=True,
        temperature = 0.25,
        top_k=20)
    
    text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(text)

# def calc_bert_scores(query, texts):
#     from bert_score import BERTScorer
#     scorer = BERTScorer(model_type='distilbert-base-uncased')
#     results = []
#     for line in texts:
#         P, R, F1 = scorer.score([query], [line])
#         results.append(F1.detach().numpy()[0])
#     print(f"The best result is: {texts[results.index(max(results))]}")

def calc_bert_scores(query, texts):
    print("bert scores")
    import numpy
    queries = numpy.repeat(query,len(texts))
    from bert_score import BERTScorer
    scorer = BERTScorer(model_type='distilbert-base-uncased')
    results = scorer.score(queries, texts)
    results2 = [tup[1].detach().numpy()[0] for tup in results]

    # results = list(map(calc_bert_score, texts, queries.tolist()))
    print(results2[0:4])

def calc_bert_score(texts, queries):
    from bert_score import BERTScorer
    scorer = BERTScorer(model_type='distilbert-base-uncased')
    P, R, F1 = scorer.score(queries, texts)
    return 

def generate_default_response(query, model, tokenizer):
    inputs = tokenizer.encode(query, return_tensors='pt')
    outputs = model.generate(inputs,
        max_length=200,
        num_beams = 5,
        no_repeat_ngram_size = 2,
        early_stopping = True,
        do_sample=True,
        temperature = 0.2,
        top_k=20)
    
    text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(text)

def main(arg1,arg2=None):
    print("System starting...")
    exit_list = ["q","quit","q()","quit()","exit","exit()"]
    if arg1=="1":
        if arg2=="1":
            texts, labels = read_datasets()
            #uniqueLabels = unique_labels_list(labels)
            #indexLabels = convert_labels(labels, uniqueLabels)
            #write_to_csv(texts,indexLabels)
            query = input()
            top10 = calc_bert_scores(query,texts[0:100])

            #train_bot()
        # while True:
        #     query = input()
        #     if query in exit_list: break
        #     generate_finetuned_response(query)
        # print("Exiting...")
    elif arg1=="0":
        import torch
        from transformers import GPT2LMHeadModel, GPT2Tokenizer
        tokenizer = GPT2Tokenizer.from_pretrained('gpt2-medium')
        model = GPT2LMHeadModel.from_pretrained('gpt2-medium', pad_token_id = tokenizer.eos_token_id)
        convHistory = ""
        
        print("System ready. Please enter your query: ")
        while True:
            query = input()
            if query in exit_list: break
            generate_default_response(convHistory+" "+query, model, tokenizer)
            convHistory = convHistory +" "+ clean_text(query)
        print("Exiting...")
            

if __name__ == "__main__":
    import sys
    if(len(sys.argv)==3):
        if sys.argv[1]=="1": main("1",sys.argv[2])
        else: print("test"), main("0")
    else: main("0")
