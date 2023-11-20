import torch
class BooksDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

def read_booksummaries(texts, labels):
    print("reading booksummaries")
    import re
    with open("datasets/uncompressed/booksummaries.txt", "r", encoding="utf-8") as file:
        sep1 = re.compile(r'"/m/\w{4,7}":')
        sep2 = re.compile(r'"/m/\w{4,7}"')
        sep3 = re.compile(r'/m/\w{4,7}')
        for line in file.readlines():
            newLine = re.sub(sep1,'',line)
            newLine = re.sub(sep2,'',newLine)
            newLine = re.sub(sep3,'',newLine)
            split = newLine.split("{")
            noNumber = split[0].split("\t\t")
            if(len(split)>1):
                genreSplit = split[1].split("}")
                text = ' '.join((noNumber[1] + genreSplit[1]).split())
                texts.append(text)
                label = re.sub('"','',genreSplit[0])
                label = label.split(",  ")
                label[0] = label[0][1:]
                labels.append(label)

def read_booksdataset(texts, labels):
    print("reading booksdataset")
    import csv
    with open("datasets/uncompressed/BooksDataSet.csv", "r", encoding="utf-8", errors="ignore") as file, open("datasets/uncompressed/total_text.txt", "a+", encoding="utf-8") as file2, open("datasets/uncompressed/cleanbooksdataset.txt", "w", encoding="utf-8") as cleanFile:
        csv_reader = csv.reader(file)
        next(file)
        for row in csv_reader:
            text = f"{row[2]} {row[4]}"
            texts.append(text)
            labels.append(row[3])

def read_datasets():
    texts = []
    labels = []
    print("reading datasets")
    # read_booksummaries(texts, labels)
    read_booksdataset(texts, labels)
    return texts, labels  

def train_bot(texts, labels):
    print("training model")    
    ratio = int(len(texts)*0.8)
    training_texts = texts[:ratio]
    testing_texts = texts[ratio:]
    training_labels = labels[:ratio]
    testing_labels = labels[ratio:]
    
    from sklearn.model_selection import train_test_split
    training_texts, validation_texts, training_labels, validation_labels = train_test_split(training_texts, training_labels, test_size=.2)

    import torch
    from transformers import GPT2LMHeadModel, GPT2Tokenizer, Trainer, TrainingArguments
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2-medium')
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id

    print("tokenizing...")
    training_encodings = tokenizer(training_texts, truncation=True, padding=True)
    validation_encodings = tokenizer(validation_texts, truncation=True, padding=True)
    testing_encodings = tokenizer(testing_texts, truncation=True, padding=True)

    print("datasets...")
    training_dataset = BooksDataset(training_encodings, training_labels)
    validation_dataset = BooksDataset(validation_encodings, validation_labels)
    testing_dataset = BooksDataset(testing_encodings, testing_labels)

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
        train_dataset=training_dataset,         # training dataset
        eval_dataset=validation_dataset             # evaluation dataset
    )
    print("training...")
    trainer.train()
    print("trained")

    trainer.save_model("models/")
    tokenizer.save_pretrained("models/")

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

def generate_default_response(query):
    import torch
    from transformers import GPT2LMHeadModel, GPT2Tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2-medium')
    print(f"{tokenizer.pad_token},{tokenizer.pad_token_id}")
    model = GPT2LMHeadModel.from_pretrained('gpt2-medium', pad_token_id = tokenizer.eos_token_id)

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
            train_bot(texts, labels)
        # while True:
        #     query = input()
        #     if query in exit_list: break
        #     generate_finetuned_response(query)
        # print("Exiting...")

    elif arg1=="0":
        print("System ready. Please enter your query: ")
        while True:
            query = input()
            if query in exit_list: break
            generate_default_response(query)
        print("Exiting...")
            

if __name__ == "__main__":
    import sys
    if(len(sys.argv)==3):
        if sys.argv[1]=="1": main("1",sys.argv[2])
        else: print("test"), main("0")
    else: main("0")
