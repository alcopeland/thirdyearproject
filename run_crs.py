def read_booksummaries():
    print("reading booksummaries")
    import re
    open('datasets/uncompressed/total_text.txt', 'w').close()
    open('datasets/uncompressed/cleanbooksummaries.txt', 'w').close()
    with open("datasets/uncompressed/booksummaries.txt", "r", encoding="utf-8") as file, open("datasets/uncompressed/total_text.txt", "a", encoding="utf-8") as newFile, open("datasets/uncompressed/cleanbooksummaries.txt", "a", encoding="utf-8") as cleanFile:
        sep1 = re.compile(r'"/m/\w{4,7}":')
        sep2 = re.compile(r'"/m/\w{4,7}"')
        sep3 = re.compile(r'/m/\w{4,7}')
        sep4 = re.compile(r'{')
        sep5 = re.compile(r'}')
        for line in file.readlines():
            newLine = re.sub(sep1,'',line)
            newLine = re.sub(sep2,'',newLine)
            newLine = re.sub(sep3,'',newLine)
            newLine = re.sub(sep4,'',newLine)
            newLine = re.sub(sep5,'',newLine)
            newLine = ' '.join(newLine.split())
            newFile.write(f"{newLine} ")
            cleanFile.write(f"{newLine} ")

def read_booksdataset():
    print("reading booksdataset")
    import csv
    with open("datasets/uncompressed/BooksDataSet.csv", "r", encoding="utf-8", errors="ignore") as file, open("datasets/uncompressed/total_text.txt", "a+", encoding="utf-8") as file2, open("datasets/uncompressed/cleanbooksdataset.txt", "w", encoding="utf-8") as cleanFile:
        csv_reader = csv.reader(file)
        next(file)
        for row in csv_reader:
            text = f"{row[2]} {row[3]} {row[4]}"
            file2.write(text)
            cleanFile.write(text)
    
def read_books1():
    print("reading all books")
    import os
    for filename in os.listdir("datasets/uncompressed/books1/epubtxt"):
        read_book("datasets/uncompressed/books1/epubtxt/"+filename)

def read_book(path):
    with open(path, "r", encoding = "utf-8") as file, open("datasets/uncompressed/total_text.txt", "a+", encoding="utf-8") as file2:
        book_text = file.read()
        book_text = ' '.join(book_text.split())
        file2.write(book_text)

def read_datasets():
    print("reading datasets")
    read_booksummaries()
    read_booksdataset()
    # read_books1()

    with open("datasets/uncompressed/total_text.txt", "r", encoding="utf-8") as file:
        total_text = file.read()
        training_ratio = int(0.8 * len(total_text))
        training_data = total_text[:training_ratio]
        validation_data = total_text[training_ratio:]
    with open("datasets/uncompressed/training_data.txt", "w", encoding="utf-8") as f:
        f.write(training_data)
    with open("datasets/uncompressed/validation_data.txt", "w", encoding="utf-8") as f:
        f.write(validation_data)    

def train_bot():
    print("training model")    
    import torch
    from transformers import GPT2LMHeadModel, GPT2Tokenizer, TextDataset, DataCollatorForLanguageModeling, Trainer, TrainingArguments
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2-large')
    model = GPT2LMHeadModel.from_pretrained('gpt2-large', pad_token_id = tokenizer.eos_token_id)
    
    training_dataset = TextDataset(tokenizer=tokenizer, file_path="datasets/uncompressed/training_data.txt", block_size=128)
    validation_dataset = TextDataset(tokenizer=tokenizer, file_path="datasets/uncompressed/validation_data.txt", block_size=128)
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    training_args = TrainingArguments(
        output_dir="models/",
        overwrite_output_dir=True,
        per_device_train_batch_size=64,
        per_device_eval_batch_size=64,
        num_train_epochs=100,
        save_steps=10_000,
        save_total_limit=2,
        logging_dir='./logs',
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=training_dataset,
        eval_dataset=validation_dataset,
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
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2-large')
    model = GPT2LMHeadModel.from_pretrained('gpt2-large', pad_token_id = tokenizer.eos_token_id)

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

def main(arg):
    print("System starting...")
    print(arg)
    exit_list = ["q","quit","q()","quit()","exit","exit()"]
    if arg == 1:
        print("test")
        read_datasets()
        # train_bot()
        # while True:
        #     query = input()
        #     if query in exit_list: break
        #     generate_finetuned_response(query)
        # print("Exiting...")

    elif arg == 0:
        print("System ready. Please enter your query: ")
        while True:
            query = input()
            if query in exit_list: break
            generate_default_response(query)
        print("Exiting...")
            

if __name__ == "__main__":
    import sys
    if(len(sys.argv)>1):
        print(sys.argv[1])
        if sys.argv[1] in {0,1}: print("test"), main(sys.argv[1])
        else: print("test2"), main(0)
    else: main(0)
