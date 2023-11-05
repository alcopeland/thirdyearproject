import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

tokenizer = GPT2Tokenizer.from_pretrained('gpt2-large')
model = GPT2LMHeadModel.from_pretrained('gpt2-large', pad_token_id = tokenizer.eos_token_id)

sequence = "What is The Wheel of Time?"

inputs = tokenizer.encode(sequence, return_tensors='pt')
outputs = model.generate(inputs,
    max_length=200,
    num_beams = 5,
    no_repeat_ngram_size = 2,
    early_stopping = True) #do_sample=True,temperature=1,top_k=50,

text = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(text)
