import transformers
from transformers import PegasusForConditionalGeneration, PegasusTokenizer
import random

# A small program I created to experiment with phrasing of my prompts for paraphrasing, a well as test variables in the generate command to produce the best results

def get_response(context: str) -> str:
    # Method found in links below
    # https://thepythoncode.com/article/paraphrase-text-using-transformers-in-python
    # https://huggingface.co/blog/how-to-generate
    # https://huggingface.co/tuner007/pegasus_paraphrase
    transformers.logging.set_verbosity_error()
    
    model_name = 'tuner007/pegasus_paraphrase'
    tokenizer = PegasusTokenizer.from_pretrained(model_name)
    model = PegasusForConditionalGeneration.from_pretrained(model_name)
    num_beams, num_return_sequences = 10, 5

    batch = tokenizer([context], truncation=True, padding='longest', max_length=15, return_tensors="pt")
    translated = model.generate(**batch, max_length=15, num_beams=num_beams, num_return_sequences=num_return_sequences, temperature=2.25, do_sample=True, early_stopping=True, top_p=0.92, top_k=15)
    response = tokenizer.batch_decode(translated, skip_special_tokens=True)
    print(response)
    
    rand = random.randint(0, 4)
    return response[rand]

def main(): print(get_response("What further assistance am I able to offer you with today?"))

if __name__ == "__main__": main()