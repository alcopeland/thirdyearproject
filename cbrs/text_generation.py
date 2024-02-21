from transformers.pipelines import Pipeline

import transformers
from transformers import pipeline

import concurrent.futures

import random
from transformers import PegasusForConditionalGeneration, PegasusTokenizer

def make_main_summary(input_summary: str, summarizer:Pipeline) -> str:
    # Previous methods attempted to make the summary
    # https://www.turing.com/kb/5-powerful-text-summarization-techniques-in-python 
    # from transformers import AutoTokenizer, AutoModelWithLMHead
    # from transformers import T5ForConditionalGeneration, T5Tokenizer

    # Method found at link below
    # https://thepythoncode.com/article/text-summarization-using-huggingface-transformers-python
    # Takes the first 1028 chracters of the summary and summarises it using the Pipeline provided
    # Length of the new summary will be between 65 and 100 tokens
    text = summarizer(input_summary[:1024], max_length=100, min_length=65, do_sample=False)[0]['summary_text']
    # Return the new summary
    return text
    
def summarize_summary(input_summaries: list) -> list:
    output_summaries = []
    # Hides warning messages from transformers
    transformers.logging.set_verbosity_error()
    # Previous summarisation models used
    # model_id = 'Falconsai/text_summarization'
    # model_id = 'pszemraj/led-large-book-summary'
    # model_id = 'pszemraj/long-t5-tglobal-base-16384-book-summary'
    
    # Model I finally settled on for summarisation
    model_id = 'facebook/bart-large-cnn'
    # Creates a summarisation pipeline using the model selected
    summarizer = pipeline('summarization', model=model_id)
    
    # import time
    # start = time.time()
    # Creates a thread pool with 5 workers to run summarisation in parallel 
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        future_scores = [executor.submit(make_main_summary, summary, summarizer) for summary in input_summaries]
        # Wait for all to finish before continuing
        for score in future_scores:
            output_summaries.append(score.result())
    # print(f'Making summaries took {time.time()-start} seconds')
    return output_summaries

def get_response(context: str) -> str:
    # Takes a string as input and will paraphrase the text
    # Method found at links below
    # https://thepythoncode.com/article/paraphrase-text-using-transformers-in-python
    # https://huggingface.co/blog/how-to-generate
    # https://huggingface.co/tuner007/pegasus_paraphrase
    # Hides warnings from transformers
    transformers.logging.set_verbosity_error()
    # Model selected for paraphrasing
    model_name = 'tuner007/pegasus_paraphrase'
    # Create a pretrained Pegasus tokeniser and model
    tokenizer = PegasusTokenizer.from_pretrained(model_name)
    model = PegasusForConditionalGeneration.from_pretrained(model_name)
    # Set variables for generation
    num_beams, num_return_sequences = 10, 5
    # Tokenise the desired sentence context
    batch = tokenizer([context], truncation=True, padding='longest', max_length=20, return_tensors="pt")
    # Generate 5 different paraphrases of the string and decode it using the tokeniser
    translated = model.generate(**batch, max_length=20, num_beams=num_beams, num_return_sequences=num_return_sequences, temperature=2.25, do_sample=True, early_stopping=True, top_p=0.92, top_k=15)
    responses = tokenizer.batch_decode(translated, skip_special_tokens=True)
    # Select a random number and choose a response to make it more random
    rand = random.randint(0, num_return_sequences-1)
    return responses[rand]