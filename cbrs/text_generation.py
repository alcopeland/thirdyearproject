def make_main_summary(input_summary, summarizer):
    # https://www.turing.com/kb/5-powerful-text-summarization-techniques-in-python 
    # from transformers import AutoTokenizer, AutoModelWithLMHead
    # from transformers import T5ForConditionalGeneration, T5Tokenizer

    # https://thepythoncode.com/article/text-summarization-using-huggingface-transformers-python
    text = summarizer(input_summary[:1024], max_length=100, min_length=65, do_sample=False)[0]['summary_text']
    return text
    
def summarize_summary(input_summaries):
    output_summaries = []
    import transformers
    transformers.logging.set_verbosity_error()
    from transformers import pipeline
    # model_id = 'Falconsai/text_summarization'
    # model_id = 'pszemraj/led-large-book-summary'
    # model_id = 'pszemraj/long-t5-tglobal-base-16384-book-summary'
    model_id = 'facebook/bart-large-cnn'
    summarizer = pipeline('summarization', model=model_id)
    
    # import time
    # start = time.time()
    import concurrent.futures
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        future_scores = [executor.submit(make_main_summary, summary, summarizer) for summary in input_summaries]
        for score in future_scores:
            output_summaries.append(score.result())
    # print(f'Making summaries took {time.time()-start} seconds')
    return output_summaries

def get_response(context):
    # https://thepythoncode.com/article/paraphrase-text-using-transformers-in-python
    # https://huggingface.co/blog/how-to-generate
    # https://huggingface.co/tuner007/pegasus_paraphrase
    import random
    import transformers
    transformers.logging.set_verbosity_error()
    from transformers import PegasusForConditionalGeneration, PegasusTokenizer
    model_name = 'tuner007/pegasus_paraphrase'
    tokenizer = PegasusTokenizer.from_pretrained(model_name)
    model = PegasusForConditionalGeneration.from_pretrained(model_name)
    num_beams, num_return_sequences = 10, 5

    batch = tokenizer([context], truncation=True, padding='longest', max_length=20, return_tensors="pt")
    translated = model.generate(**batch, max_length=20, num_beams=num_beams, num_return_sequences=num_return_sequences, temperature=2.25, do_sample=True, early_stopping=True, top_p=0.92, top_k=15)
    responses = tokenizer.batch_decode(translated, skip_special_tokens=True)
    rand = random.randint(0, 4)
    return responses[rand]