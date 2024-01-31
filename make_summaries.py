def make_main_summary(input_summary, summarizer):
    # https://www.turing.com/kb/5-powerful-text-summarization-techniques-in-python 
    # from transformers import AutoTokenizer, AutoModelWithLMHead
    # from transformers import T5ForConditionalGeneration, T5Tokenizer

    # https://thepythoncode.com/article/text-summarization-using-huggingface-transformers-python
    text = summarizer(input_summary[:1024], max_length=100, min_length=65, do_sample=False)[0]['summary_text']
    return text
    
def summarize_summary(input_summaries):
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
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        future_scores = [executor.submit(make_main_summary, summary, summarizer) for summary in input_summaries]
        for score in future_scores:
            output_summaries.append(score.result())
    # print(f'Making summaries took {time.time()-start} seconds')
    return output_summaries