def query_book_request(model, dataset, user_query):
    prediction = model.predict([user_query])[0]
    refined_dataset = dataset.filter(lambda record: record["label"]==prediction)
    import numpy as np
    top_10_index, all_scores = calculate_bert_scores(user_query, refined_dataset)
    top_5_index = top_10_index[:5]
    top_5_names = list(np.array(refined_dataset["train"]["name"])[top_5_index])
    from text_generation import summarize_summary
    top_5_summaries_short = summarize_summary(list(np.array(refined_dataset["train"]["original"])[top_5_index]))
    results = tuple(zip(top_5_names, top_5_summaries_short))
    from output_results import print_results
    print_results(results,'0')

    from user_intent import yes_or_no
    from text_generation import get_response
    print(get_response("Would you like to see more results for this search?"))
    if yes_or_no():
        bottom_5_index = top_10_index[5:]
        bottom_5_names = list(np.array(refined_dataset["train"]["name"])[bottom_5_index])
        bottom_5_summaries_short = summarize_summary(list(np.array(refined_dataset["train"]["original"])[bottom_5_index]))
        results = tuple(zip(bottom_5_names, bottom_5_summaries_short))
        print_results(results,'0')

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
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        future_scores = [executor.submit(calculate_bert_score, summary, scorer, query) for summary in dataset['train']['text']]
        for score in future_scores:
            scores.append(score.result())
    # print(f'Calculating BERT Scores took {time.time()-start} seconds')
    import numpy as np
    top_10_index = np.argsort(scores)[-10:]
    return top_10_index, scores