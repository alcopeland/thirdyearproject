from setup import clean_text
import numpy as np
from datasets import DatasetDict
from sklearn.pipeline import Pipeline
from text_generation import summarize_summary
from output_results import print_results
from user_intent import yes_or_no
from text_generation import get_response

from bert_score import BERTScorer
import concurrent.futures

def query_book_request(model: Pipeline, dataset: DatasetDict, user_query: str) -> None:
    # Takes a search query and predicts the best fitting genre using the Naive bayes model trained earlier
    prediction = model.predict([clean_text(user_query)])[0]
    # Filters the dataset by that genre
    refined_dataset = dataset.filter(lambda record: record["label"]==prediction)
    # Calculates BERTScore of the query to each summary in dataset 1 and returns the indexes of the top 10 scores
    top_10_index = calculate_bert_scores(user_query, refined_dataset)
    # Takes the top 10 and slits it in half
    top_5_index = top_10_index[:5]
    # Gets the names of the books that match the top 5 indexes 
    top_5_names = list(np.array(refined_dataset["train"]["name"])[top_5_index])
    # Summarises the book descriptions in the dataset for the selected books
    top_5_summaries_short = summarize_summary(list(np.array(refined_dataset["train"]["original"])[top_5_index]))
    # Zips the results together and prints them out
    results = tuple(zip(top_5_names, top_5_summaries_short))
    print_results(results,'0')

    # Asks the user whether they wouild like to see more results
    print(get_response("Would you like to see more results for this search?"))
    # If yes, repeat the process for the next 5 results
    if yes_or_no():
        bottom_5_index = top_10_index[5:]
        bottom_5_names = list(np.array(refined_dataset["train"]["name"])[bottom_5_index])
        bottom_5_summaries_short = summarize_summary(list(np.array(refined_dataset["train"]["original"])[bottom_5_index]))
        results = tuple(zip(bottom_5_names, bottom_5_summaries_short))
        print_results(results,'0')

def calculate_bert_score(summary: str, scorer: BERTScorer, query: str) -> float:
    P, R, F1  = scorer.score([query], [summary])
    return F1.detach().numpy()[0]

def calculate_bert_scores(query: str, dataset: DatasetDict) -> np.ndarray:
    # Creates a BERTScorer object using the model 'distilbert-base-uncased'
    scorer = BERTScorer(model_type='distilbert-base-uncased')
    scores = []
    # import time
    # start = time.time()
    # Uses concurrency with 10 workers to calculate BERTScores between query and descriptions of books simultaneously
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        future_scores = [executor.submit(calculate_bert_score, summary, scorer, query) for summary in dataset['train']['text']]
        # Wait for all results before continuing
        for score in future_scores:
            scores.append(score.result())
    # print(f'Calculating BERT Scores took {time.time()-start} seconds')
    # Sort by BERTScore to find the most similar results to the query and take the top 10 results
    top_10_index = np.argsort(scores)[-10:]
    return top_10_index