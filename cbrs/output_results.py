from datasets import DatasetDict
from sklearn.pipeline import Pipeline
from text_generation import summarize_summary, get_response
from extract_keywords import extract_summary_keywords
from user_intent import yes_or_no

from text_generation import get_response

def output_results(results: list, first: bool, model: Pipeline, dataset: DatasetDict, full_query: str) -> str:
    # Gets the summaries of the results provided
    summaries = [a[2] for a in results]
    # Use a model to write short summaries of first 5 texts
    new_summaries = summarize_summary(summaries[:5])
    # Re-group the results with the new summaries
    final_results = tuple(zip([a[0] for a in results], [a[1] for a in results], new_summaries+(['']*(len(summaries)-5)), [a[3] for a in results], [a[4] for a in results], [a[5] for a in results]))
    # Print the first 5 results with summaries, and the remainder without to save on cimputation time and page spam
    print_results(final_results[:5],'1')
    if len(final_results)>5:
        print_results(final_results[5:],'2')
    new_query = ""
    # Create a string containing book names, genres and keywords from all the results given
    for result in results:
        new_query += (result[0]+" ")
        new_query += (((str(result[3]).replace("[","")).replace("]","")).replace("'","")+" ")
    new_query += extract_summary_keywords(summaries)
    # Add the new query string to the culmination of all searches
    full_query = full_query + " " + new_query
    # Ask if the user would like to see some simialr results
    print(get_response("Would you be interested finding some similar results?"))
    # If yes, and not on the first run through the system, ask if you would like to use previous searches as well, i.e. the full query string calculated 
    # If yes, run a query search through dataset 1 using BERTScore to find similarity
    if yes_or_no():
        from find_similar_books import query_book_request
        if not first:
            print(get_response("Would you like to use your previous searches in this search?"))
            if yes_or_no(): query_book_request(model, dataset, full_query)
            else: query_book_request(model, dataset, new_query)
        else: query_book_request(model, dataset, new_query)
    # Return the conversation history
    return full_query

def print_results(results: list, intent: str) -> None:
    # Depending on the intent value provided, output the results in the desired format
    match intent:
        # Used for query request results as they do not come with an author
        # Outputs the name of the book and a summary of it
        case '0':
            print("\n\n"+ get_response("Here are some books I think you might enjoy") + "\n")
            for result in results:
                print(f" -- {result[0]} -- ")
                print(f"{result[1]}")
                print("\n")
        # Used for the first 5 results from a search in dataset 2
        # Outputs the name of the book, author, genres, rating and rating number and a summary of the book
        case '1':
            print("\n\n"+ get_response("Here are some books I think you might enjoy") + "\n")
            for result in results:
                genres = ((str(result[3]).replace("[","")).replace("]","")).replace("'","")
                print(f" -- {result[0]} by {result[1]} -- ")
                print(f"Average rating: {result[4]} from {result[5]} reviews")
                print(f"Genres: {genres}")
                print(f"{result[2]}")
                print("\n")
        # Used for the remainder of the results from a search in dataset 2
        # Outputs the name of the book, author, genres, rating and rating number without a summary of the book
        case '2':
            for result in results:
                genres = ((str(result[3]).replace("[","")).replace("]","")).replace("'","")
                print(f" -- {result[0]} by {result[1]} -- ")
                print(f"Average rating: {result[4]} from {result[5]} reviews")
                print(f"Genres: {genres}")
                print("\n")