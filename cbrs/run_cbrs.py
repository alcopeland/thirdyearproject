# This is the main function for my conversational recommeder system
# It will keep asking the user if it can help until closed or the user says they want to stop
# Dataset 1 = final_text.csv
# Dataset 2 = goodreads_data.csv

# Will take a user input and attempt to determine whether they are asking about an author, genre book or general request
# Then it will extract the important words depending on the intent and search the dataset 2 for books
# Then it will offer to find similar books using BERTScore in dataset 1

import sys
from setup import system_setup
from user_intent import get_initial_query
from extract_keywords import extract_author_keywords, extract_book_keywords, extract_genre_keywords
from search_database import find_authors_in_dataset, find_books_in_dataset, find_genres_in_dataset
from output_results import output_results
from text_generation import get_response
from extract_keywords import extract_summary_keywords
from find_similar_books import query_book_request
from user_intent import yes_or_no

def main() -> None:
    print("System starting...")
    # Checks if the debug tag is in the command line and rusn the setup method with a correct identifier
    if len(sys.argv)>1 and sys.argv[1]=="-d": model, dataset = system_setup(1)
    else: model, dataset = system_setup(0)
    # Sets varibales to record state of the system i.e. whether it is the first loop
    loop, first = True, True
    # Creates an empty string to store the previous queries in after each search
    full_query = ""
    # while True, offer assistance
    while loop:
        # Get user input and determine intent
        initial_query, intent = get_initial_query(first)
        # Match the intent of the user to the correct action to perform
        match intent:
            # If intent is to exit, break the loop
            case 'exit': loop = False
            # If intent is general, ask questions to get more information then perform a search
            case 'general_request':
                print(get_response("I can help find the right book for you."))
                # Ask about an author and get user input
                print(get_response("What is an author you particularly enjoyed reading?"))
                response = input()
                # Extract the potential names of authors in the user input
                authors = extract_author_keywords(response)
                # If there is a potential name, find books in the dataset 2
                if authors: author_results = find_authors_in_dataset(authors)
                else: print(get_response("I failed to determine what author you wanted to find."))

                # Ask about what genre's they like to read and get user input
                print(get_response("What is your favourite genre to read?"))
                response = input()
                # Extract potential genres from the user input
                genres = extract_genre_keywords(response)
                # If there is a potenital genre, search dataset 2 to find books
                if genres: genre_results = find_genres_in_dataset(genres)
                else: print(get_response("I failed to determine what genre you wanted to find."))

                # Ask about previous books they liked and get user input
                print(get_response("What book or series do you consider to be your all time favourite?"))
                response = input()
                # Extract book names from the user input
                books = extract_book_keywords(response)
                # If there is a potential book name, search dataset 2 and return books
                if books: book_results = find_books_in_dataset(books)
                else: print(get_response("I failed to determine what book/series you wanted to find."))
                
                results = author_results + genre_results + book_results
                # Take the summaries of each book found
                if results:
                    summaries = [a[2] for a in results]
                    # Create the new search query by taking book name, genres and keywords from the sumamries for each book found previously into a single string
                    new_query = ""
                    for result in results:
                        new_query += (result[0] + " ")
                        new_query += (((str(result[3]).replace("[","")).replace("]","")).replace("'","")+" ")
                    new_query += extract_summary_keywords(summaries)
                    full_query = full_query + " " + new_query
                    # Search for books in dataset 1 using BERTScore similarity
                    # Either using the new query or the full query if this is not the first time
                    if not first:
                        print(get_response("Would you like to use your previous searches in this search?"))
                        if yes_or_no(): query_book_request(model, dataset, full_query)
                        else: query_book_request(model, dataset, new_query)
                    else: query_book_request(model, dataset, new_query)
            # If intent is an author
            case 'author_request': 
                # Extract potential author names from the input
                authors = extract_author_keywords(initial_query)
                # If there are potential authors, find matching books from dataset 2
                if authors:
                    results = find_authors_in_dataset(authors)
                    # If there are results for that author, output them to the user
                    if results: full_query = output_results(results, first, model, dataset, full_query)
                    else: print(get_response("I was unable to find any books by that author."))
                else: print(get_response("I failed to determine what author you wanted to find."))
            # If intent is a book series
            case 'book_request':
                # Extract potential book names from the input
                book_keywords = extract_book_keywords(initial_query)
                # If there are potential books, find matching books from dataset 2
                if book_keywords: 
                    results = find_books_in_dataset(book_keywords)
                    # If there are results for that book, output them to the user
                    if results: full_query = output_results(results, first, model, dataset, full_query)
                    else: print(get_response("I was unable find any books by that name."))
                else: print(get_response("I failed to determine what book/series you wanted to find."))
            # If intent is a genre
            case 'genre_request':
                # Extract potential genres from the input
                genre_keywords = extract_genre_keywords(initial_query)
                # If there are potential genres, find matching books from dataset 2
                if genre_keywords:
                    results = find_genres_in_dataset(genre_keywords)
                    # If there are results for that genre, output them to the user
                    if results: full_query = output_results(results, first, model, dataset, full_query)
                    else: print(get_response("I was unable to find any books in that genre."))
                else: print(get_response("I failed to determine what genre you wanted to find."))
        # After first run, set first to False
        first = False

if __name__ == "__main__":
    main()