def main():
    print("System starting...")
    from setup import system_setup
    model, dataset = system_setup()
    loop, first = True, True
    full_query = ""
    from user_intent import get_initial_query
    from extract_keywords import extract_author_keywords, extract_book_keywords, extract_genre_keywords
    from search_database import find_authors_in_dataset, find_books_in_dataset, find_genres_in_dataset
    from output_results import output_results
    while loop:
        initial_query, intent = get_initial_query(first)
        print(intent)
        match intent:
            case 'exit': loop = False

            case 'general_request':
                print("general")

            case 'author_request': 
                authors = extract_author_keywords(initial_query)
                if authors:
                    results = find_authors_in_dataset(authors)
                    if results: full_query = output_results(results, first, model, dataset, full_query)
                    else: print("Sorry, I was unable to find any books by that author.")
                else: print("Sorry, I failed to determine what author you wanted.")

            case 'book_request':
                book_keywords = extract_book_keywords(initial_query)
                if book_keywords:
                    results = find_books_in_dataset(book_keywords)
                    if results: full_query = output_results(results, first, model, dataset, full_query)
                    else: print("Sorry, I was unable find any books by that name.")
                else: print("Sorry, I failed to determine what book/series you wanted.")

            case 'genre_request':
                genre_keywords = extract_genre_keywords(initial_query)
                if genre_keywords:
                    results = find_genres_in_dataset(genre_keywords)
                    if results: full_query = output_results(results, first, model, dataset, full_query)
                    else: print("Sorry, I was unable to find any books in that genre.")
                else: print("Sorry, I failed to determine what genre you wanted.")

        first = False

if __name__ == "__main__":
    main()