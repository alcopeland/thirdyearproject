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
    from text_generation import get_response
    while loop:
        # print(f"Full Query: {full_query}")
        initial_query, intent = get_initial_query(first)
        # print(intent)
        match intent:
            case 'exit': loop = False

            case 'general_request':
                print(get_response("I can help find the right book for you."))
                print(get_response("What is an author you particularly enjoyed reading?"))
                response = input()
                authors = extract_author_keywords(response)
                if authors: results = find_authors_in_dataset(authors)
                else: print(get_response("I failed to determine what author you wanted to find."))
                print(results)

                print(get_response("What is your favourite genre to read?"))
                response = input()
                genres = extract_genre_keywords(response)
                if genres: results.append(find_genres_in_dataset(genres))
                else: print(get_response("I failed to determine what genre you wanted to find."))
                print(results)

                print(get_response("What book or series do you consider to be your all time favourite?"))
                response = input()
                books = extract_book_keywords(response)
                if books: results.append(find_books_in_dataset(books))
                else: print(get_response("I failed to determine what book/series you wanted to find."))
                print(results)

                summaries = [a[2] for a in results]

                new_query = ""
                from extract_keywords import extract_summary_keywords
                for result in results:
                    new_query += (result[0]+" ")
                    new_query += (((str(result[3]).replace("[","")).replace("]","")).replace("'","")+" ")
                new_query += extract_summary_keywords(summaries)
                full_query = full_query + " " + new_query

                from find_similar_books import query_book_request
                from user_intent import yes_or_no
                if not first:
                    print(get_response("Would you like to use your previous searches in this search?"))
                    if yes_or_no(): query_book_request(model, dataset, full_query)
                    else: query_book_request(model, dataset, new_query)
                else: query_book_request(model, dataset, new_query)

            case 'author_request': 
                authors = extract_author_keywords(initial_query)
                if authors:
                    results = find_authors_in_dataset(authors)
                    if results: full_query = output_results(results, first, model, dataset, full_query)
                    else: print(get_response("I was unable to find any books by that author."))
                else: print(get_response("I failed to determine what author you wanted to find."))

            case 'book_request':
                book_keywords = extract_book_keywords(initial_query)
                if book_keywords:
                    results = find_books_in_dataset(book_keywords)
                    if results: full_query = output_results(results, first, model, dataset, full_query)
                    else: print(get_response("I was unable find any books by that name."))
                else: print(get_response("I failed to determine what book/series you wanted to find."))

            case 'genre_request':
                genre_keywords = extract_genre_keywords(initial_query)
                if genre_keywords:
                    results = find_genres_in_dataset(genre_keywords)
                    if results: full_query = output_results(results, first, model, dataset, full_query)
                    else: print(get_response("I was unable to find any books in that genre."))
                else: print(get_response("I failed to determine what genre you wanted to find."))

        first = False

if __name__ == "__main__":
    main()