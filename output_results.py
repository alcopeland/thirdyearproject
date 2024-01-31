def output_results(results, first, model, dataset, full_query):
    summaries = [a[2] for a in results]
    from text_generation import summarize_summary
    new_summaries = summarize_summary(summaries[:5])
    final_results = tuple(zip([a[0] for a in results], [a[1] for a in results], new_summaries+(['']*(len(summaries)-5)), [a[3] for a in results], [a[4] for a in results], [a[5] for a in results]))
    
    print_results(final_results[:5],'1')
    if len(final_results)>5:
        print_results(final_results[5:],'2')

    new_query = ""
    from extract_keywords import extract_summary_keywords
    for result in results:
        new_query += (result[0]+" ")
        new_query += (((str(result[3]).replace("[","")).replace("]","")).replace("'","")+" ")
    new_query += extract_summary_keywords(summaries)
    full_query = full_query + " " + new_query
    
    print("Would you like to find some similar books?")
    from user_intent import yes_or_no
    if yes_or_no(): 
        from find_similar_books import query_book_request
        if not first:
            print("Would you like to use your previous searches in this search?")
            if yes_or_no(): query_book_request(model, dataset, full_query)
            else: query_book_request(model, dataset, new_query)
        else: query_book_request(model, dataset, new_query)
    return full_query

def print_results(results, intent):
    match intent:
        case '0':
            print("\n\nHere are some similar books you might like: \n")
            for result in results:
                print(f" -- {result[0]} -- ")
                print(f"{result[1]}")
                print("\n")
        case '1':
            print("\n\nHere are some resutls for you: \n")
            for result in results:
                genres = ((str(result[3]).replace("[","")).replace("]","")).replace("'","")
                print(f" -- {result[0]} by {result[1]} -- ")
                print(f"Average rating: {result[4]} from {result[5]} reviews")
                print(f"Genres: {genres}")
                print(f"{result[2]}")
                print("\n")
        case '2':
            for result in results:
                genres = ((str(result[3]).replace("[","")).replace("]","")).replace("'","")
                print(f" -- {result[0]} by {result[1]} -- ")
                print(f"Average rating: {result[4]} from {result[5]} reviews")
                print(f"Genres: {genres}")
                print("\n")