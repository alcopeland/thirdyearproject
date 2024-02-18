def find_authors_in_dataset(authors):
    results = []
    import csv
    import string
    with open('goodreads_data.csv', encoding="utf-8") as file:
        reader = csv.reader(file, delimiter=',')
        next(file)
        for row in reader:
            if any(name in row[1].lower() for name in authors):
                results.append([row[0], row[1], row[2], row[3], row[4], int(row[5].replace(",","")), row[1].lower().translate(str.maketrans('', '', string.punctuation)), 0])
    import spacy
    from math import sqrt
    nlp = spacy.load('en_core_web_sm')
    authors_vectors = []
    for author in authors:
        authors_vectors.append(nlp(author))
    for result in results:
        result_vector = nlp(result[6])
        scores = []
        for author_vector in authors_vectors:
            numerator = sum(a*b for a,b in zip(author_vector.vector,result_vector.vector))
            denominator = sqrt(sum([a*a for a in author_vector.vector]))*sqrt(sum([a*a for a in result_vector.vector]))
            scores.append(numerator/float(denominator))
        result[7] = max(scores)
    sorted_authors = sorted(results, key=lambda x:x[7], reverse=True)
    if sorted_authors[0][7] > 0.65:
        filtered_results = [a for a in sorted_authors if a[7]==sorted_authors[0][7]]
        sorted_results = sorted(filtered_results, key=lambda x:x[5], reverse=True)
        return sorted_results
    else:
        return []
    
def find_genres_in_dataset(genre_keywords):
    results = []
    import csv
    with open('goodreads_data.csv', encoding="utf-8") as file:
        reader = csv.reader(file, delimiter=',')
        next(file)
        for row in reader:
            if int(row[5].replace(",","")) > 1000:
                if any(name in row[3] for name in genre_keywords):
                    results.append([row[0], row[1], row[2], row[3], row[4], int(row[5].replace(",",""))])
    sorted_results = sorted(results, key=lambda x:x[5], reverse=True)
    return sorted_results[:10]

def find_books_in_dataset(book_keywords):
    import csv
    results = []
    lower_book_keywords = []
    for book in book_keywords:
        lower_book_keywords.append(book.lower())

    with open('goodreads_data.csv', encoding="utf-8") as file:
        reader = csv.reader(file, delimiter=',')
        next(file)
        for row in reader:
            if any(book in row[0].lower() for book in lower_book_keywords):
                results.append([row[0], row[1], row[2], row[3], row[4], int(row[5].replace(",",""))])
    sorted_results = sorted(results, key=lambda x:x[5], reverse=True)
    return sorted_results[:10]