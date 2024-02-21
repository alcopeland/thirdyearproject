import csv
import string
import spacy
from math import sqrt

def find_authors_in_dataset(authors: list) -> list:
    # Takes a list of possible author aliases and searches for books by authors in dataset 2
    results = []
    # Reads through each row in dataset 2 and checks if any alias is in the author column
    with open('goodreads_data.csv', encoding="utf-8") as file:
        reader = csv.reader(file, delimiter=',')
        next(file)
        for row in reader:
            if any(name in row[1].lower() for name in authors):
                # Stores the book title, author, description, genre, rating, rating number and author name in lowercase without punctuation for each match as well as 0 for use later
                results.append([row[0], row[1], row[2], row[3], row[4], int(row[5].replace(",","")), row[1].lower().translate(str.maketrans('', '', string.punctuation)), 0])
    # Loads a spaCy pipeline to tokenise text
    nlp = spacy.load('en_core_web_sm')
    authors_vectors = []
    # Tokenise the alias list into a vector to perform cosine similarity with authors found in dataset 2 
    for author in authors:
        authors_vectors.append(nlp(author))
    # For each result found, calculate cosine similarity to each author alias
    # Store the highest similarity value in the list of results final element
    for result in results:
        result_vector = nlp(result[6])
        scores = []
        for author_vector in authors_vectors:
            # Calculates cosine similarity between two strings
            numerator = sum(a*b for a,b in zip(author_vector.vector,result_vector.vector))
            denominator = sqrt(sum([a*a for a in author_vector.vector]))*sqrt(sum([a*a for a in result_vector.vector]))
            scores.append(numerator/float(denominator))
        result[7] = max(scores)
    # Sort the results in order of similarity
    sorted_authors = sorted(results, key=lambda x:x[7], reverse=True)
    # If the highest similarity is above 0.65 than count it as a match
    if sorted_authors[0][7] > 0.65:
        # Filter the results to only contain those written by the highest similarity author and sort by number of reviews
        filtered_results = [a for a in sorted_authors if a[7]==sorted_authors[0][7]]
        sorted_results = sorted(filtered_results, key=lambda x:x[5], reverse=True)
        return sorted_results
    # Else return an empty list as it is not close enough to be a match
    else: return []
    
def find_genres_in_dataset(genre_keywords: list) -> list:
    results = []
    # Takes a list of possible genre keywords and searches for books of those genres in dataset 2
    # Reads through each row in dataset 2 and see if any genre keywords are present in the genres for each book in it has more than 1000 reviews
    with open('goodreads_data.csv', encoding="utf-8") as file:
        reader = csv.reader(file, delimiter=',')
        next(file)
        for row in reader:
            if int(row[5].replace(",","")) > 1000:
                if any(name in row[3] for name in genre_keywords):
                    # Stores the book title, author, description, genre, rating and rating number
                    results.append([row[0], row[1], row[2], row[3], row[4], int(row[5].replace(",",""))])
    # Sort the results by number of reviews
    sorted_results = sorted(results, key=lambda x:x[5], reverse=True)
    # Return the top 10 sorted results
    return sorted_results[:10]

def find_books_in_dataset(book_keywords: list) -> list:
    # Takes a list of possible books and searches for books of those names in dataset 2
    results = []
    lower_book_keywords = []
    # Makes all names lowercase to make searching easier
    for book in book_keywords:
        lower_book_keywords.append(book.lower())
    # Reads through each row in dataset 2 and see if any genre keywords are present in the genres for each book in it has more than 1000 reviews
    with open('goodreads_data.csv', encoding="utf-8") as file:
        reader = csv.reader(file, delimiter=',')
        next(file)
        for row in reader:
            if any(book in row[0].lower() for book in lower_book_keywords):
                # Stores the book title, author, description, genre, rating and rating number
                results.append([row[0], row[1], row[2], row[3], row[4], int(row[5].replace(",",""))])
    # Sort the results by number of reviews
    sorted_results = sorted(results, key=lambda x:x[5], reverse=True)
    # Return the top 10 sorted results
    return sorted_results[:10]