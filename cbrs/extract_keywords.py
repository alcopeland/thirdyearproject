import spacy
import string
from nltk import ne_chunk, pos_tag, word_tokenize
from nltk.tree import Tree

from nltk.stem import PorterStemmer

from keybert import KeyBERT
import concurrent.futures

def extract_author_keywords(query: str) -> list:
    # Loads a spaCy pipeline to tokenise text
    nlp = spacy.load("en_core_web_sm")
    # Tokenises the text into a document
    doc = nlp(query)
    author_aliases = []
    # Goes through each word in the text and extracts proper nouns and entities found by spaCy
    for word in doc:
        if word.pos_=='PROPN':
            author_aliases.append(word.text.lower().translate(str.maketrans('', '', string.punctuation)))
    for entity in doc.ents:
            author_aliases.append(entity.text.lower().translate(str.maketrans('', '', string.punctuation)))
    # Also tries to extract author names using nltk chunks
    nltk_results = ne_chunk(pos_tag(word_tokenize(query)))
    for nltk_result in nltk_results:
        if type(nltk_result) == Tree:
            name = ''
            for nltk_result_leaf in nltk_result.leaves():
                name += nltk_result_leaf[0] + ' '
            author_aliases.append(name.lower().translate(str.maketrans('', '', string.punctuation)))
    # Returns a list of potential names with no repeats
    return  list(set(author_aliases))

def extract_book_keywords(query: str) -> list:
    # Loads a spaCy pipeline to tokenise text
    nlp = spacy.load("en_core_web_sm")
    # Tokenises the text into a document
    doc = nlp(query)
    keywords = []
    # Method found at link below
    # https://spacy.io/usage/linguistic-features#named-entities
    # Use spaCy's named entity recognition to find potential book names
    for entity in doc.ents:
        keywords.append(entity.text)
    # If it couldn't find any, ask use to input it by hand
    if not keywords:
        from text_generation import get_response
        print(get_response("Please re-enter the name of the book/series again."))
        answer = input()
        # Just in case, take user input in lower and capitalised form
        keywords.append([answer.lower(), answer.capitalize()])
    fail_safe = []
    # As another precaution, remove the word 'the' from any names
    for name in keywords:
        if "the" in name:
            fail_safe.append(name.replace("the ",''))
    # Return the list of potential names with no repeats
    return list(set(keywords+fail_safe))

def extract_genre_keywords(query: str) -> list:
    # Loads a spaCy pipeline to tokenise text
    nlp = spacy.load("en_core_web_sm")
    # Tokenises the text into a document
    doc = nlp(query)
    keywords = []
    # Takes potential genre keywords as proper nouns, nouns and adjectives
    for word in doc:
        if word.pos_ == 'PROPN':
            keywords.append(word.text.capitalize())
        if word.pos_ == 'NOUN':
            keywords.append(word.text.capitalize())
        if word.pos_ == 'ADJ':
            keywords.append(word.text.capitalize())
    # Creates a stemmer to stem potential genres to their roots, increasing chances to match other spellings in the genres
    stemmer = PorterStemmer()
    stems = []
    for word in keywords:
        stems.append(stemmer.stem(word).capitalize())
    keywords = list(set(keywords+stems))
    final_keywords = []
    book_words = ["Book","Novel","Text","Tome"]
    # Remove the nouns book, novel, text and tome from the list as it messes up the search
    for word in keywords:
        if all(book not in word for book in book_words):
            final_keywords.append(word)
    return final_keywords

def extract_summary_keywords(texts: list) -> str:
    # Method found at links below
    # https://pypi.org/project/keybert/
    # https://www.sbert.net/docs/pretrained_models.html 
    # Loads the KeyBERT model using 'all-MiniLM-L6-v2'
    model = KeyBERT(model='all-MiniLM-L6-v2') #all-mpnet-base-v2
    keywords_list = []
    # import time
    # start = time.time()
    # Uses a thread pool with 5 workers to extract keywords simultaneously 
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        future_scores = [executor.submit(extract_single_keywords, text, model) for text in texts]
        for score in future_scores: keywords_list.append(score.result())
    # Join keywords from all the summaries into a single string and return 
    keywords = ' '.join(keywords_list)
    # print(f'Extracting keywords took {time.time()-start} seconds')
    return keywords

def extract_single_keywords(text: str, model: KeyBERT) -> str:
    # Takes a single string and extracts the top 20 keywords, and top 5 n-gram phrases and stoes them all in a string
    keywords = model.extract_keywords(text, keyphrase_ngram_range=(1, 1), stop_words='english', highlight=False, top_n=20)
    keyphrases = model.extract_keywords(text, keyphrase_ngram_range=(1, 2), stop_words='english', highlight=False, top_n=5)
    keywords = keywords + keyphrases
    keywords = ' '.join(list(dict(keywords).keys()))
    return keywords