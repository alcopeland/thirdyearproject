def extract_author_keywords(query):
    import spacy
    import string
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(query)
    author_aliases = []
    for word in doc:
        if word.pos_=='PROPN':
            author_aliases.append(word.text.lower().translate(str.maketrans('', '', string.punctuation)))
    for entity in doc.ents:
            author_aliases.append(entity.text.lower().translate(str.maketrans('', '', string.punctuation)))
    import nltk
    from nltk import ne_chunk, pos_tag, word_tokenize
    from nltk.tree import Tree
    nltk_results = ne_chunk(pos_tag(word_tokenize(query)))
    for nltk_result in nltk_results:
        if type(nltk_result) == Tree:
            name = ''
            for nltk_result_leaf in nltk_result.leaves():
                name += nltk_result_leaf[0] + ' '
            author_aliases.append(name.lower().translate(str.maketrans('', '', string.punctuation)))
    return  list(set(author_aliases))

def extract_book_keywords(query):
    import spacy
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(query)
    keywords = []
    # https://spacy.io/usage/linguistic-features#named-entities
    for entity in doc.ents:
        keywords.append(entity.text)
    if not keywords:
        from text_generation import get_response
        print(get_response("Please re-enter the name of the book/series again."))
        answer = input()
        keywords.append([answer.lower(), answer.capitalize()])
    fail_safe = []
    for name in keywords:
        if "the" in name:
            fail_safe.append(name.replace("the ",''))
    return list(set(keywords+fail_safe))

def extract_genre_keywords(query):
    import spacy
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(query)
    keywords = []
    # https://spacy.io/usage/linguistic-features#named-entities
    for word in doc:
        #print(f"{word.text} {word.pos_}")
        if word.pos_ == 'PROPN':
            keywords.append(word.text.capitalize())
        if word.pos_ == 'NOUN':
            keywords.append(word.text.capitalize())
        if word.pos_ == 'ADJ':
            keywords.append(word.text.capitalize())
    from nltk.stem import PorterStemmer
    stemmer = PorterStemmer()
    stems = []
    for word in keywords:
        stems.append(stemmer.stem(word).capitalize())
    keywords = list(set(keywords+stems))
    final_keywords = []
    book_words = ["Book","Novel","Text","Tome"]
    for word in keywords:
        if all(book not in word for book in book_words):
            final_keywords.append(word)
    return final_keywords

def extract_summary_keywords(texts):
    # https://pypi.org/project/keybert/
    # https://www.sbert.net/docs/pretrained_models.html 
    from keybert import KeyBERT
    model = KeyBERT(model='all-mpnet-base-v2')
    keywords_list = []
    # import time
    # start = time.time()
    import concurrent.futures
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        future_scores = [executor.submit(extract_single_keywords, text, model) for text in texts]
        for score in future_scores:
            keywords_list.append(score.result())
    keywords = ' '.join(keywords_list)
    # print(f'Extracting keywords took {time.time()-start} seconds')
    return keywords

def extract_single_keywords(text, model):
    keywords = model.extract_keywords(text, keyphrase_ngram_range=(1, 1), stop_words='english', highlight=False, top_n=10)
    keywords = ' '.join(list(dict(keywords).keys()))
    return keywords