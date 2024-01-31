def get_initial_query(first):
    if first: print("Greetings! How can I help you today?")
    else: print("How else can I assist you today?")
    initial_query = input()
    intent = user_intent(initial_query)
    return initial_query, intent

def user_intent(initial_query):
    # https://stackoverflow.com/questions/56836865/how-to-use-nlp-in-python-to-analyze-questions-from-a-chat-conversation
    X = [
        'I am looking for a good book to read.',
        'What is a good book to read?',
        "hey, I'm looking for a new book to read, any suggestions?",
        'Give me recommendations for books to read.',
        'What are some popular books I might not have read?',
        'I am looking for a good book to read, nothing specific.',
        'What are some must-read books?',
        'Book recommendations for teens / young adults.',
        "I'm looking for some good fantasy books, any recommendations?",
        'What are some good action books?',
        'I am really into science fiction books, do you know any books I might like?',
        'I enjoy reading thrillers, do you know any books I might like?',
        'Do you know any good crime fiction or thriller books?',
        'Please can you list some good books to read?',
        'List some epic fantasy books, please.',
        'Got any good fantasy/sci-fi books?',
        'What are some books like Harry Potter?',
        'Books similar to Game of Thrones',
        'What books are like Airborne?',
        'Books set in space.',
        'Adventure books.',
        'Books similar to Star Wars.',
        'What books are linked to the Harry Potter series?',
        'What books are in the Game of Thrones series?',
        'What are some highly rated books?',
        'List books in the Mistborn series',
        'What are some other books written by George Martin?',
        'Books by J. R. R. Tolkien',
        'Other books by Tolkien',
        'List books written by Brandon Sanderson',
        'I like Orwell novels, any suggestions?',
        'More books by stephen king',
        'Similar books by Roald Dahl',
        'If i like books written by JRR Tolkien, what else might I enjoy?',
        'What books are written by Agatha Christie?',
        'What are some good spy novels?',
        'tell me similar books to Stormlight Archive',
        'I would like a book recommendation',
        'list any books that Frank Herbert composed',
        'What books did Kenneth Oppel write',
        'novels written by Stuart Hill',
        'books that JK Rowling wrote',
        'books in the Dune set',
        'What books have Frank Herbert made?',
        'I would like some fantasy book recommendations',
        "I'm in the mood for a great romance book. Any recommendations?",
        "Looking for adventure books",
        "feeling like a young adult novel",
        'I want to dive into classic mystery literature',
        'In search of a gripping horror page-turner',
        'searching for a good science fiction book',
        'want a sci-fi book',
        'What are some books like the Wheel of Time series?',
        'Is there anything like to the Gone book set?',
        'exit',
        'close',
        'done',
        'finished',
        'end'
    ]
    y = [
        'general_request',
        'general_request',
        'general_request',
        'general_request',
        'general_request',
        'general_request',
        'general_request',
        'general_request',
        'genre_request',
        'genre_request',
        'genre_request',
        'genre_request',
        'genre_request',
        'general_request',
        'genre_request',
        'genre_request',
        'book_request',
        'book_request',
        'book_request',
        'genre_request',
        'genre_request',
        'book_request',
        'book_request',
        'book_request',
        'general_request',
        'book_request',
        'author_request',
        'author_request',
        'author_request',
        'author_request',
        'author_request',
        'author_request',
        'author_request',
        'author_request',
        'author_request',
        'genre_request',
        'book_request',
        'general_request',
        'author_request',
        'author_request',
        'author_request',
        'author_request',
        'book_request',
        'author_request',
        'genre_request',
        'genre_request',
        'genre_request',
        'genre_request',
        'genre_request',
        'genre_request',
        'genre_request',
        'genre_request',
        'book_request',
        'book_request',
        'exit',
        'exit',
        'exit',
        'exit',
        'exit'
    ]
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import SGDClassifier
    from sklearn.pipeline import Pipeline
    clf = Pipeline([('tfidf', TfidfVectorizer()),('sgd', SGDClassifier())])
    clf.fit(X, y)
    predict_intent = clf.predict([initial_query])
    return predict_intent

def yes_or_no():
    # https://stackoverflow.com/questions/62156781/how-do-i-get-a-list-of-all-combinations-for-both-words-given
    query = input()
    yes_words = ['yes','okay','yeah','y','ye','yep','sure','ok','cool','please','yes please','certainly','with pleasure']
    no_words = ['no','nope','n','no thanks','no thank you','nay','nah','no way','not']
    X = []
    y = []
    for word in yes_words:
        X+=[word,word.upper(),word.capitalize()]
        y+=['yes','yes','yes']
    for word in no_words:
        X+=[word,word.upper(),word.capitalize()]
        y+=['no','no','no']
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import SGDClassifier
    from sklearn.pipeline import Pipeline
    clf = Pipeline([('tfidf', TfidfVectorizer()),('sgd', SGDClassifier())])
    clf.fit(X, y)
    predict_intent = clf.predict([query])
    if predict_intent == 'yes':
        return True
    else:
        return False