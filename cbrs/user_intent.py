from numpy import ndarray
from text_generation import get_response

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline

def get_initial_query(first: bool) -> tuple[str, ndarray]:
    # If first is True, print a welcome and offer assistance
    if first: print("Hello! " + get_response("How can I be of assistance for you today?"))
    # Else, offer further assistance
    else: print(get_response("What further assistance am I able to offer you with today?"))
    # Take user input and determine the intent of it
    initial_query = input()
    intent = user_intent(initial_query)
    return initial_query, intent

def user_intent(initial_query: str) -> ndarray:
    # Method found at link below
    # https://stackoverflow.com/questions/56836865/how-to-use-nlp-in-python-to-analyze-questions-from-a-chat-conversation
    # Used human-annotated example sentences with intent to train a classifier
    # Initalise a list of example sentences
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
        'end',
        'leave',
        'stop'
    ]
    # Initalise a list of labels to refer to the example sentences
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
        'exit',
        'exit',
        'exit'
    ]
    # Create a Pipeline to vectorise sentences using TF-IDF and then train a classifier using SGD on that data
    clf = Pipeline([('tfidf', TfidfVectorizer()),('sgd', SGDClassifier())])
    # Fit the data to a model using the pipeline
    clf.fit(X, y)
    # Predict the intent of the user input provided using the model created
    predict_intent = clf.predict([initial_query])
    return predict_intent

def yes_or_no() -> bool:
    # Method found at link below
    # https://stackoverflow.com/questions/62156781/how-do-i-get-a-list-of-all-combinations-for-both-words-given
    # Take user input
    query = input()
    # List of different words meaning yes or no in a list
    yes_words = ['yes','okay','yeah','y','ye','yep','sure','ok','cool','please','yes please','certainly','with pleasure']
    no_words = ['no','nope','n','no thanks','no thank you','nay','nah','no way','not']
    X = []
    y = []
    # Add yes words in upper, lower and capitalised case to X and matching labels to Y
    for word in yes_words:
        X+=[word,word.upper(),word.capitalize()]
        y+=['yes','yes','yes']
    # Add no words in upper, lower and capitalised case to X and matching labels to Y
    for word in no_words:
        X+=[word,word.upper(),word.capitalize()]
        y+=['no','no','no']
    # Create a Pipeline to vectorise sentences using TF-IDF and then train a classifier using SGD on that data
    clf = Pipeline([('tfidf', TfidfVectorizer()),('sgd', SGDClassifier())])
    # Fit the data to a model using the pipeline
    clf.fit(X, y)
    # Predict the intent of the user input provided using the model created
    predict_intent = clf.predict([query])
    # Return True if intent is yes
    if predict_intent == 'yes': return True
    # Else return False
    else: return False