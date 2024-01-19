def clean_text(text):
    from nltk.tokenize import word_tokenize
    tokenizedText = word_tokenize(text)

    from nltk.corpus import stopwords
    stop_words = set(stopwords.words('english'))
    cleanText = []
    for word in tokenizedText:
        if word not in stop_words:
            cleanText.append(word)
    cleanTextS = ' '.join(cleanText)
    return cleanTextS

def read_booksdataset(names, labels, texts, original_texts):
    import csv
    with open("BooksDataSet.csv", "r", encoding="utf-8", errors="ignore") as file:
        csv_reader = csv.reader(file)
        next(file)
        for row in csv_reader:
            names.append(row[2])
            cleanText = clean_text(row[4])
            labels.append(row[3]) # put square brackets around to make into list for multiclass classification
            texts.append(f"{row[2]} {row[3]} {cleanText}")
            original_texts.append(row[4])
    return names, labels, texts, original_texts

def read_datasets():
    names, labels, texts, original_texts = read_booksdataset([],[],[],[])
    return names, labels, texts  , original_texts

def write_to_csv(names, labels, texts, original_texts):
    combinedArray = list(zip(names,labels,texts,original_texts))
    import csv
    with open('final_text.csv', 'w+', encoding="utf-8", errors="ignore", newline='') as file:
        writer = csv.writer(file)        
        writer.writerow(["name","label","text","original"])
        writer.writerows(combinedArray)    

def load_dataset_from_csv():
    # https://medium.com/@lokaregns/fine-tuning-transformers-with-custom-dataset-classification-task-f261579ae068 
    from datasets import load_dataset
    dataset = load_dataset('csv', data_files = 'final_text.csv')
    dataset = dataset['train'].train_test_split(test_size=0.2)
    return dataset

def train_naive_bayes(dataset):
    # https://www.turing.com/kb/document-classification-using-naive-bayes
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.naive_bayes import MultinomialNB
    from sklearn.pipeline import make_pipeline    
    from sklearn.metrics import accuracy_score
    
    model = make_pipeline(TfidfVectorizer(), MultinomialNB())
    model.fit(dataset["train"]["text"],dataset["train"]["label"])
    predicted_labels = model.predict(dataset["test"]["text"])
    #print(accuracy_score(dataset["test"]["label"], predicted_labels))
    return model

def predict_user_input(model, query):
    prediction = model.predict([query])
    # print(prediction)
    return prediction[0]

def limit_dataset(prediction, dataset):
    refined_dataset = dataset.filter(lambda record: record["label"]==prediction)
    return refined_dataset

def calculate_bert_score(summary, scorer, query):
    P, R, F1  = scorer.score([query], [summary])
    return F1.detach().numpy()[0]

def calculate_bert_scores(query, dataset):
    from bert_score import BERTScorer
    scorer = BERTScorer(model_type='distilbert-base-uncased')
    scores = []

    # import time
    # start = time.time()
    import concurrent.futures
    with concurrent.futures.ThreadPoolExecutor(max_workers=60) as executor:
        future_scores = [executor.submit(calculate_bert_score, summary, scorer, query) for summary in dataset['train']['text']]
        for score in future_scores:
            scores.append(score.result())
    # print(f'Took {time.time()-start} seconds')
    import numpy as np
    top_10_index = np.argsort(scores)[-10:]
    return top_10_index, scores

def make_summary(input_summary, summarizer):
    # https://www.turing.com/kb/5-powerful-text-summarization-techniques-in-python 
    # from transformers import AutoTokenizer, AutoModelWithLMHead
    # from transformers import T5ForConditionalGeneration, T5Tokenizer

    # https://thepythoncode.com/article/text-summarization-using-huggingface-transformers-python
    text = summarizer(input_summary[:1024], max_length=100, min_length=65, do_sample=False)[0]['summary_text']
    return text
    
def summarize_summary(input_summaries):
    #input_summaries.reverse()
    output_summaries = []
    
    from transformers import pipeline
    # model_id = 'Falconsai/text_summarization'
    # model_id = 'pszemraj/led-large-book-summary'
    # model_id = 'pszemraj/long-t5-tglobal-base-16384-book-summary'
    model_id = 'facebook/bart-large-cnn'
    summarizer = pipeline('summarization', model=model_id)
    
    # import time
    # start = time.time()
    import concurrent.futures
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
        future_scores = [executor.submit(make_summary, summary, summarizer) for summary in input_summaries]
        for score in future_scores:
            output_summaries.append(score.result())
    # print(f'Took {time.time()-start} seconds')
    return output_summaries

def print_results(results, intent):
    match intent:
        case 'general_request':
            print("\n\nHere are some similar books you might like: \n")
            for result in results:
                print(f" -- {result[0]} -- ")
                print(f"{result[1]}")
                print("\n")
        case 'author_request_1':
            print("\n\nHere are some results that might match authors mentioned: \n")
            for result in results:
                print(f" -- {result[0]} -- ")
                print(f"Average rating: {result[2]} from {result[3]} reviews")
                print(f"{result[1]}")
                print("\n")
        case 'author_request_2':
            print("\n\nHere are some more results: \n")
            for result in results:
                print(f" -- {result[0]} -- ")
                print(f"Average rating: {result[2]} from {result[3]} reviews")
                print("\n")
        case 'genre_request_1':
            print("\n\nHere are some resutls for the genres mentioned: \n")
            for result in results:
                genres = ((str(result[3]).replace("[","")).replace("]","")).replace("'","")
                print(f" -- {result[0]} by {result[1]} -- ")
                print(f"Average rating: {result[4]} from {result[5]} reviews")
                print(f"Genres: {genres}")
                print(f"{result[2]}")
                print("\n")
        case 'genre_request_2':
            print("\n\nHere are more resutls: \n")
            for result in results:
                genres = ((str(result[3]).replace("[","")).replace("]","")).replace("'","")
                print(f" -- {result[0]} by {result[1]} -- ")
                print(f"Average rating: {result[4]} from {result[5]} reviews")
                print(f"Genres: {genres}")
                print("\n")

def query_book_request(model, dataset, user_query):
    prediction = predict_user_input(model, user_query)
    refined_dataset = limit_dataset(prediction, dataset)
    import numpy as np
    top_10_index, all_scores = calculate_bert_scores(user_query, refined_dataset)
    top_5_index = top_10_index[:5]
    top_5_names = list(np.array(refined_dataset["train"]["name"])[top_5_index])
    top_5_summaries_short = summarize_summary(list(np.array(refined_dataset["train"]["original"])[top_5_index]))
    results = tuple(zip(top_5_names, top_5_summaries_short))
    print_results(results,'general_request')

    print("Would you like to see more results?")
    if yes_or_no():
        bottom_5_index = top_10_index[6:]
        bottom_5_names = list(np.array(refined_dataset["train"]["name"])[bottom_5_index])
        bottom_5_summaries_short = summarize_summary(list(np.array(refined_dataset["train"]["original"])[bottom_5_index]))
        results = tuple(zip(bottom_5_names, bottom_5_summaries_short))
        print_results(results,'general_request')

def system_setup():
    import sys
    if len(sys.argv)>1:
        names, labels, texts, original_texts = read_datasets()
        write_to_csv(names, labels, texts, original_texts)

    dataset = load_dataset_from_csv()
    model = train_naive_bayes(dataset)
    return model, dataset

def user_intent(initial_query):
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
        'What books have Frank Herbert made?'
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
        'author_request'
    ]
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import SGDClassifier
    from sklearn.pipeline import Pipeline
    clf = Pipeline([('tfidf', TfidfVectorizer()),('sgd', SGDClassifier())])
    clf.fit(X, y)
    predict_intent = clf.predict([initial_query])
    return predict_intent

def get_initial_query(first):
    # https://stackoverflow.com/questions/56836865/how-to-use-nlp-in-python-to-analyze-questions-from-a-chat-conversation
    # use intent/entity extraction to ask some guided questions
    if first: print("How can I help you today?")
    else: print("Tell me about a genre, author or book series you enjoy.")
    initial_query = input()
    intent = user_intent(initial_query)
    return initial_query, intent

def get_author(query):
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

def find_authors_in_dataset(authors):
    results = []
    import csv
    import string
    with open('goodreads_data.csv', encoding="utf-8") as file:
        reader = csv.reader(file, delimiter=',')
        next(file)
        for row in reader:
            if any(name in row[1].lower() for name in authors):
                results.append([row[0], row[1].lower().translate(str.maketrans('', '', string.punctuation)), row[2], row[4], int(row[5].replace(",","")),0])
    import spacy
    from math import sqrt
    nlp = spacy.load('en_core_web_sm')
    authors_vectors = []
    for author in authors:
        authors_vectors.append(nlp(author))
    for result in results:
        result_vector = nlp(result[1])
        scores = []
        for author_vector in authors_vectors:
            numerator = sum(a*b for a,b in zip(author_vector.vector,result_vector.vector))
            denominator = sqrt(sum([a*a for a in author_vector.vector]))*sqrt(sum([a*a for a in result_vector.vector]))
            scores.append(numerator/float(denominator))
        result[5] = max(scores)
    sorted_authors = sorted(results, key=lambda x:x[5], reverse=True)
    if sorted_authors[0][5] > 0.65:
        filtered_results = [a for a in sorted_authors if a[5]==sorted_authors[0][5]]
        sorted_results = sorted(filtered_results, key=lambda x:x[4], reverse=True)
        return sorted_results
    else:
        return []

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

def get_book_keywords(query):
    import spacy
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(query)
    keywords = []
    for word in doc:
        print(f"{word.text} {word.pos_}")
        if word.pos_ == 'PROPN':
            keywords.append(word.text)
        if word.pos_ == 'NOUN':
            keywords.append(word.text)
    from nltk import ne_chunk, pos_tag, word_tokenize
    from nltk.tree import Tree
    nltk_results = ne_chunk(pos_tag(word_tokenize(query)))
    for nltk_result in nltk_results:
        if type(nltk_result) == Tree:
            name = ''
            for nltk_result_leaf in nltk_result.leaves():
                name += nltk_result_leaf[0] + ' '
            keywords.append(name)
    return list(set(keywords))

def get_genre_keywords(query):
    import spacy
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(query)
    keywords = []
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
    return list(set(keywords+stems))

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
    return sorted_results[:20]

def main():
    print("System starting...")
    model, dataset = system_setup()
    loop = True
    first = True
    while loop:
        initial_query, intent = get_initial_query(first)
        print(intent)
        if intent=='general_request':
            print("in")
            query = input()
            query_book_request(model, dataset, query)

            # ask fixed questions to get info about what books they like
            # run general lookup based on query
        match intent:
            case 'author_request': 
                authors = get_author(initial_query)
                results = find_authors_in_dataset(authors)
                summaries = [a[2] for a in results]
                if len(summaries)!=0:
                    new_summaries = summarize_summary(summaries[:5])
                    final_results = tuple(zip([a[0] for a in results], new_summaries+(['']*(len(summaries)-5)), [a[3] for a in results], [a[4] for a in results]))
                    
                    print_results(final_results[:5],'author_request_1')
                    if len(final_results)>5:
                        print("Would you like to see more?")
                        if yes_or_no():
                            print_results(final_results[6:],'author_request_2')
                    print("Would you like to find some similar books?")
                    if yes_or_no():
                        clean_summaries = []
                        for summary in summaries:
                            clean_summaries.append(clean_text(summary))
                        query = ''.join(clean_summaries)
                        query_book_request(model, dataset, query)
                else:
                    print("Sorry, I was unable to find any books by that author.")

            case 'book_request':
                book_keywords = get_book_keywords(initial_query)
                print(book_keywords)
            # If book
                # works similar to author search
                # get book name from user input
                # search larger dataset for other books in series
            case 'genre_request':
                genre_keywords = get_genre_keywords(initial_query)
                results = find_genres_in_dataset(genre_keywords)
                summaries = [a[2] for a in results]
                if len(summaries)!=0:
                    new_summaries = summarize_summary(summaries[:5])
                    final_results = tuple(zip([a[0] for a in results], [a[1] for a in results], new_summaries+(['']*(len(summaries)-5)), [a[3] for a in results], [a[4] for a in results], [a[5] for a in results]))
                    
                    print_results(final_results[:5],'genre_request_1')
                    if len(final_results)>5:
                        print("Would you like to see more?")
                        if yes_or_no():
                            print_results(final_results[6:],'genre_request_2')
                    print("Would you like to find some similar books?")
                    if yes_or_no():
                        clean_summaries = []
                        for summary in summaries:
                            clean_summaries.append(clean_text(summary))
                        query = ''.join(clean_summaries)
                        query_book_request(model, dataset, query)
                else:
                    print("Sorry, I was unable to find any books in that genre.")
        print("Would you like to continue?")
        if yes_or_no(): first = False
        else: loop = False
    print("Goodbye.")

if __name__ == "__main__":
    main()