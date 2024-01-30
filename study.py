def main():
    from sumy.parsers.plaintext import PlaintextParser
    from sumy.nlp.tokenizers import Tokenizer
    from sumy.summarizers.lex_rank import LexRankSummarizer
    summaries = [
        "Stories of Music explores the incredible ways in which music has impacted people's lives, from its role in healing, community, and family relationships to musicianship, travel, and much more. Bringing together the works of more than 40 authors and artists from 11 countries, the book features poetry, nonfiction, photography, audio, and video (the print book includes URLs and QR codes that direct readers to a free companion web edition for accessing audio and video components). Readers will learn how rock and blues music helped to heal the war-torn country of Bosnia, about the tradition of candombe drumming in Uruguay, and about the history of musicians who travelled on foot—from the balladeers of Victorian England and the Delta bluesmen of the early 20th century to present day musicians who participate in the Massachusetts Walking Tour. Along with these and other stories, the book includes photography from around the world, poetry readings, and original music, including a song that was performed to honor first responders at Ground Zero. The book was compiled and edited by Holly E. Tripp, and also includes a foreword by Dan Cohen, MSW, Founder and Executive Director of MUSIC & MEMORY℠. Ten percent of book proceeds will be donated to Hungry for Music and Music & Memory. Learn more about their work at www.hungryformusic.org and www.musicandmemory.org.",
        "A stunning new approach to how individuals can not only change their lives for the better in the workplace, but also their lives away from the office, including (but not limited to) finding ways to improve one's working relationship with others, one's overall health, outlook on life, and so on. For example, why is it that 95% of all diet attempts fail? Why do New Year's Resolutions last no more than a few days? Why can't people with good intentions seem to make consistent and positive strides in the way they want to improve their careers, financial fitness, physical fitness, and so on? Based upon the latest research in a number of psychological and medical fields, the authors of CHANGE ANYTHING will show that traditional will-power is not necessarily the answer to these strivings, that people are affected in their behaviors by far more subtle influences. CHANGE ANYTHING shows how individuals can come to understand these powerful and influential forces, and how to put these forces to work in a positive manner that brings real and meaningful results. The authors present an array of everyday examples that will change and truly empower you to reexamine the way you go about your business and life.",
        "Best known for the 1892 title story of this collection, a harrowing tale of a woman's descent into madness, Charlotte Perkins Gilman wrote more than 200 other short stories. Seven of her finest are reprinted here. Written from a feminist perspective, often focusing on the inferior status accorded to women by society, the tales include ""turned,"" an ironic story with a startling twist, in which a husband seduces and impregnates a naïve servant; ""Cottagette,"" concerning the romance of a young artist and a man who's apparently too good to be true; ""Mr. Peebles' Heart,"" a liberating tale of a fiftyish shopkeeper whose sister-in-law, a doctor, persuades him to take a solo trip to Europe, with revivifying results; ""The Yellow Wallpaper""; and three other outstanding stories. These charming tales are not only highly readable and full of humor and invention, but also offer ample food for thought about the social, economic, and personal relationship of men and women — and how they might be improved.",
        "There are some places in this world that go far beyond any normal definition of “haunted.” These places are so evil, so diabolical, that they become gateways to Hell itself. The Fuller Farm is one such place. It is said that old man Fuller conducted unspeakable acts, blood rituals and human sacrifices, all in an attempt to gain the ultimate knowledge, the ultimate power. And then, he was killed–horribly murdered on his own lands, leaving the house to stand as a vacant monument to his wickedness. But once a door is opened, it can never really be closed. Now, the stars are right. The gateway is ready to once more unleash unspeakable horror upon the town of Harmony, Indiana. And this will be one Halloween that they will never forget!",
        "Olivia Kaspen has just discovered that her ex-boyfriend, Caleb Drake, has lost his memory. With an already lousy reputation for taking advantage of situations, Olivia must decide how far she is willing to go to get Caleb back. Wrestling to keep her true identity and their sordid past under wraps, Olivia’s greatest obstacle is Caleb’s wicked, new girlfriend; Leah Smith. It is a race to the finish as these two vipers engage in a vicious tug of war to possess a man who no longer remembers them. But, soon enough Olivia must face the consequences of her lies, and in the process discover that sometimes love falls short of redemption."
    ]
    import time
    start = time.time()
    for summary in summaries:
        print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
        parser=PlaintextParser.from_string(summary,Tokenizer("english"))
        summarizer = LexRankSummarizer()
        result1 = summarizer(parser.document,1)
        print(result1[0])
        sentences = []
        for sentence in result1:
            sentences.append(str(sentence))
        text = ' '.join(sentences)
        print(text)
    print(f"Took {time.time()-start} seconds")

    start = time.time()
    for summary in summaries:
        print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
        from transformers import pipeline
        model_id = 'facebook/bart-large-cnn'
        summarizer2 = pipeline('summarization', model=model_id)
        text2 = summarizer2(summary[:1024], max_length=100, min_length=65, do_sample=False)[0]['summary_text']
        print(text2)
    print(f"Took {time.time()-start} seconds")

if __name__ == "__main__":
    main()