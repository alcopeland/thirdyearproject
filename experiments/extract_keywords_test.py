from keybert import KeyBERT
import time

# A short program I used to be able to quickly test KeyBERT keyword extraction using different models and test their speeds

def main(text):
    # Method found at links below
    # https://pypi.org/project/keybert/
    # https://www.sbert.net/docs/pretrained_models.html 
    start = time.time()
    #model = KeyBERT(model='all-mpnet-base-v2')
    model = KeyBERT(model='all-MiniLM-L6-v2')
    keywords = model.extract_keywords(text, keyphrase_ngram_range=(1, 1), stop_words='english', highlight=False, top_n=20)
    keyphrases = model.extract_keywords(text, keyphrase_ngram_range=(1, 2), stop_words='english', highlight=False, top_n=5)
    keywords = ' '.join(list(dict(keywords).keys()))
    keywords = ' '.join(list(dict(keyphrases).keys()))
    print(f'Extracting keywords took {time.time()-start} seconds')
    print(f"Keywords: {keywords}")
    print(f"Keyphrases: {keyphrases}")

if __name__ == "__main__": main("Cugel is easily persuaded by the merchant Fianosther to attempt the burglary of the manse of Iucounu the Laughing Magician. Trapped and caught, he agrees that in exchange for his freedom he will undertake the recovery of a small hemisphere of violet glass, an Eye of the Overworld, to match one already in the wizard's possession. A small sentient alien entity of barbs and hooks, named Firx, is attached to his liver to encourage his ""unremitting loyalty, zeal and singleness of purpose,"" and Iucounu uses a spell to transport Cugel via flying demon to the remote Land of Cutz. There, Cugel finds two villages, one occupied by wearers of the violet lenses, the other by peasants who work on behalf of the lens-wearers, in hopes of being promoted to their ranks. The lenses cause their wearers to see, not their squalid surroundings, but the Overworld, a vastly superior version of reality where a hut is a palace, gruel is a magnificent feast, etc.   ""seeing the world through rose-colored glasses"" on a grand scale. Cugel gains an Eye by trickery, and escapes from Cutz. He then undertakes an arduous trek back to Iucounu, cursing the magician the entire way; this forms the principal part of the book. After many pitfalls, setbacks, and harrowing escapes, including the eviction of Firx from his system, Cugel returns to Iucounu's manse, where he finds the wizard's volition has been captured by a twin to Firx. Cugel manages to extirpate the alien, subdue the magician, and enjoy the easy life in the manse, until he tries to banish Iucounu and Fianosther (who himself has come to pilfer from Cugel) with the same spell that the magician had used on him. But Cugel's tongue slips in uttering the incantation, and the flying demon seizes him instead, delivering him to the same spot as before. Author Michael Shea wrote an authorized sequel, A Quest for Simbilis (DAW Books, NY, 1974). Vance's own Cugel sequel was published as Cugel's Saga in 1983.")