import spacy
from spacy import displacy
from pathlib import Path
text = "An example tagged sentence."
nlp = spacy.load("en_core_web_sm")
doc = nlp(text)
options = options = {"ents": None,
           "compact": True}
# displacy.serve(doc, options= options, style = "dep", host="127.0.0.1")
svg = displacy.render(doc, options= options, style = "dep")
filename = 'pos_tagging.svg'
output_path = Path('./images/' + filename)
output_path.open('w', encoding='utf-8').write(svg)
