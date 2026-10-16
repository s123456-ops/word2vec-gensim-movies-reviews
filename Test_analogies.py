from gensim.models import Word2Vec
from gensim.test.utils import datapath

fichier = datapath("questions-words.txt")

for nom, chemin in [("Films", "Results/movie_word2vec_clean.model"),
                    ("Avis Amazon", "Results/reviews_word2vec_corrected.model")]:
    wv = Word2Vec.load(chemin).wv
    score, sections = wv.evaluate_word_analogies(fichier)
    total = sections[-1]
    testees = len(total["correct"]) + len(total["incorrect"])
    print(f"\n{nom} : {len(total['correct'])} bonnes réponses sur {testees} questions testées "
          f"-> score = {score:.1%}")
    for s in sections[:-1]:
        n = len(s["correct"]) + len(s["incorrect"])
        if n:
            print(f"   {s['section']:<28} {len(s['correct']):>4} / {n:<5} ({len(s['correct'])/n:.1%})")