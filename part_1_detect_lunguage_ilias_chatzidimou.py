import nltk
from nltk.corpus import udhr
from nltk import FreqDist
from nltk.metrics import spearman_correlation

def extract_language_data(lang_code):
    lang_data = ""
    for fileid in udhr.fileids():
        if fileid.startswith(lang_code):
            lang_data += udhr.raw(fileid)
    return lang_data

def get_char_freq(text):
    char_freq = FreqDist(text)
    return char_freq

def identify_language(unknown_text):
    known_languages = ['eng', 'fra', 'spa', 'deu', 'ita', 'por', 'nld', 'dan', 'swe', 'fin', 'nor', 'pol', 'hun', 'ron', 'ces', 'slk', 'hrv', 'bul', 'ell']
    correlations = {}
    
    for lang in known_languages:
        known_data = extract_language_data(lang)
        unknown_freq = get_char_freq(unknown_text)
        known_freq = get_char_freq(known_data)
        spearman_corr = spearman_correlation(unknown_freq, known_freq)
        correlations[lang] = spearman_corr

    identified_lang = max(correlations, key=correlations.get)
    return identified_lang

import langid

unknown_text = "Your unknown text here"
identified_language = langid.classify(unknown_text)[0]
print("Identified language:", identified_language)

