import json
import re
import os
from nltk import flatten
from azureml.core.model import Model
from pathlib import Path
from os import path
import spacy
from spacy.lang.en import English
import logging
import glob

# Import NLP architect
from nlp_architect.models.absa.inference.inference import SentimentInference
from spacy.cli.download import download as spacy_download

# Load language model
nlp = English()
sentencizer = nlp.create_pipe("sentencizer")
nlp.add_pipe(sentencizer)

def init():
    global inference
    spacy_download('en')
    aspect_lex_path = Model.get_model_path('c_aspect_lex')
    opinion_lex_path = Model.get_model_path('c_opinion_lex') 
    print("%------------------------------------------%")
    print("aspect_lex_path: ", Path(aspect_lex_path))
    print("current wd: ", os.getcwd())
    path = Path(aspect_lex_path)
    print("pathlib-exists()---->",path.exists())
    print("Path :", path)
    print("Parent :", Path(aspect_lex_path).parent.parent.parent)
    print(os.listdir(Path(aspect_lex_path).parent.parent.parent))
    print("%-----------------------------------------%")
    inference = SentimentInference(aspect_lex_path, opinion_lex_path)
    
def run(raw_data):
    try:
        line_1 = json.loads(raw_data)['name']
        line_2 = json.loads(raw_data)['news']
        sentiment_doc = inference.run(doc = line_2)
        if sentiment_doc != None:
            labels = doc2label(sentiment_doc)
            labels['_vendor_name'] = line_1
            labels = labels_enhancer(labels)
            return labels        
    except Exception as e:
        error = str(e)
        return error
    
# Custom functions
def word_freq(word_list):
    """
    Return Polarity    
    """
    word_freq = [word_list.count(w) for w in word_list]
    return(dict(zip(word_list, word_freq)))

def doc2label(doc):
    """
    Converts ABSA Inference Doc to Sentiment Labels  
    """
    documents = {}
    sentences_list = []
    line_json = json.loads(doc.json())
    text = line_json['_doc_text']
    doc = nlp(text)
    num_sents = len(list(doc.sents))
    sents = line_json['_sentences']
    events = []
    for i in range(len(sents)):
        for e in sents[i]['_events']:
            for ev in e:
                if ev['_type'] == 'OPINION':
                     events.append(ev)
    events = {d['_text']:d for d in events}.values() # get unique events
    tokens = text.split()
    io = [[re.sub(r'(\,)|(\.)|(\')|(\))|(\()|(\!)|(\")', '', token), 'O'] for token in tokens] # remove punctuation from token terms
    index = 0
    for token_id, token in enumerate(tokens):
        for event in events:
            if event['_start'] == index:
                io[token_id][1] = "<{}>".format(event['_polarity'])
        index += len(token) + 1
    io = flatten(io)
    while 'O' in io:
        io.remove('O')
    output = " ".join([l for l in io])
    # collect opinion terms for review
    for id, sent in enumerate(sents):
        sent_polarity = {}
        s = text[sent['_start']:sent['_end'] + 1]
        s_tokens = s.split()
        s_io = [[re.sub(r'(\,)|(\.)|(\')|(\))|(\()|(\!)|(\")', '', tok), 'O'] for tok in s_tokens]
        terms_dict = {}
        pos_terms_list, neg_terms_list = [], []
        for tok_id, tok in enumerate(s_tokens):
            for event in events:
                if event['_text'] == re.sub(r'(\,)|(\.)|(\')|(\))|(\()|(\!)|(\")', '', tok):
                    s_io[tok_id][1] = "<{}>".format(event['_polarity'])
                    if event['_polarity'] == 'POS':
                        if event['_polarity'] in terms_dict:
                            pos_terms_list.append(event['_text'])
                            terms_dict[event['_polarity']] = pos_terms_list
                        else:
                            pos_terms_list.append(event['_text'])
                            terms_dict[event['_polarity']] = pos_terms_list
                    else:
                        if event['_polarity'] in terms_dict:
                            neg_terms_list.append(event['_text'])
                        else:
                            neg_terms_list.append(event['_text'])
                            terms_dict[event['_polarity']] = neg_terms_list
        s_io = flatten(s_io)
        while 'O' in s_io:
            s_io.remove('O')
        sentence = " ".join([l for l in s_io])
        sent_polarity['sentence ' + str(id + 1)] = sentence # sentence dict
        sent_polarity['_opinion_terms'] = terms_dict
        sentences_list.append(sent_polarity)
    documents['_news_text'] = text
    documents['_sentences'] = sentences_list
    documents['#sents_actual'] = num_sents
    documents['#sents_model'] = len(sents)
    documents['#sents_no_model'] = num_sents - len(sents)
    return documents

def labels_enhancer(documents):
    scores = {}
    name = re.sub(r'(\,)|(\.)|(\))|(\()', '', documents['_vendor_name'])
    keyword = name.lower().split()
    keyword_short = ''.join([word[0] for word in keyword])
    keyword = ' '.join([word for word in keyword])
    keyword_split = keyword.split()
    regex = re.compile(r'\b(?:%s)' %  '|'.join(flatten([keyword_short, keyword_split, keyword])))
    lookup_index_wt_name, lookup_index_wo_name = [], []
    # Lookup vendor name string in sentences and calculate polarity
    for sent in documents['_sentences']:
        polarity = ''
        lookup = [1 if re.search(regex, str(v).lower()) else 0 for k, v in sent.items()][0]
        kv_pair = {'_vendor_name':lookup}
        sent.update(kv_pair)
        # re-calculate sentences polarity for tie cases
        sent_polarity = re.findall(r'<NEG>|<POS>', [v for k, v in sent.items()][0])
        _sentence = [v for k, v in sent.items()][0]
        _vendor_name = [v for k, v in sent.items()][1]
        sent_polarity = [re.sub(r'<|>', '', i) for i in sent_polarity]
        if sent_polarity:
            # polarity = ''
            pol_list = word_freq(sent_polarity)
            _pos = max([v if k == 'POS' else 0 for k, v in pol_list.items()])
            _neg = max([v if k == 'NEG' else 0 for k, v in pol_list.items()])
            if _pos == _neg:
                words_cnt = len(_sentence.split(' '))
                if _vendor_name == 1:
                    pos_score = (1 / words_cnt) * 1 * 1
                    neg_score = (1 / words_cnt) * 1 * 1.5 
                    if pos_score < neg_score:polarity = 'Negative'
                    else:polarity = 'Positive'
                else:
                    pos_score = (1 / words_cnt) * 1 
                    neg_score = (1 / words_cnt) * 1.5 
                    if pos_score < neg_score:polarity = 'Negative'
                    else:polarity = 'Positive'                     
            else:
                if max(word_freq(sent_polarity), key = word_freq(sent_polarity).get) == 'POS':polarity = 'Positive'
                else:polarity = 'Negative'
        kv_pair_2 = {'polarity': polarity}
        sent.update(kv_pair_2)

        if lookup == 1:
            sent_pol = [v if k == 'polarity' else None for k, v in sent.items()]
            lookup_index_wt_name.append(sent_pol)
        else:
            sent_pol = [v if k == 'polarity' else None for k, v in sent.items()]
            lookup_index_wo_name.append(sent_pol)
    lookup_index_wt_name = flatten(lookup_index_wt_name)
    lookup_index_wo_name = flatten(lookup_index_wo_name)
    while None in lookup_index_wt_name:
        lookup_index_wt_name.remove(None)
    while None in lookup_index_wo_name:
        lookup_index_wo_name.remove(None)

    # Count no of pos and neg sentences in a given news text
    sent_pol_list = []
    for sentences in documents['_sentences']:
        sent_pol_list.append([v if k == 'polarity' else None for k, v in sentences.items()])   
    sent_pol_count = word_freq(flatten(sent_pol_list))
    del sent_pol_count[None] # remove None Key from dict
    
    # Calculate final scores
    documents['#neg_sents'] = max([v if k == 'Negative' else 0 for k, v in sent_pol_count.items()])
    documents['#pos_sents'] = max([v if k == 'Positive' else 0 for k, v in sent_pol_count.items()])
    documents['#pol_sents_wt_name'] = word_freq(lookup_index_wt_name)
    documents['#pol_sents_wo_name'] = word_freq(lookup_index_wo_name)
    if documents['#pol_sents_wt_name']:
        pos_sents_wt_name = max([v if k == 'Positive' else 0 for k, v in documents['#pol_sents_wt_name'].items()])
        neg_sents_wt_name = max([v if k == 'Negative' else 0 for k, v in documents['#pol_sents_wt_name'].items()])
    else:pos_sents_wt_name, neg_sents_wt_name = 0, 0
    if documents['#pol_sents_wo_name']:
        pos_sents_wo_name = max([v if k == 'Positive' else 0 for k, v in documents['#pol_sents_wo_name'].items()])
        neg_sents_wo_name = max([v if k == 'Negative' else 0 for k, v in documents['#pol_sents_wo_name'].items()])
    else:pos_sents_wo_name, neg_sents_wo_name = 0, 0
    neutral_score = round((documents['#sents_no_model'] / documents['#sents_actual']) * 0.25 , 3)
    pos_score = round((pos_sents_wt_name / documents['#sents_actual'] * 1 * 1) + (pos_sents_wo_name / documents['#sents_actual'] * 1), 3)
    neg_score = round((neg_sents_wt_name / documents['#sents_actual'] * 1 * 1.5) + (neg_sents_wo_name / documents['#sents_actual'] * 1.5), 3)
    scores['Neutral'] = neutral_score
    scores['Positive'] = pos_score
    scores['Negative'] = neg_score
    documents['scores'] = scores  
    documents['_doc_polarity'] = max(documents['scores'], key = documents['scores'].get)
    return documents
