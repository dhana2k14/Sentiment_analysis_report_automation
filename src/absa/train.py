import argparse
import json
import os, re
import shutil
from pathlib import Path
from nltk import flatten
from azureml.core import Run
from azureml.core.model import Model
from sklearn.metrics import f1_score

# import NLP architect
from nlp_architect.models.absa.train.train import TrainSentiment
from nlp_architect.models.absa.inference.inference import SentimentInference

#inputs 
parser = argparse.ArgumentParser(description='ABSA Train')
parser.add_argument('--data_folder', type=str, dest='data_folder', help='data folder mounting point')
parser.add_argument('--asp_thresh', type=int, default=3)
parser.add_argument('--op_thresh', type=int, default=2)
parser.add_argument('--max_iter', type=int, default=3)

args = parser.parse_args()

from spacy.cli.download import download as spacy_download
from nlp_architect.utils.io import uncompress_file
from nlp_architect.models.absa import TRAIN_OUT

spacy_download('en')
GLOVE_ZIP = os.path.join(args.data_folder, 'clothing_data/glove.840B.300d.zip')
EMBEDDING_PATH = TRAIN_OUT / 'word_emb_unzipped' / 'glove.840B.300d.txt'
print("Embedding Data File Path...")
print(EMBEDDING_PATH)

uncompress_file(GLOVE_ZIP, Path(EMBEDDING_PATH).parent)

news_train = os.path.join(args.data_folder,'news_data/all_news_content_lang_en.txt.csv') # change here for input dataset
print("Input Dataset Location...")
print(news_train)

aspect_path = TRAIN_OUT / 'lexicons' / 'generated_aspect_lex.csv'
opinion_path = TRAIN_OUT / 'lexicons' / 'generated_opinion_lex_reranked.csv'

os.makedirs('outputs', exist_ok=True)

train = TrainSentiment(asp_thresh=args.asp_thresh, op_thresh=args.op_thresh, max_iter=args.max_iter)

try:
    opinion_lex, aspect_lex = train.run(data=news_train, out_dir = './outputs')
except:
    print("ValueError!")
    
# helper function 
def word_freq(word_list):
    """
    Return Polarity    
    """
    word_freq = [word_list.count(w) for w in word_list]
    return(dict(zip(word_list, word_freq)))

# helper function
def doc2label(doc):
    """
    Converts ABSA Inference Doc to Sentiment Labels  
    """
    documents = {}
    sentences_list = []
    line_json = json.loads(doc.json())
    text = line_json['_doc_text']
    sents = line_json['_sentences']
    events = []
    for i in range(len(sents)):
        for e in sents[i]['_events']:
            for ev in e:
                if ev['_type'] == 'OPINION':
                     events.append(ev)
    events = {d['_text']:d for d in events}.values() # get unique events
    label_list = [e['_polarity'] for e in list(events)] # extract polarity for count
    if label_list:
        if max(word_freq(label_list), key = word_freq(label_list).get) == 'POS':doc_sentiment = 'Positive'
        else:doc_sentiment = 'Negative'
    tokens = text.split()
    io = [[token, 'O'] for token in tokens]
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
    for id, sent in enumerate(sents):
        sent_polarity = {}
        s = text[sent['_start']:sent['_end'] + 1]
        s_tokens = s.split()
        s_io = [[tok, 'O'] for tok in s_tokens]
        for tok_id, tok in enumerate(s_tokens):
            for event in events:
                if event['_text'] == tok:
                    s_io[tok_id][1] = "<{}>".format(event['_polarity'])
        s_io = flatten(s_io)
        while 'O' in s_io:
            s_io.remove('O')
        sentence = " ".join([l for l in s_io])
        polarity = re.findall(r'<NEG>|<POS>', sentence)
        polarity = [re.sub(r'<|>', '', i) for i in polarity]
        if polarity:
            if max(word_freq(polarity), key = word_freq(polarity).get) == 'POS':polarity = 'Positive'
            else:polarity = 'Negative'
        sent_polarity['sentence ' + str(id + 1)] = sentence # sentence dict
        sent_polarity['polarity'] = polarity
        sentences_list.append(sent_polarity)
    documents['_news_text'] = text
    documents['_doc_polarity'] = doc_sentiment
    documents['_sentences'] = sentences_list
    return documents

inference = SentimentInference(aspect_path, opinion_path)
shutil.copyfile(aspect_path, './outputs/news_content_aspect.csv')
shutil.copyfile(opinion_path, './outputs/news_content_opinion.csv')
  
input_file_path = os.path.join(args.data_folder,'news_data/all_news_content.csv')
print(f'Aspect and Opinion lexicons files loaded from {aspect_path} and {opinion_path}')
print(f'Input file loaded from {input_file_path}')

inference = SentimentInference(aspect_path, opinion_path)

# Get Inference Results               
with open(input_file_path, 'r') as csv_file:
    lines = csv_file.readlines()
    for id, line in enumerate(lines):
        if line:
            sentiment_doc = inference.run(doc = line)
            if sentiment_doc != None:
                labels = doc2label(sentiment_doc)
                labels['sent_id'] = id + 1
                with open('./outputs/' + 'sentiment_labels_v1.json', 'a') as json_file:
                    json_file.write(json.dumps(labels))
                    json_file.write('\n')
