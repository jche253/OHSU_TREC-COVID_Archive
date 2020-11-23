#Title: TREC_COVID_Round1_OHSU.py
#Author: Jimmy Chen, School of Medicine, OHSU
#Description: Generate 1000 documents per topic in Round 1 TREC_COVID and get trec_eval metrics

import pandas as pd
import numpy as np
import torch
import os
from tqdm.auto import tqdm
import json
from pyserini.search import pysearch
from jnius import autoclass
import xml.etree.ElementTree as ET
import requests
import urllib.request
from trectools import misc, TrecRun, TrecQrel, procedures
from pyserini.analysis.pyanalysis import get_lucene_analyzer, Analyzer
import nltk
from nltk.corpus import stopwords
from datetime import datetime
from transformers import *
import pickle

curr_dir = os.getcwd()
Pyserini_files = os.path.join(curr_dir, 'Round3_data')
if (os.path.exists(Pyserini_files) == False):
    os.mkdir(Pyserini_files)
R3_fulltext = r'/home/exacloud/lustre1/chianglab/chenjim/CORD19/Pyserini_Lucene_CORD_index/lucene-index-cord19-full-text-2020-05-19'
R3_Queries = r'/home/exacloud/lustre1/chianglab/chenjim/CORD19/Round3_data/topics-rnd3-udel.xml'
with open(R3_Queries, 'r') as f:
    queries = f.readlines()
full_df = pd.read_csv(r'/home/exacloud/lustre1/chianglab/chenjim/CORD19/Round3_data/anserini.covid-r3.fusion2.txt', sep = ' ', header = None)
full_df.columns = ['topic', 'q0', 'docid', 'rank', 'score', 'qid']

tokenizer = AutoTokenizer.from_pretrained("mrm8488/scibert_scivocab-finetuned-CORD19")

model = AutoModelWithLMHead.from_pretrained("mrm8488/scibert_scivocab-finetuned-CORD19"])

def extract_scibert(text, tokenizer, model):
    text_ids = torch.tensor([tokenizer.encode(text, add_special_tokens=True, max_length = 512)])
    text_words = tokenizer.convert_ids_to_tokens(text_ids[0])[1:-1]

    n_chunks = int(np.ceil(float(text_ids.size(1))/510))
    states = []
    for ci in range(n_chunks):
        text_ids_ = text_ids[0, 1+ci*510:1+(ci+1)*510]
        text_ids_ = torch.cat([text_ids[0, 0].unsqueeze(0), text_ids_])
        if text_ids[0, -1] != text_ids[0, -1]:
            text_ids_ = torch.cat([text_ids_, text_ids[0,-1].unsqueeze(0)])

        with torch.no_grad():
            state = model(text_ids_.unsqueeze(0))[0]
            state = state[:, 1:-1, :]
        states.append(state)

    state = torch.cat(states, axis=1)
    return text_ids, text_words, state[0]

#Cross match similarity (uses cosine similarity matrix) to figure out match between query and each paragraph
def cross_match(state1, state2):
    state1 = state1 / torch.sqrt((state1 ** 2).sum(1, keepdims=True))
    state2 = state2 / torch.sqrt((state2 ** 2).sum(1, keepdims=True))
    sim = (state1.unsqueeze(1) * state2.unsqueeze(0)).sum(-1)
    return sim

#Use Pyjnius to extract full texts from collection
JString = autoclass('java.lang.String')
JIndexReaderUtils = autoclass('io.anserini.index.IndexReaderUtils')
reader = JIndexReaderUtils.getReader(R3_fulltext)

#Getting relevances for abstracts and paragraph
print('Searching paragraphs')
paragraph_rel_list = []
topic_list = []
for i in tqdm(range(len(full_df)), position = 0, leave = True):
    query = queries[int(full_df['topic'][i]) - 1]
    docid = full_df['docid'][i]
    rank = full_df['rank'][i]
    score = full_df['score'][i]

    #Only re-rank top 100 documents
    if (rank > 100):
        continue
    topic_list = full_df['topic'][i]
    # Fetch raw document contents by docid:
    rawdoc = JIndexReaderUtils.documentRaw(reader, JString(docid))
    doc_json = json.loads(rawdoc)

    #Create tokens for our query
    query_ids, query_words, query_state = extract_scibert(query, tokenizer, model)

    paragraph_states = []
    #Load document as title + abstract + paragraph
    try:
        abstract = str(doc_json['csv_metadata']['abstract'])
    except:
        abstract = ''
    try:
        title = str(doc_json['csv_metadata']['title'])
    except:
        title = ''

    abst_title = str(abstract) + ' ' + str(title)
    state = extract_scibert(abst_title, tokenizer, model)
    paragraph_states.append(state)

    #Some items don't have full manuscripts, load the body text if available
    sim_matrices = []
    try:
        for par in doc_json['body_text']:
            state = extract_scibert(par['text'], tokenizer, model)
            paragraph_states.append(state)

        for pid in range(len(paragraph_states)):
            #Skip empty paragraph states
            sim_score = cross_match(query_state, paragraph_states[pid][-1])
            if (sim_score.nelement() != 0):
                sim_matrices.append(sim_score)
    except:
        #Just process abstract
        sim_score = cross_match(query_state, paragraph_states[0][-1])
        sim_matrices.append(sim_score)
    #If there's any item in sim_matrices, the function should work
    try:
        paragraph_relevance = [torch.max(sim).item() for sim in sim_matrices]
        paragraph_rel_list.append(paragraph_relevance)
    except:
        paragraph_rel_list.append([])

#Write score list
print('Saving paragraph relevances')
with open(os.path.join(Pyserini_files,'paragraph_scores_anserini.pkl'), 'wb') as handle:
    pickle.dump(paragraph_rel_list, topic_list, handle)

#TODO: filter down to 1000 docs from BERT
