import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer
from splade.splade.models.transformer_rep import Splade
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import json
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity
import time

class args:
    DATA_PATH = "../data"
    DOCUMENT_NAME = "DOC_NQ_first64.tsv"
    QUERY_TRAIN_NAME = "GTQ_NQ_train.tsv"
    QUERY_DEV_NAME = "GTQ_NQ_dev.tsv"
    TOPK_VALUES = [10,1000]  # Nonzero topk 값 리스트
    SHORT_INFERENCE = True
    SAVE_PATH = "../data/inference"
    PRED_SAVE_NAME = "SPLADE_Baseline.json"
    METRIC_SAVE_NAME = "SPLADE_Result.json"
    MODEL_TYPE_OR_DIR = "naver/splade_v2_max"

model = Splade(args.MODEL_TYPE_OR_DIR, agg="max")
model.eval()
tokenizer = AutoTokenizer.from_pretrained(args.MODEL_TYPE_OR_DIR)
reverse_voc = {v: k for k, v in tokenizer.vocab.items()}

def clean_text(text):
    text = text.replace("\n", "")
    text = text.replace("", "")
    text = text.replace('"', "")
    text = text.replace("'", "")
    return text.lower().strip()

document_corpus = pd.read_csv(f"{args.DATA_PATH}/{args.DOCUMENT_NAME}", sep="\t", dtype=str)
query_train_corpus = pd.read_csv(f"{args.DATA_PATH}/{args.QUERY_TRAIN_NAME}", sep="\t", dtype=str)
query_dev_corpus = pd.read_csv(f"{args.DATA_PATH}/{args.QUERY_DEV_NAME}", sep="\t", dtype=str)

document_corpus['query'] = document_corpus['query'].apply(clean_text)
query_train_corpus['query'] = query_train_corpus['query'].apply(clean_text)
query_dev_corpus['query'] = query_dev_corpus['query'].apply(clean_text)

document_corpus = dict(zip(document_corpus["oldid"], document_corpus['query'])) 
query_train_corpus = dict(zip(query_train_corpus["oldid"], query_train_corpus['query'])) 
query_dev_corpus = dict(zip(query_dev_corpus["oldid"], query_dev_corpus['query'])) 

index2oldid = {index: oldid for index, oldid in enumerate(document_corpus.keys())}

def return_rep_result(query=None,
                      doc=None,
                      model=model,
                      reverse_voc=reverse_voc,
                      topk=None):
    rep = "d_rep"
    _input = doc if doc != None else query

    with torch.no_grad():
        input_rep = model(d_kwargs=tokenizer(_input, return_tensors="pt"))[rep].squeeze()

    if topk:
        input_rep = input_rep.sort(descending=True)[0]
        threshold = input_rep[topk - 1].item()
        input_rep[input_rep < threshold] = 0
    
    col = torch.nonzero(input_rep).squeeze().cpu().tolist()
    weights = input_rep[col].cpu().tolist()
    d = {k: v for k, v in zip(col, weights)}
    sorted_d = {k: v for k, v in sorted(d.items(), key=lambda item: item[1], reverse=True)}
    bow_rep = [(reverse_voc[k], round(v, 2)) for k, v in sorted_d.items()]
    
    return input_rep, bow_rep, len(bow_rep)

def search_sparse_vectors(query_sparse, sparse_matrix, top_k=1000):
    similarities = cosine_similarity(query_sparse, sparse_matrix)
    top_k_indices = np.argsort(similarities[0])[::-1][:top_k]
    return top_k_indices, similarities[0][top_k_indices]

def recall_at_k(relevant_docs, predicted_docs, k):
    relevant_set = set(relevant_docs)
    predicted_at_k = set(predicted_docs[:k])
    recall = len(relevant_set & predicted_at_k) / min(len(relevant_set),k)
    return recall

k_values = [1, 5, 10, 50, 100, 200]
results = {topk: {k: [] for k in k_values} for topk in args.TOPK_VALUES}
time_measurements = {topk: [] for topk in args.TOPK_VALUES}

for topk in args.TOPK_VALUES:
    document_sparse_vector_dict = dict()
    for dev_index, (answer_docid, dev_query) in tqdm(enumerate(query_dev_corpus.items()), total=len(query_dev_corpus.items())):
        document = document_corpus.get(answer_docid)
        if document == None:
            continue
        doc_sparse_rep, doc_bow_rep, doc_length = return_rep_result(doc=document, topk=topk)
        document_sparse_vector_dict[answer_docid] = (doc_sparse_rep, doc_bow_rep)

    docid_list = [k for (k, v) in document_sparse_vector_dict.items()]
    doc_sparse_rep_list = [v[0] for (k, v) in document_sparse_vector_dict.items()]
    doc_sparse_rep_nparray = np.array([tensor.numpy() for tensor in doc_sparse_rep_list])
    doc_sparse_matrix = csr_matrix(doc_sparse_rep_nparray)

    for dev_index, (answer_docid, dev_query) in tqdm(enumerate(query_dev_corpus.items()), total=len(query_dev_corpus.items())):
        start_time = time.time()
        query_sparse_rep, query_bow_rep, query_length = return_rep_result(query=dev_query, topk=topk)
        query_sparse_rep = csr_matrix(query_sparse_rep)
        top_k_indices, scores = search_sparse_vectors(query_sparse_rep, doc_sparse_matrix)
        top_k_docid = [docid_list[idx] for idx in top_k_indices]
        end_time = time.time()

        time_measurements[topk].append(end_time - start_time)

        for k in k_values:
            recall = recall_at_k([answer_docid], top_k_docid, k)
            results[topk][k].append(recall)

        if args.SHORT_INFERENCE and dev_index == 99:
            break

average_recall = {topk: {k: sum(recalls) / len(recalls) for k, recalls in result.items()} for topk, result in results.items()}
average_time = {topk: sum(times) / len(times) for topk, times in time_measurements.items()}

for topk, recall in average_recall.items():
    print(f"\nTopK: {topk}")
    for k, rec in recall.items():
        print(f"Recall@{k}: {rec:.4f}")
    print(f"Average time per query: {average_time[topk]:.4f} seconds")

plt.figure(figsize=(16, 6))
for topk, recall in average_recall.items():
    plt.plot(list(recall.keys()), list(recall.values()), marker='o', linestyle='-', label=f'TopK={topk}')
plt.title('Recall@K for different TopK values')
plt.xlabel('K')
plt.ylabel('Recall')
plt.xticks(k_values)
plt.grid(True)
plt.legend()
plt.show()

plt.figure(figsize=(16, 6))
plt.plot(list(average_time.keys()), list(average_time.values()), marker='o', linestyle='-', color='r')
plt.title('Average Query Time for different TopK values')
plt.xlabel('TopK')
plt.ylabel('Time (seconds)')
plt.grid(True)
plt.show()
