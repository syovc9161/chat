# pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
# pip3 install faiss-cpu
# pip3 install -U sentence-transformers

import torch
import numpy as np
# import faiss
import pandas as pd

from sentence_transformers import util

# from sentence_transformers import SentenceTransformer
# model = SentenceTransformer("Huffon/sentence-klue-roberta-base")
# torch.save(model, f'./model.pt')
device = "cuda" if torch.cuda.is_available() else "cpu"
embedder = torch.load("model.pt", map_location=device)

# em_func = lambda x : embedder.encode(x)
# em_array = em_func(data['insert_words'].to_numpy())
# np.save('em.npy', em_array)
em_array = np.load('em.npy')

data = pd.read_csv('menu.md')
data['insert_words'] = data[['eat_time', 'sector', 'store', 'menu_type', 'menu', 'remark']].apply(" ".join, axis=1)
# data.head()




from flask import Flask, render_template, request
import requests
import sys

application = Flask(__name__)

@application.route("/")
def hello():
    return "Hello goorm!" 

def format_response(resp, useCallback=False):
    data = {
            "version": "2.0",
            "useCallback": useCallback,
            "template": {
                "outputs": [
                    {
                        "simpleText": {
                            "text": resp
                        }
                    }
                ]
            }
        }
    return data

# executor = ThreadPoolExecutor(max_workers=1)

# @application.route('/chat-kakao', methods=['POST'])
# def chat_kakao():
#     print('request.json', request.json)
#     response_to_kakao = format_response('짜장면이나 먹어')
#     return response_to_kakao

executor = ThreadPoolExecutor(max_workers=1)

@application.route('/chat-kakao', methods=['POST'])
def chat_kakao():
    print("request.json:", request.json)
    request_message = request.json['userRequest']['utterance']
    print("request_message:", request_message)
    query_embedding = embedder.encode(request_message)

    docs = len(em_array)
    top_k = min(6, len(em_array))
    cos_scores = util.pytorch_cos_sim(query_embedding, em_array)[0]
    top_results = torch.topk(cos_scores, k=top_k)

    store_list = []
    for i, (score, idx) in enumerate(zip(top_results[0], top_results[1])):
        # print(f"{i+1}: {data.store.to_numpy()[idx]} {'(유사도: {:.4f})'.format(score)}")
        store_list.append(data.store.to_numpy()[idx])
    store_list = list(set(store_list))
    store_list = ', '.join(store_list)

    response_message = store_list

    print("response_message:", response_message)
    return format_response(response_message)

if __name__ == '__main__':
    application.run(host='0.0.0.0', port=5000)