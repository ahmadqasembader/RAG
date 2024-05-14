import os
import re
import glob
import faiss
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer

# preprocess the content of the files
def preprocess_text(text):
    # Remove new lines
    text = text.replace('\n', ' ')
    # Remove commas
    text = text.replace(',', '')
    # Remove extra white spaces
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def load_docs():
    #preapre the folder path
    folderPath = './Standards Docs/'
    file_paths = glob.glob(os.path.join(folderPath, '*'))
    file_contents = {}

    # loop over all the files in the folder
    for file_path in file_paths:
        with open(file_path, 'r') as file:
            file_name = os.path.basename(file_path).replace('.', '_')
            raw_content = file.read()
            processed_content = preprocess_text(raw_content)
            file_contents[file_name] = processed_content

    # for key, value in file_contents.items():
    #     print(key, value)
    return file_contents



model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
temp_docs = load_docs()
documents = list(temp_docs.items())

def docs_embeddings_and_indexing() -> list:
    embeddings = model.encode(documents, convert_to_tensor=True)
    embedding_matrix = embeddings.cpu().numpy()
    index = faiss.IndexFlatL2(embedding_matrix.shape[1])
    index.add(embedding_matrix)
    faiss.write_index(index, './index_document.faiss')

    with open('index_document.pkl', 'wb') as file:
        pickle.dump(documents, file)

    query = "RTP Control Protocol"
    query_emb = model.encode([query], convert_to_tensor=True).cpu().numpy()

    _ , I = index.search(query_emb, k=3)

    top_docs = [documents[i] for i in I[0]]
    with open('rag.txt', 'w') as file:
        file.write(f'{top_docs}')
    # print(top_docs)

    return top_docs

docs_embeddings_and_indexing()