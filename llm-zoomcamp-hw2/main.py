
from embedder import Embedder
from gitsource import GithubRepositoryDataReader

from gitsource import chunk_documents
import numpy as np

from minsearch import VectorSearch

from minsearch import Index


reader = GithubRepositoryDataReader(
    repo_owner="DataTalksClub",
    repo_name="llm-zoomcamp",
    commit_id="8c1834d",
    allowed_extensions={"md"},
    filename_filter=lambda path: "/lessons/" in path,
)

documents = [file.parse() for file in reader.read()]


def main():
    print("Hello from llm-zoomcamp-hw2!")
    
    #Q1
    print("\nQ1")
    embedder = Embedder()

    query = "How does approximate nearest neighbor search work?"
    v = embedder.encode(query)

    print(f"First vector value: {v[0]:.2f}")

    #Q2
    print("\nQ2")

    target_file = "02-vector-search/lessons/07-sqlitesearch-vector.md"
    target_doc = next(doc for doc in documents if doc['filename'] == target_file)

    doc_vector = embedder.encode(target_doc['content'])

    similarity = v.dot(doc_vector)

    print(f"Cosine similarity: {similarity:.2f}")

    #Q3
    print("\nQ3")
    chunks = chunk_documents(documents, size=2000, step=1000)

    chunk_texts = [chunk['content'] for chunk in chunks]
    chunk_vectors = embedder.encode_batch(chunk_texts)

    X = np.array(chunk_vectors)

    scores = X.dot(v)

    highest_index = np.argmax(scores)
    highest_chunk = chunks[highest_index]
    print(f"Highest scoring chunk filename: {highest_chunk['filename']}")

    #Q4
    print("\nQ4")
    vector_index = VectorSearch(
        keyword_fields=["filename"],  
        numeric_fields=["start"]
    )

    vector_documents = []
    for i, chunk in enumerate(chunks):
        vector_documents.append({
            'vector': chunk_vectors[i],
            'filename': chunk['filename'],
            'start': chunk['start'],
            'content': chunk['content']
        })

    vector_index.fit(X, vector_documents)

    query_q4 = "What metric do we use to evaluate a search engine?"
    query_vector_q4 = embedder.encode(query_q4)
    results_q4 = vector_index.search(query_vector_q4, num_results=5)

    print(f"Filename of the first result: {results_q4[0]['filename']}")

    #Q5
    print("\nQ5")
    keyword_index = Index(
        text_fields=["content"] 
    )

    keyword_documents = []
    for chunk in chunks:
        keyword_documents.append({
            'content': chunk['content'],
            'filename': chunk['filename'],
            'start': chunk['start']
        })

    keyword_index.fit(keyword_documents)

    query_q5 = "How do I store vectors in PostgreSQL?"

    keyword_results_q5 = keyword_index.search(query_q5, num_results=5)
    keyword_filenames = {result['filename'] for result in keyword_results_q5}

    query_vector_q5 = embedder.encode(query_q5)
    vector_results_q5 = vector_index.search(query_vector_q5, num_results=5)
    vector_filenames = {result['filename'] for result in vector_results_q5}

    in_vector_not_keyword = vector_filenames - keyword_filenames

    in_keyword_not_vector = keyword_filenames - vector_filenames

    print("Keyword Search:")

    for i, result in enumerate(keyword_results_q5, 1):
        print(f"  {i}. {result['filename']} ")

    print("\nVector Search:")
    for i, result in enumerate(vector_results_q5, 1):
        print(f"  {i}. {result['filename']} ")

    if in_vector_not_keyword:
        print(f"\nIn Vector Results and not in Keyword Results:\n {list(in_vector_not_keyword)[0]}")
    else:
        print("\nNo files found in vector that weren't in keyword results.")

    def rrf(result_lists, k=60, num_results=5):
    
        scores = {}
        docs = {}
        
        for results in result_lists:
            for rank, doc in enumerate(results):
                
                key = (doc.get('filename', doc.get('file', '')), 
                    doc.get('start', 0))
                
                rrf_score = 1 / (k + rank)
                
                scores[key] = scores.get(key, 0) + rrf_score
                
                if key not in docs:
                    docs[key] = doc
        
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        
        results = []
        for key, score in ranked[:num_results]:
            doc = docs[key].copy()
            doc['_rrf_score'] = score 
            results.append(doc)
        
        return results
    
    #Q6
    print("\nQ6")
    query_q6 = "How do I give the model access to tools?"

    keyword_results_q6 = keyword_index.search(query_q6, num_results=10)

    query_vector_q6 = embedder.encode(query_q6)
    vector_results_q6 = vector_index.search(query_vector_q6, num_results=10)

    combined_results = rrf([keyword_results_q6, vector_results_q6], k=60, num_results=5)

    print("Keyword Search:")
    for i, result in enumerate(keyword_results_q6[:5], 1):
        print(f" {i}. {result['filename']}")

    print("\nVector Search:")
    for i, result in enumerate(vector_results_q6[:5], 1):
        print(f" {i}. {result['filename']} ")

    print(f"The file that shows up in the vector results but not in the text results:\n{combined_results[0]['filename']}")

if __name__ == "__main__":
    main()
