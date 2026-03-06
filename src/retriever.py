import json
from typing import List
from src.schemas import Chunk
from src.clients import chroma_client,open_ai_client
from src.config import Config   
import numpy as np
from rank_bm25 import BM25Okapi


class Retriever:
    def __init__(self):
        self.chunks = self.load_chunks()
        tokenized_corpus = [chunk.content.lower().split(' ') for chunk in self.chunks]
        self.bm25 = BM25Okapi(tokenized_corpus)

    def load_chunks(self) -> List[Chunk]:
        with open("./data/chunked_data.json", "r") as f:
            data = json.load(f)
        return [Chunk(**item) for item in data]
    

    def hybrid_reteriver_rrf(self, query:str, chunks:List[Chunk], collection_name:str = Config.collection_name, top_k:int = 5):
        # BM25 Retrieval
        tokenized_query = query.lower().split()
        doc_scores_bm25 = self.bm25.get_scores(tokenized_query)
        doc_scores_bm25 = doc_scores_bm25.argsort()[::-1][:top_k]  # Sort in descending order
        
        # RRF Retrieval
        collection = chroma_client.get_collection(collection_name)
        query_embedding = open_ai_client.embeddings.create(input=[query], model=Config.EMBEDDING_MODEL).data[0].embedding
        results_rrf = collection.query(query_embeddings=[query_embedding], n_results=top_k, include=["documents","metadatas"])
        
        rrf={}
        result_ids = results_rrf['ids'][0]
        k = 60
        for i,doc_id in enumerate(result_ids):
            rrf[doc_id] = 1/(i+1+k)
        for i,idx in enumerate(doc_scores_bm25):
            doc_id = self.chunks[idx].id
            rrf[doc_id] = rrf.get(doc_id,0) + 1/(i+1+k)
        sorted_rrf = sorted(rrf.items(), key=lambda x: x[1], reverse=True)[:top_k]
        top_fused_ids = [item[0] for item in sorted_rrf]
        reterieved_docs = [chunk.content for chunk in self.chunks if chunk.id in top_fused_ids]
        return reterieved_docs
    

