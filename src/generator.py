from typing import List
from src.schemas import Chunk
from src.clients import chroma_client,open_ai_client,rerank_model
from src.retriever import Retriever
from src.config import Config

class Generator:
    def __init__(self):
        self.reteriver = Retriever()

    @staticmethod
    def ask_llm(question: str, retrieved_docs: list):
    
        context = retrieved_docs
        prompt = f"""
        Context:
        {context}

        Question:
        {question}

        Answer:
        """

        response = open_ai_client.chat.completions.create(
            model="gpt-5-mini",
            messages=[
                {"role": "system", "content": Config.system_prompt},
                {"role": "user", "content": prompt}
            ],
        )

        return response.choices[0].message.content
    
    def hybrid_reteriver_rerank(self,query:str, chunks:List[Chunk], collection_name:str = "medical_data_collection", top_k:int = 5):
        # query = query_llm_rewrite(query)
        rrf_results= self.reteriver.hybrid_reteriver_rrf(query=query, chunks=chunks, collection_name=collection_name, top_k=top_k)
        # Reranking using cross-encoder
        rerank_scores = rerank_model.predict([(query, doc) for doc in rrf_results])
        print(f"Rerank Scores: {rerank_scores}")

        sorted_chunks = sorted(zip(rrf_results, rerank_scores), key=lambda x: x[1], reverse=True)
        top_reranked_docs = [item[0] for item in sorted_chunks[:10]]  # Get top 10 reranked documents

        response = Generator.ask_llm(question=query, retrieved_docs=top_reranked_docs)
        return response,top_reranked_docs
