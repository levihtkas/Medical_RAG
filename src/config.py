import os
from dotenv import load_dotenv
from sentence_transformers import CrossEncoder

load_dotenv()

class Config:
    #API Keys
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    #RAG System Configurations
    VECTOR_STORE_PATH = os.getenv("VECTOR_STORE_PATH", "./data/vector_store")
    collection_name = os.getenv("COLLECTION_NAME", "medical_data_collection")
    GOLDEN_DATASET_PATH = "./data/golden_dataset.json"
    EMBEDDING_CACHE_PATH = os.getenv("EMBEDDING_CACHE_PATH", "./data/embedding_cache.json")

    #Model Configurations
    EMBEDDING_MODEL = "text-embedding-3-small"
    RERANK_MODEL = 'cross-encoder/ms-marco-MiniLM-L-6-v2'
    LLM_MODEL = "gpt-5-mini"
    GRADER_MODEL = os.getenv("GRADER_MODEL", "gpt-4o-mini")

    # Cost/quality knobs
    RETRIEVAL_TOP_K = int(os.getenv("RETRIEVAL_TOP_K", "5"))
    EVAL_MAX_QUESTIONS = int(os.getenv("EVAL_MAX_QUESTIONS", "5"))

    #Evaluation Threasholds
    MIN_ANSWER_RELEVANCY = 0.6
    MIN_CONTEXT_PRECISION = 0.5
    MIN_CONTEXT_RECALL = 0.5

    #Initialize Reranker
    reranker = CrossEncoder(RERANK_MODEL)

    #Prompts
    chunk_generation_prompt = """" 
      Hi, you are a synthetic data generator especially 
      medical data generator so I want you to give me a data 
      so that I can use it for medical data testing so I want it 
      in a two formats one is the prescription for and 
      other is the form of lab reports (make sure to include the doctor's comments in the lab report)
      so these two things I need and you will be provided 
      with history generation which is the data that has been generated till now you got to use that data to generate the next data and you have to make sure that the data is not repeated and it is unique and also make sure that the data is in the form of a json format and it should be parsable by pydantic model and also make sure that the data is realistic and it should be in the form of a medication record which includes patient id, patient name, age, medications and lab reports.yeah! 
      CRITICAL RULE: The data must be unique. 
    DO NOT USE THESE PREVIOUSLY GENERATED DETAILS: 
    {history_generation}
     """
    system_prompt ="""You are a clinical reasoning assistant.

    Answer the question strictly using the provided context.

    You also need to provide the citation for the information you are using to answer the question. The citation should be in the format of [Patient Name]_[Document Type]_[Index]. 
    Index get it from the index of the retrieved document in the retrieved_docs list. For example, if you are using the first document in the retrieved_docs list, the index will be 0, if you are using the second document, the index will be 1 and so on.
    And strictly, as much as possible, try to avoid extra statements like the conversation starters. Answer precisely to the point. Answer the user's question directly using only the provided context.
    You will be provided with several pieces of context. Some of it may be irrelevant to the specific question. You must identify and use ONLY the parts of the context that directly answer the prompt. Actively ignore the rest. Do not add conversational filler
    """
