import json
from datasets import Dataset
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
import pandas as pd
from src.config import Config
from src.clients import chroma_client,open_ai_client,rerank_model
from src.retriever import Retriever
from src.generator import Generator
import sys

class PipelineEvaluator:
    def __init__(self, dataset_path: str, generator_function, chunks, collection_name: str = "medical_data_collection", top_k: int = Config.RETRIEVAL_TOP_K):
        # 1. Store configuration and dependencies
        self.dataset_path = dataset_path
        self.generator_function = generator_function
        self.chunks = chunks
        self.collection_name = collection_name
        self.top_k = top_k
        
        # 2. Initialize the Grader LLMs
        self.grader_llm = ChatOpenAI(model=Config.GRADER_MODEL, temperature=0)
        self.grader_embeddings = OpenAIEmbeddings(model=Config.EMBEDDING_MODEL)  # Use the same embedding model for consistency
        
        # 3. Define the metrics to track
        self.metrics = [faithfulness, answer_relevancy, context_precision, context_recall]

    def _prepare_dataset(self) -> Dataset:
        """Private method: Runs the retriever and formats the data for Ragas."""
        with open(self.dataset_path, "r") as f:
            golden_data = json.load(f)

        ragas_data = {
            "question": [],
            "answer": [],
            "contexts": [],
            "ground_truth": [] 
        }
        
        print(f"Generating answers for {len(golden_data)} test cases...")
        for item in golden_data[: Config.EVAL_MAX_QUESTIONS]:
            query = item["question"]

            # Call the dynamically passed generator function
            llm_answer, retrieved_docs = self.generator_function(
                query=query, 
                chunks=self.chunks, 
                collection_name=self.collection_name, 
                top_k=self.top_k
            )
            
            ragas_data["question"].append(query)
            ragas_data["answer"].append(llm_answer)
            ragas_data["contexts"].append(retrieved_docs)
            ragas_data["ground_truth"].append(item["ground_truth"])

        return Dataset.from_dict(ragas_data)

    def run_evaluation(self) -> pd.DataFrame:
        """Public method: Executes the evaluation and returns the results."""
        dataset = self._prepare_dataset()
        
        print("Running Ragas evaluation metrics...")
        results = evaluate(
            dataset, 
            metrics=self.metrics, 
            llm=self.grader_llm, 
            embeddings=self.grader_embeddings
        )

        df = results.to_pandas()
        
        print("\n--- Individual Question Scores ---")
        print(df[["faithfulness", "answer_relevancy", "context_precision", "context_recall"]])
        
        print("\n--- Aggregate Pipeline Scores ---")
        print(results)
        
        return df

# ==========================================
# How to use it in your evaluate.py script:
# ==========================================
if __name__ == "__main__":
    # Assuming 'hybrid_reteriver_rerank', 'chunks', and 'bm25' are imported or defined above
    reteriver = Retriever()

    chunks = reteriver.load_chunks()  # Load chunks from your ingest_db module
    my_generator = Generator()
    evaluator = PipelineEvaluator(
        dataset_path="./data/golden_dataset.json",
        generator_function=my_generator.hybrid_reteriver_rerank,
        chunks=chunks,
        top_k=Config.RETRIEVAL_TOP_K
    )
    
    results_df = evaluator.run_evaluation()
    avg_relevancy = results_df['answer_relevancy'].mean()

    print("\n=====================================")
    print(f"       CI/CD QUALITY GATE            ")
    print("=====================================")
    print(f"Average Answer Relevancy: {avg_relevancy:.2f}")
    print("Target Threshold: 0.60")

    # 2. The Bouncer Logic
    if avg_relevancy < 0.60:
        print("\n❌ FAILED: Pipeline performance dropped below the 60% threshold.")
        print("GitHub Actions will now block this code.")
        sys.exit(1)  # Throws a fatal error to GitHub Actions
        
    else:
        print("\n✅ PASSED: Pipeline meets production standards!")
        sys.exit(0)  # Tells GitHub Actions everything is perfect