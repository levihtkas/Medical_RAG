from typing import List
from src.clients import open_ai_client
from src.schemas import MedicationRecord,MedicationExtraction,MedicationLabReport,Chunk
from src.config import Config
import json
import hashlib
from src.clients import chroma_client 
class DocumentIngestor:
    def load_documents(self) -> List[MedicationRecord]:
        with open("./data/synthetic_medication_data.json", "r") as f:
            data = json.load(f)
        return data
    
    @staticmethod
    def generate_index(patient_id:str, doc_type:str, content:str) -> str:
        unique_string = f"{patient_id}_{doc_type}_{content}"
        return f"{patient_id}_{hashlib.sha256(unique_string.encode()).hexdigest()[:16]}"  # Generate a unique index based on patient_id, doc_type and content"

    def load_chunks(self) -> List[str]:
        data = self.load_documents()
        chunks = []
        for record in data:
            patient_name = record["patient_name"]
            patient_id = record["patient_id"]
            medications = record.get("medications", [])
            lab_reports = record.get("Lab_reports", [])
            
            for med in medications:
                if med is None:
                    continue
                chunk_content = f"Patient id: {patient_id}, Patient name: {patient_name}, Medication Name: {med['medication_name']}, Date of Prescription: {med['date_of_prescription']}, Dosage: {med['dosage']}, Frequency: {med['frequency']}, Duration: {med['duration']}"
                chunk_id = DocumentIngestor.generate_index(patient_id=patient_id, doc_type="prescription", content=chunk_content)
                chunks.append(Chunk(id=chunk_id, patient_name=patient_name, patient_id=patient_id, doc_type="prescription", content=chunk_content))
            
            for report in lab_reports:
                if report is None:
                    continue
                chunk_content = f"Patient id: {patient_id}, Patient name: {patient_name}, Test Name: {report['test_name']}, Date of Test: {report['date_of_test']}, Result: {report['result']}, Normal Range: {report['normal_range']}, Doctor Comments: {report.get('doctor_comments', 'N/A')}"
                chunk_id = DocumentIngestor.generate_index(patient_id=patient_id, doc_type="lab_report", content=chunk_content)
                chunks.append(Chunk(id=chunk_id, patient_name=patient_name, patient_id=patient_id, doc_type="lab_report", content=chunk_content))
        return chunks
    
    def ingest_to_vector_store(self, chunks:List[Chunk]):
        existing_collections = [col.name for col in chroma_client.list_collections()]
        
        if Config.collection_name in existing_collections:
            chroma_client.delete_collection(Config.collection_name)
        collection = chroma_client.get_or_create_collection(name=Config.collection_name)
        embeddings = open_ai_client.embeddings.create(input=[chunk.content for chunk in chunks], model=Config.EMBEDDING_MODEL)
        vector_embeddings = [embedding.embedding for embedding in embeddings.data]
            
        collection.add(
                ids=[chunk.id for chunk in chunks],
                metadatas=[{"patient_name": chunk.patient_name,"patient_id": chunk.patient_id,"date_of_prescription": chunk.content.split("Date of Prescription: ")[1].split(",")[0] if "Date of Prescription: " in chunk.content else "N/A","date_of_lab_test": chunk.content.split("Date of Test: ")[1].split(",")[0] if "Date of Test: " in chunk.content else "N/A","doc_type": chunk.doc_type} for chunk in chunks],
                documents=[chunk.content for chunk in chunks],
                embeddings=vector_embeddings
        )
        print(f"Ingested {len(chunks)} chunks into the vector store.")
if __name__ == "__main__":
    ingestor = DocumentIngestor()
    chunks = ingestor.load_chunks()
    ingestor.ingest_to_vector_store(chunks)



    
