from pydantic import BaseModel
from typing import List,Optional

class MedicationExtraction(BaseModel):
    medication_name: str
    date_of_prescription: str
    dosage: str
    frequency: str
    duration: str
class MedicationLabReport(BaseModel):
    test_name: str
    date_of_test: str
    result: str
    normal_range: str
    doctor_comments: Optional[str]
class MedicationRecord(BaseModel):
    patient_id: str
    patient_name: str
    age: int
    medications: Optional[List[MedicationExtraction]]
    Lab_reports: Optional[List[MedicationLabReport]]
class Chunk(BaseModel):
    id: str
    patient_name: str
    patient_id: str
    doc_type: str
    content: str