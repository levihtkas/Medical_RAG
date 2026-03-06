from src.clients import open_ai_client,ollama_client
from schemas import MedicationRecord,MedicationExtraction,MedicationLabReport


def summarized_function(history_generation:str = "") -> str:
     prompt = f"""
      Here is the history of the data that has been generated till now : {history_generation},
    Could you please summarise this please don't add any extra text. Only write the summarization. I'll be passing it to another LLM, and I don't want it to make the copy or duplicate items. Just pick out the few specifics that you think might be copied to the next item and give it back. It should be very concise, and in the short format. Summarise it with low characters as much as possible. """
     
     response = ollama_client.chat.completions.create(model="gemma3:4b", messages=[{"role": "user", "content": prompt}])
     return response.choices[0].message.content  


def generate_medication_record(history_generation:str = "") -> MedicationRecord:
     prompt = f"""
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
     
     response = open_ai_client.chat.completions.parse(model="gpt-5-mini", messages=[{"role": "user", "content": prompt}],response_format=MedicationRecord)
     
     parsed_record = response.choices[0].message.parsed
     compressed_addition = summarized_function(parsed_record.model_dump_json())
     history_generation += f" | {compressed_addition}"
    
     print(f"Current Compressed History: {history_generation}\n------------------")
     return parsed_record,history_generation

