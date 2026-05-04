# Medical RAG Documentation

## Introduction
The Medical RAG (Retrieval-Augmented Generation) system is designed to assist in medical queries by leveraging a large knowledge base and advanced machine learning models. This document provides comprehensive coverage of the system's architecture, file descriptions, installation instructions, and usage guide.

## System Architecture
The architecture of the Medical RAG system consists of the following major components:
- **Data Retrieval Module**: Responsible for fetching relevant medical data from databases.
- **Augmentation Module**: Enhances retrieved information using NLP techniques.
- **Response Generation Module**: Generates responses based on augmented data using AI models.

![System Architecture Diagram](link_to_diagram)

## File Descriptions
- **/data**: Directory containing datasets used for training and evaluation.
- **/models**: Pre-trained models used in the system.
- **/scripts**: Python scripts for data processing, training, and evaluation. 
- **/tests**: Unit tests for ensuring system functionality.

## Installation Instructions
1. Clone the repository:
   ```bash
   git clone https://github.com/levihtkas/Medical_RAG.git
   cd Medical_RAG
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Set up the environment variables needed for the application (see `.env.example` for reference).
4. Run the application:
   ```bash
   python app.py
   ```

## Usage Guide
- To make a query, send a POST request to the `/query` endpoint with the necessary parameters.
   ```json
   {
       "query": "Symptoms of Diabetes"
   }
   ```
- The system will return a JSON response containing the relevant information and advice.

## Conclusion
The Medical RAG system is a powerful tool for processing and responding to medical inquiries. Follow the instructions above to set up and use the system efficiently.