# Medical RAG System Documentation

## Overview
The Medical RAG (Retrieval-Augmented Generation) system is designed to enhance medical information retrieval and generation for various applications.

## Architecture
- **Frontend**: User interface where queries are made.
- **Backend**: Processes queries and retrieves relevant medical documents using NLP techniques.
- **Database**: Stores documents and user queries for future reference.

## Installation Guide
1. Clone the repository:
   ```bash
   git clone https://github.com/levihtkas/Medical_RAG.git
   cd Medical_RAG
   ```
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Set up the database by running:
   ```bash
   python setup_db.py
   ```
4. Start the server:
   ```bash
   python app.py
   ```

## Usage Examples
- **Querying the System**: Send a GET request to `/api/query` with your question to retrieve answers from the database.
- **Adding New Documents**: Use the POST method to add medical documentation via `/api/add-doc`. Example:
   ```json
   {
     "title": "New Medical Document",
     "content": "Content of the medical document."
   }
   ```

## Key Features
- **NLP Integration**: Utilizes advanced Natural Language Processing for improved query understanding.
- **User-Friendly Interface**: Simplified user interface for easy navigation and query submission.
- **Data Searchability**: Allows users to search through a vast medical database effectively.