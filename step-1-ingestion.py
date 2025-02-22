import os
import csv
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.docstore.document import Document

# Configuration variables
CHUNK_SIZE = 300
CHUNK_OVERLAP = 50

# Function to read the CSV file and return the data as a list of dictionaries
def read_csv(file_path):
    data = []
    with open(file_path, mode='r', newline='', encoding='utf-8') as file:
        reader = csv.DictReader(file, delimiter='|')
        for row in reader:
            data.append(row)
    return data

def process_csv_content(data):
    # Combine all the text content from the CSV into a single string
    combined_text = "\n".join([row['Query'] for row in data])
    
    # Split the combined text into chunks
    text_splitter = CharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    texts = text_splitter.split_text(combined_text)
    
    # Convert the text chunks into document format
    documents = [Document(page_content=text) for text in texts]
    
    print(f"Number of text chunks after splitting: {len(documents)}")
    return documents

def save_vector_store(documents, vector_store_path):
    # Create embeddings and vector store
    embeddings = OllamaEmbeddings(
        model="nomic-embed-text",
        show_progress=True
    )
    
    vectorstore = FAISS.from_documents(documents, embeddings)
    
    # Save the vector store to disk
    vectorstore.save_local(vector_store_path)
    print(f"Vector store saved to {vector_store_path}")

# Main function for ingestion
def main_ingestion():
    # Path to the CSV file and vector store
    csv_file_path = 'rfp_main.csv'
    vector_store_path = 'rfp_vector_store'

    # Step 1: Read the CSV file
    data = read_csv(csv_file_path)
    print("CSV data read successfully.")

    # Step 2: Process CSV content for the RAG model
    print("Processing CSV content...")
    documents = process_csv_content(data)
    
    if not documents:
        print("No content found in the CSV file.")
        return
    
    # Step 3: Create and save the vector store
    save_vector_store(documents, vector_store_path)
    print("Ingestion completed.")

# Run the ingestion process
if __name__ == '__main__':
    main_ingestion()