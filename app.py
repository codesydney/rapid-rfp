import sqlite3
import csv
import random
from rag_pipeline import process_website, Ollama, OllamaEmbeddings, FAISS, RetrievalQA, PromptTemplate, ConversationBufferMemory, rag_pipeline

# Configuration variables
CHUNK_SIZE = 300
CHUNK_OVERLAP = 50
TEMPERATURE = 0.4

# Function to read the CSV file and return the data as a list of dictionaries
def read_csv(file_path):
    data = []
    with open(file_path, mode='r', newline='', encoding='utf-8') as file:
        reader = csv.DictReader(file, delimiter='|')
        for row in reader:
            data.append(row)
    return data

def update_with_rag_answers(data, qa_chain, vectorstore):
    for row in data:
        query = row['Query']
        answer = rag_pipeline(query, qa_chain, vectorstore)
        
        # Clean the answer: remove newlines and replace double quotes with single quotes
        cleaned_answer = answer.replace('\n', ' ').replace('"', "'")
        
        # Estimate confidence based on the response
        confidence = estimate_confidence(answer)
        
        row['Answer'] = cleaned_answer
        row['Confidence'] = confidence  # Use the estimated confidence score
    return data

def write_csv(data, file_path):
    fieldnames = ['ID', 'Query', 'Confidence', 'Answer']
    with open(file_path, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
            delimiter='|',
            quoting=csv.QUOTE_MINIMAL,  # Only quote fields that contain special characters
            lineterminator='\n'  # Use a single newline character to terminate rows
        )
        writer.writeheader()
        writer.writerows(data)

# Function to insert or update data in the SQLite database
def insert_or_update_db(data, db_path):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    for row in data:
        # Check if the ID already exists in the database
        cursor.execute('SELECT * FROM rfp WHERE ID = ?', (row['ID'],))
        existing_record = cursor.fetchone()

        if existing_record:
            # Update the existing record
            cursor.execute('''
                UPDATE rfp
                SET Query = ?, Confidence = ?, Answer = ?
                WHERE ID = ?
            ''', (row['Query'], row['Confidence'], row['Answer'], row['ID']))
            print(f"Updated record with ID {row['ID']} in the database.")
        else:
            # Insert a new record
            cursor.execute('''
                INSERT INTO rfp (ID, Query, Confidence, Answer)
                VALUES (?, ?, ?, ?)
            ''', (row['ID'], row['Query'], row['Confidence'], row['Answer']))
            print(f"Inserted new record with ID {row['ID']} into the database.")

    # Commit the changes and close the connection
    conn.commit()
    conn.close()


def estimate_confidence(response):
    # Convert response to lowercase for case-insensitive checks
    response_lower = response.lower()

    # Low confidence if the response indicates uncertainty
    if "i don't have enough information" in response_lower or "i don't know" in response_lower:
        return random.randint(1, 3)  # Low confidence (1-3)

    # Medium confidence if the response is generic or vague
    if "this appears to be" in response_lower or "the provided text mentions" in response_lower:
        return random.randint(4, 6)  # Medium confidence (4-6)

    # High confidence if the response is specific and directly answers the query
    return random.randint(7, 9)  # High confidence (7-9)


# Main function
def main():
    # Path to the CSV file and SQLite database
    csv_file_path = 'rfp.csv'
    db_file_path = 'rfp.db'

    # Step 1: Read the CSV file
    data = read_csv(csv_file_path)
    print("CSV data read successfully.")

    # Step 2: Initialize RAG pipeline components
    url = input("\nPlease enter the URL of the website you want to query: ")
    print("Processing website content...")
    texts = process_website(url)
    
    if not texts:
        print("No content found on the website. Please try a different URL.")
        return
    
    print("Creating embeddings and vector store...")
    embeddings = OllamaEmbeddings(
        model="nomic-embed-text",
        show_progress=True
    )
    
    vectorstore = FAISS.from_documents(texts, embeddings)
    
    qa = RetrievalQA.from_chain_type(
        llm=Ollama(model="mannix/gemma2-2b:latest", temperature=TEMPERATURE),
        chain_type="stuff",
        retriever=vectorstore.as_retriever(),
        memory=ConversationBufferMemory(memory_key="chat_history", return_messages=True),
        chain_type_kwargs={"prompt": PromptTemplate(
            template="""Context: {context}

Question: {question}

Answer the question concisely based only on the given context. If the context doesn't contain relevant information, say "I don't have enough information to answer that question."

But, if the question is generic, then go ahead and answer the question, example what is a electric vehicle?

Answer:""",
            input_variables=["context", "question"]
        )}
    )

    # Step 3: Update with answers from the RAG model
    updated_data = update_with_rag_answers(data, qa, vectorstore)
    print("Updated data with answers from the RAG model.")

    # Step 4: Write updated data back to the CSV file
    write_csv(updated_data, csv_file_path)
    print("Updated data written back to the CSV file.")

    # Step 5: Insert or update data in the SQLite database
    insert_or_update_db(updated_data, db_file_path)
    print("Data insertion/update in the SQLite database completed.")

# Run the program
if __name__ == '__main__':
    main()