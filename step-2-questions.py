import csv
import random
from langchain_ollama import OllamaLLM  # Updated import
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain_community.embeddings import OllamaEmbeddings

# Configuration variables
TEMPERATURE = 0.4

# Define the prompt template
PROMPT = PromptTemplate(
    template="""Context: {context}

Question: {question}

Answer the question concisely based only on the given context. If the context doesn't contain relevant information, say "I don't have enough information to answer that question."

But, if the question is generic, then go ahead and answer the question, example what is a electric vehicle?

Answer:""",
    input_variables=["context", "question"]
)

# Function to read the CSV file and return the data as a list of dictionaries
def read_csv(file_path):
    data = []
    with open(file_path, mode='r', newline='', encoding='utf-8') as file:
        reader = csv.DictReader(file, delimiter='|')
        for row in reader:
            data.append(row)
    return data

def load_vector_store(vector_store_path):
    # Load the vector store from disk
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    vectorstore = FAISS.load_local(
        vector_store_path,
        embeddings,
        allow_dangerous_deserialization=True  # Allow loading pickle files
    )
    print(f"Vector store loaded from {vector_store_path}")
    return vectorstore

def rag_pipeline(query, qa_chain, vectorstore):
    relevant_docs = vectorstore.similarity_search_with_score(query, k=3)
    
    print("\nTop 3 most relevant chunks:")
    context = ""
    for i, (doc, score) in enumerate(relevant_docs, 1):
        print(f"{i}. Relevance Score: {score:.4f}")
        print(f"   Content: {doc.page_content[:200]}...")
        print()
        context += doc.page_content + "\n\n"

    # Print the full prompt
    full_prompt = PROMPT.format(context=context, question=query)
    print("\nFull Prompt sent to the model:")
    print(full_prompt)
    print("\n" + "="*50 + "\n")

    response = qa_chain.invoke({"query": query})
    return response['result']

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

# Main function for answering questions
def main_questions():
    # Path to the CSV file with questions and vector store
    questions_csv_path = 'questions.csv'
    vector_store_path = 'rfp_vector_store'

    # Step 1: Read the CSV file with questions
    data = read_csv(questions_csv_path)
    print("Questions CSV data read successfully.")

    # Step 2: Load the pre-processed vector store
    vectorstore = load_vector_store(vector_store_path)
    
    # Step 3: Set up the RAG pipeline
    qa = RetrievalQA.from_chain_type(
        llm=OllamaLLM(model="mannix/gemma2-2b:latest", temperature=TEMPERATURE),  # Updated to OllamaLLM
        chain_type="stuff",
        retriever=vectorstore.as_retriever(),
        memory=ConversationBufferMemory(memory_key="chat_history", return_messages=True),
        chain_type_kwargs={"prompt": PROMPT}  # Use the defined PROMPT
    )

    # Step 4: Update with answers from the RAG model
    updated_data = update_with_rag_answers(data, qa, vectorstore)
    print("Updated data with answers from the RAG model.")

    # Step 5: Write updated data back to the CSV file
    write_csv(updated_data, 'answers.csv')
    print("Updated data written back to the CSV file.")

# Run the question-answering process
if __name__ == '__main__':
    main_questions()