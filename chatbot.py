import os
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Load and process the resume PDF
def load_resume(pdf_path):
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=400,  # Optimized chunk size for resume content
        chunk_overlap=50  # Reduced overlap for efficiency
    )
    chunks = text_splitter.split_documents(documents)
    
    return chunks

# Create vector store from document chunks
def create_vector_store(chunks):
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vector_store = Chroma.from_documents(chunks, embeddings)
    
    return vector_store

# Define the resume chatbot prompt with enhanced instructions
RESUME_QA_TEMPLATE = """
You are an AI assistant with access to a resume. Answer questions about the person's qualifications, experience, skills, education, projects, and other relevant details from their resume.

Resume Context:
{context}

Conversation History:
{chat_history}

Question: {question}

Guidelines for Response:
1. Provide concise and accurate answers based solely on the resume.
2. Avoid making assumptions or inferring information not explicitly stated.
3. If asked about something not in the resume, state that the information is unavailable.
4. Format responses professionally and clearly.

Respond only using the information provided in the resume context above.
"""

# Initialize the resume chatbot
def create_resume_chatbot(vector_store):
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        max_token_limit=1000  # Limit conversation history for efficiency
    )
    
    qa_prompt = PromptTemplate.from_template(RESUME_QA_TEMPLATE)
    
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash",
        google_api_key=os.environ["GOOGLE_API_KEY"],
        temperature=0.1  # Lower temperature for factual responses
    )
    
    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 5}  # Retrieve more relevant chunks for better answers
    )
    
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        combine_docs_chain_kwargs={"prompt": qa_prompt}
    )
    
    return qa_chain

# Main function to run the chatbot
def main():
    print("Resume Chatbot Initializing...")
    
    # Replace with your resume file path
    pdf_path = "Resume_Ayan.pdf"
    
    if not os.path.exists(pdf_path):
        print(f"Error: File {pdf_path} not found.")
        return
    
    print("Processing resume...")
    chunks = load_resume(pdf_path)
    print(f"Resume processed into {len(chunks)} chunks.")
    
    print("Creating vector database...")
    vector_store = create_vector_store(chunks)
    
    print("Initializing chatbot...")
    chatbot = create_resume_chatbot(vector_store)
    
    print("\nResume Chatbot Ready! Ask questions about the resume (type 'exit' to end)")
    
    while True:
        query = input("\nQuestion: ")
        
        if query.lower() == "exit":
            print("Exiting Resume Chatbot.")
            break
            
        try:
            response = chatbot({"question": query})
            print(f"\nAnswer: {response['answer']}")
        except Exception as e:
            print(f"Error: {e}")
            print("Ensure your Google API key is correctly set in the .env file.")

if __name__ == "__main__":
    main()
