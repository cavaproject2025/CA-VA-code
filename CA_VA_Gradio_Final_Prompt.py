
"""
Collective Agreement Virtual Assistant with Gradio UI
A web-based Q&A interface for employment practices and collective agreements.

Features:
- Persistent vector store using ChromaDB (automatically saves/loads embeddings)
- Multi-source document loading (PDFs, Word docs, web pages)
- Azure OpenAI integration for embeddings and chat

Usage:
- Run normally: python CA_VA_Gradio_draft_refactored.py
- Force refresh embeddings: python CA_VA_Gradio_draft_refactored.py --refresh

The script will automatically create and persist a vector store on first run.
Subsequent runs will load the existing vector store for faster startup.
"""

import os
import time
import logging
import shutil
from typing import List, Tuple
from tqdm import tqdm
import argparse

# Core imports
from dotenv import load_dotenv
import gradio as gr

# LangChain imports
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from langchain_community.document_loaders import (
    DirectoryLoader, UnstructuredWordDocumentLoader,
    WebBaseLoader, PyPDFDirectoryLoader
)
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.tools import tool
from langchain.agents import create_openai_functions_agent, AgentExecutor
from langchain.schema import SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# Import MyCareersFuture scraper
from mycareerfutures import get_salary_info

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# =============================================================================
# CONFIGURATION
# =============================================================================

# Environment setup
os.environ["ALLOW_RESET"] = "TRUE"
load_dotenv('.env', override=True)

# Document sources configuration
WORD_DIRECTORY = "./wst"
WEB_URLS = [
    "https://www.mom.gov.sg/employment-practices/leave/shared-parental-leave"
    "https://sso.agc.gov.sg/act/ema1968"
    "https://sso.agc.gov.sg/Act/CDCSA2001"
    "https://sso.agc.gov.sg/Act/IRA1960"
    "https://www.mom.gov.sg/-/media/mom/documents/employment-practices/guidelines/tripartite-advisory-on-managing-excess-manpower-and-responsible-retrenchment.pdf"
]

# Persistence configuration
PERSIST_DIRECTORY = "./chromadb_persist"

# Processing configuration
CHUNK_SIZE = 500
CHUNK_OVERLAP = 0
BATCH_SIZE = 10
BATCH_DELAY = 2
RETRY_DELAY = 60

# Retrieval configuration
SEARCH_TYPE = "mmr"
K_DOCUMENTS = 5
MAX_SOURCE_DISPLAY = 3

# =============================================================================
# AZURE OPENAI SETUP
# =============================================================================

def initialize_azure_models() -> Tuple[AzureOpenAIEmbeddings, AzureChatOpenAI]:
    """Initialize Azure OpenAI embedding and chat models."""
    try:
        # Initialize embedding model
        embedding_model = AzureOpenAIEmbeddings(
            model=os.getenv('AZURE_OPENAI_EMBEDDING_DEPLOYMENT'),
            azure_endpoint=os.getenv('AZURE_OPENAI_EMBEDDING_ENDPOINT'),
            api_version=os.getenv('AZURE_OPENAI_EMBEDDING_API_VERSION'),
            api_key=os.getenv('AZURE_OPENAI_EMBEDDING_API_KEY')
        )

        # Initialize chat model
        llm = AzureChatOpenAI(
            azure_deployment=os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT_NAME"),
            api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
            temperature=0,
            max_tokens=None,
            timeout=None,
            max_retries=2,
        )
        
        logger.info("Azure OpenAI models initialized successfully")
        return embedding_model, llm
        
    except Exception as e:
        logger.error(f"Error initializing Azure OpenAI models: {e}")
        raise

# =============================================================================
# DOCUMENT LOADING
# =============================================================================

def load_word_documents(directory: str) -> List:
    """Load Word documents from specified directory and all subdirectories."""
    try:
        logger.info(f"Loading Word documents from {directory} (including subdirectories)")
        loader = DirectoryLoader(
            directory, 
            glob="**/*.docx",  # Recursive search pattern
            loader_cls=UnstructuredWordDocumentLoader,
            recursive=True
        )
        documents = loader.load()
        logger.info(f"Loaded {len(documents)} Word documents")
        return documents
    except Exception as e:
        logger.error(f"Error loading Word documents: {e}")
        return []

def load_web_documents(urls: List[str]) -> List:
    """Load documents from web URLs."""
    try:
        logger.info(f"Loading {len(urls)} web documents")
        loader = WebBaseLoader(urls)
        documents = loader.load()
        logger.info(f"Loaded {len(documents)} web documents")
        return documents
    except Exception as e:
        logger.error(f"Error loading web documents: {e}")
        return []

def load_all_documents() -> List:
    """Load all documents from configured sources."""
    logger.info("Loading documents from all sources...")
    
    # Load documents from different sources
    word_docs = load_word_documents(WORD_DIRECTORY)
    web_docs = load_web_documents(WEB_URLS)
    
    # Combine all documents
    all_documents = word_docs + web_docs
    logger.info(f"Total documents loaded: {len(all_documents)}")
    
    return all_documents

# =============================================================================
# VECTOR STORE CREATION
# =============================================================================

def create_document_splits(documents: List) -> List:
    """Split documents into chunks for processing."""
    logger.info("Splitting documents into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, 
        chunk_overlap=CHUNK_OVERLAP
    )
    splits = text_splitter.split_documents(documents)
    logger.info(f"Created {len(splits)} document chunks")
    return splits

def create_vector_store(document_splits: List, embedding_model: AzureOpenAIEmbeddings) -> Chroma:
    """Create and populate vector store with document chunks."""
    logger.info("Creating vector store...")
    
    # Create vector store with persistence
    vectorstore = Chroma(
        embedding_function=embedding_model,
        persist_directory=PERSIST_DIRECTORY
    )
    
    # Process documents in batches with progress tracking
    logger.info(f"Processing {len(document_splits)} documents in batches of {BATCH_SIZE}")
    
    for i in tqdm(range(0, len(document_splits), BATCH_SIZE), desc="Processing batches"):
        batch = document_splits[i:i+BATCH_SIZE]
        
        try:
            vectorstore.add_documents(documents=batch)
        except Exception as e:
            logger.warning(f"Error on batch {i}-{i+BATCH_SIZE}: {e}")
            logger.info(f"Retrying after {RETRY_DELAY} seconds...")
            time.sleep(RETRY_DELAY)
            
            try:
                vectorstore.add_documents(documents=batch)
                logger.info("Batch retry successful")
            except Exception as retry_error:
                logger.error(f"Batch retry failed: {retry_error}")
                continue
        
        # Small delay between successful batches
        time.sleep(BATCH_DELAY)
    
    logger.info("Vector store creation completed")
    return vectorstore

def load_existing_vector_store(embedding_model: AzureOpenAIEmbeddings) -> Chroma:
    """Load existing persisted vector store."""
    logger.info(f"Loading existing vector store from {PERSIST_DIRECTORY}")
    vectorstore = Chroma(
        embedding_function=embedding_model,
        persist_directory=PERSIST_DIRECTORY
    )
    return vectorstore

def check_vector_store_exists() -> bool:
    """Check if a persisted vector store already exists."""
    if not os.path.exists(PERSIST_DIRECTORY):
        return False
    
    # Check if the directory contains ChromaDB files
    chroma_files = [
        "chroma.sqlite3",
        "data_level0.bin"  # One of the typical ChromaDB files
    ]
    
    # Check if any ChromaDB-related files exist
    directory_contents = os.listdir(PERSIST_DIRECTORY) if os.path.exists(PERSIST_DIRECTORY) else []
    has_chroma_files = any(
        any(chroma_file in file for chroma_file in chroma_files) 
        for file in directory_contents
    ) or len(directory_contents) > 0
    
    return has_chroma_files

def delete_vector_store():
    """Delete the persisted vector store directory."""
    if os.path.exists(PERSIST_DIRECTORY):
        logger.info(f"Deleting existing vector store at {PERSIST_DIRECTORY}")
        shutil.rmtree(PERSIST_DIRECTORY)
        logger.info("Vector store deleted successfully")
    else:
        logger.info("No existing vector store to delete")

# =============================================================================
# Q&A SYSTEM SETUP
# =============================================================================

def setup_qa_system(vectorstore: Chroma, llm: AzureChatOpenAI):
    """Set up the agent-based Q&A system with tools."""
    logger.info("Setting up Q&A system with tools...")
    
    # Create retriever
    retriever = vectorstore.as_retriever(
        search_type=SEARCH_TYPE,
        search_kwargs={"k": K_DOCUMENTS}
    )
    
    # Create retrieval tool
    @tool
    def search_collective_agreements(query: str) -> str:
        """
        Search collective agreements, employment legislations, and MOM guidelines for information.
        
        Args:
            query: The search query to find relevant information
            
        Returns:
            Relevant information from collective agreements and employment guidelines with source attribution
        """
        try:
            docs = retriever.get_relevant_documents(query)
            if not docs:
                return "No relevant information found in collective agreements."
            
            # Combine the content from retrieved documents with source information
            content_with_sources = []
            sources_used = set()
            
            for doc in docs[:5]:  # Limit to top 5 documents
                # Extract source information
                source_name = "Unknown source"
                if hasattr(doc, 'metadata') and doc.metadata:
                    source = doc.metadata.get('source', '')
                    if source:
                        # Clean up file paths to show just filename
                        if '/' in source or '\\' in source:
                            source_name = source.split('/')[-1].split('\\')[-1]
                        else:
                            source_name = source
                
                content_with_sources.append(f"**Source: {source_name}**\n{doc.page_content}")
                sources_used.add(source_name)
            
            result = "\n\n---\n\n".join(content_with_sources)
            
            # Add summary of sources at the end
            if sources_used:
                source_list = ", ".join(sorted(sources_used))
                result += f"\n\n**Information retrieved from:** {source_list}"
            
            return result
            
        except Exception as e:
            logger.error(f"Error in retrieval tool: {e}")
            return f"Error searching collective agreements: {str(e)}"
    
    # Define available tools
    tools = [search_collective_agreements, get_salary_from_mycareersfuture]
    
    # System prompt for the agent
    system_prompt = """You are a specialized virtual assistant for collective agreements (CAs), employment practices, and progressive wage models in Singapore. 

You have access to two main tools:
1. search_collective_agreements - Search through collective agreements, MOM guidelines, and employment practices
2. get_salary_from_mycareersfuture - Get real-time salary information from MyCareersFuture website

Guidelines for responses:
- "CA" refers to collective agreements. For questions about salary ranges in the CA or employee benefits like annual leave or allowances, use search_collective_agreements.
- For questions about salary ranges of a position across CAs in a specific industry, refer to collective agreements in the specific industry.
- For questions about salary ranges in Hotel CAs, refer to collective agreements from Parkroyal Beach Rd, Raffles Hotel, RC Hotels, Ritz Carlton, Shangri-La Hotel loaded.
- For questions about salary ranges in Food Manufacturing CAs, refer to collective agreements from Asia Pacific Breweries, Lam Soon, Meiji Seika, Mondelez and Olam Cocoa loaded.
- For questions about salary ranges in CA, do not refer to the webpages loaded. Only use search_collective_agreements for questions related to CA or collective agreements.
- For questions about market salary, or current salary offerings, use get_salary_from_mycareersfuture only
- For questions about employment legislations or tripartite guidelines, use the webpages loaded
- ALWAYS clearly indicate the source of information in your responses
- DO NOT use get_salary_from_mycareersfuture if the question is about salary ranges in the CA
- Present quantitative data clearly with company names and specific amounts when available
- Always be accurate and base responses on the information retrieved from the tools

Source Attribution Requirements:
- When using search_collective_agreements, mention that information comes from "collective agreements" or specific document names
- When using get_salary_from_mycareersfuture, mention that information comes from "MyCareersFuture"
- Always preserve the source information provided by the tools

Response format:
- Start with the most relevant information for the user's question
- Clearly separate information from different sources
- For CA salary queries: Extract specific company names, amounts, and entitlements with document sources
- For market salary queries: Show ranges, averages, and sample sizes from MyCareersFuture with timestamp
- When combining sources, use clear headers or sections to distinguish between them"""

    # Create the agent prompt
    prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content=system_prompt),
        MessagesPlaceholder(variable_name="chat_history", optional=True),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad")
    ])
    
    # Create the agent
    agent = create_openai_functions_agent(llm, tools, prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
    
    logger.info("Q&A system with tools setup completed")
    return agent_executor

# ==========================================================================
# FEW SHOT PROMPT TEMPLATE
# ==========================================================================
from langchain.prompts.few_shot import FewShotPromptTemplate
from langchain.prompts.prompt import PromptTemplate

examples = [
    {
        "question": "What are the salary ranges for Housekeepers in Hotel CAs?",
        "answer": """
The salary ranges for Housekeepers in Hotel Collective Agreements are as follows:
1. Shangri-La Hotel
- Housekeeping (Room Attendant): $1,800 to $2,700
- Housekeeping (Clerk): $1,500 to $2,250
- Housekeeping (Linen)/(Uniform): $1,800 to $2,700
- Housekeeping (Public Area): $1,800 to $2,700 
- Housekeeping (Seamstress): $1,800 to $2,700


2. Raffles Hotel
- Senior Housekeeping Supervisor: $3,300 to $4,950
- Housekeeping Supervisor: $2,500 to $3,750
- Housekeeping Runner: $1,900 to $2,850
- Senior Housekeeping Attendant: $2,000 to $3,000
- Housekeeping Attendant: $1,700 to $2,550
- Housekeeping Coordinator: $2,400 to $3,600

3. RC Hotels
- Room Assistant (Modified): $1,450 tp $2,175
- Room Attendant: $1,650 to $2,475
- Senior Room Attendant: $1,850 to $2,775
- Housekeeping Team Leader: $2,100 to $3,150
- Leader: $2,300 to $3,450
- House Attendant: $1,550 to $2,325
- Senior House Attendant: $1,650 to $2,475

4. Ritz Carlton
- Housekeeping Room Attendant: $1,800 to $2,700
- Housekeeping Attendant: $1,800 to $2,700
""",
    },
    {
        "question": "What are the salary ranges for Drivers in Food Manufacturing CAs?",
        "answer": """
The salary ranges for Drivers in Food Manufacturing Collective Agreements are as follows:
1. Lam Soon
- Forklift Driver: $1,600 to $2,400

2. Meiji Seika: 
- Driver: $2,200 to $3,300 (Scheme A), $2,350 to $3,530 (Scheme B)
""",
    },
    {
        "question": "What are the market salary ranges for Chef De Partie in hotels on MyCareersFuture?",
        "answer": """
The market salary ranges for Chef De Partie in hotels on MyCareersFuture are as follow: 

1. Wyndham Singapore Hotel
- Cook/Chef De Partie: $2,380 to $2,550

2. South Beach International Hotel Management Pte Ltd
- Chef De Partie: $2,500 to $3,400

3. Hotel Miramar (Singapore) Limited
- Chef De Partie (Cold Kitchen): $2,200 to $3,400

4. Grand Mecure Roxy Hotel 
- Chef De Partie: $2,500 to $2,500

5. TPC Hotel Pte. Ltd
- Chef De Partie - Asian: $2,600 to $3,000

6. Beach Road Hotel (1886) Ltd
- Chef De Partie: $3,800 to $4,000
""",
    }
]

example_prompt = PromptTemplate(input_variables=["question", "answer"], template="Question: {question}\n{answer}")

prompt = FewShotPromptTemplate(
    examples=examples,
    example_prompt=example_prompt,
    suffix="Question: {input}",
    input_variables=["input"],
)

# =============================================================================
# Q&A PROCESSING
# =============================================================================

def ask_question(question: str) -> str:
    """
    Process user questions using the agent with tools.
    This function is called by the Gradio interface.
    """
    if not question.strip():
        return "Please enter a question."
    
    try:
        logger.info(f"Processing question: {question[:50]}...")
        
        # Use the agent to process the question
        response = qa_system.invoke({"input": question})
        
        # Extract the output from the agent response
        answer = response.get('output', 'No answer available.')
        
        logger.info("Question processed successfully")
        return answer
        
    except Exception as e:
        logger.error(f"Error processing question: {e}")
        return f"An error occurred while processing your question: {str(e)}"

# =============================================================================
# GRADIO UI SETUP
# =============================================================================

def create_gradio_interface():
    """Create and configure the Gradio web interface."""
    
    with gr.Blocks(title="Collective Agreement Virtual Assistant") as app:
        # Header
        gr.Markdown("""
        # Collective Agreement Virtual Assistant
        
        Ask questions about collective agreements, employment practices, and progressive wage models.
        The assistant can automatically access multiple sources:
        - Tripartite guidelines, Collective Agreements, and MOM resources
        - Real-time salary data from MyCareersFuture
        
        **Note:** Vector embeddings are automatically saved and reused for faster startup.
        
        **Smart Tools:** The assistant automatically decides whether to search collective agreements or fetch current salary data based on your question.
        
        ---
        """)
        
        # Main interface
        with gr.Row():
            with gr.Column(scale=1):
                # Question input
                question_input = gr.Textbox(
                    label="Your Question",
                    placeholder="e.g., What are the salary ranges for Housekeepers in Hotel CAs?",
                    lines=3
                )
                
                # Submit button
                submit_btn = gr.Button("Ask Question", variant="primary")
                
                # Tips
                gr.Markdown("**Tip:** Be specific in your questions for better results.")
                gr.Markdown("**Refresh:** Use `--refresh` flag to update embeddings with new documents.")
                gr.Markdown("**AI Tools:** Ask about salaries to get salary data from Collective Agreements or MyCareersFuture, or employment practices for CA information.")
                
            with gr.Column(scale=2):
                # Answer output
                answer_output = gr.Textbox(
                    label="Answer",
                    lines=15,
                    max_lines=20,
                    show_copy_button=True
                )
        
        # Example questions
        gr.Markdown("### Example Questions")
        gr.Examples(
            examples=[
                ["What are the salary ranges for Housekeepers across Hotels CA?"],
                ["What are the salary ranges for Technicians across Food Manufacturing CAs?"],
                ["What is the current entitlement for Shared Parental Leave?"],
                ["What are the progressive wage model wage requirements for restroom cleaners?"],
                ["What are the market salary ranges for Operators in food manufacturing companies on MyCareersFuture?"],
                ["What are the market salary ranges for Chef De Partie in hotels on MyCareersFuture?"],
                ["What is the current salary entitlement for work performed on rest days?"]
            ],
            inputs=question_input
        )
        
        # Event handlers
        submit_btn.click(
            fn=ask_question,
            inputs=question_input,
            outputs=answer_output,
            show_progress=True
        )
        
        question_input.submit(
            fn=ask_question,
            inputs=question_input,
            outputs=answer_output,
            show_progress=True
        )
    
    return app

# =============================================================================
# TOOLS DEFINITION
# =============================================================================

@tool
def get_salary_from_mycareersfuture(job_role: str) -> str:
    """
    Get salary information for a specific job role from MyCareersFuture website.
    
    Args:
        job_role: The job title to search for (e.g., 'Housekeepers', 'Operators', 'Technicians')
    
    Returns:
        Formatted salary information including ranges, average, and median salaries with source attribution
    """
    try:
        logger.info(f"Fetching salary data for: {job_role}")
        salary_data = get_salary_info(job_role, headless=True, timeout=30)
        
        if "error" in salary_data:
            return f"Error retrieving salary information for {job_role} from MyCareersFuture: {salary_data['error']}"
        
        # Get current timestamp for freshness indication
        from datetime import datetime
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M")
        
        response = f"**Source: MyCareersFuture (Retrieved: {current_time})**\n\n"
        response += f"Salary information for {job_role}:\n\n"
        
        if salary_data.get("salary_range"):
            response += "**Salary Ranges from Job Postings:**\n"
            for i, salary_range in enumerate(salary_data["salary_range"][:10], 1):
                response += f"- {salary_range}\n"
            response += "\n"
        
        if salary_data.get("average_salary"):
            response += f"**Average Salary:** {salary_data['average_salary']}\n"
        
        if salary_data.get("median_salary"):
            response += f"**Median Salary:** {salary_data['median_salary']}\n"
        
        if salary_data.get("sample_size") and salary_data["sample_size"] > 0:
            response += f"**Sample Size:** {salary_data['sample_size']} job postings\n"
        
        if not salary_data.get("salary_range") and not salary_data.get("average_salary"):
            response += f"No specific salary information found for {job_role} on MyCareersFuture."
        
        response += f"\n**Information retrieved from:** MyCareersFuture website"
        
        return response
        
    except Exception as e:
        logger.error(f"Error in MyCareersFuture tool: {e}")
        return f"Error retrieving salary information from MyCareersFuture: {str(e)}"

# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    """Main function to initialize and run the application."""
    try:
        # Parse command line arguments
        parser = argparse.ArgumentParser(description='Collective Agreement Virtual Assistant')
        parser.add_argument('--refresh', action='store_true', 
                          help='Force refresh of the vector store (re-embed all documents)')
        args = parser.parse_args()
        
        logger.info("=== Starting Collective Agreement Virtual Assistant ===")
        
        # Initialize Azure OpenAI models
        embedding_model, llm = initialize_azure_models()
        
        # Handle force refresh
        if args.refresh:
            logger.info("Force refresh requested. Deleting existing vector store...")
            delete_vector_store()
        
        # Check if vector store already exists
        if check_vector_store_exists() and not args.refresh:
            logger.info("Existing vector store found. Loading from disk...")
            vectorstore = load_existing_vector_store(embedding_model)
            logger.info("Vector store loaded successfully from disk")
        else:
            logger.info("No existing vector store found. Creating new one...")
            
            # Load all documents
            documents = load_all_documents()
            if not documents:
                raise ValueError("No documents were loaded. Please check your document directories and web URLs.")
            
            # Create document splits
            document_splits = create_document_splits(documents)
            
            # Create vector store
            vectorstore = create_vector_store(document_splits, embedding_model)
            logger.info("New vector store created and persisted to disk")
        
        # Setup Q&A system
        global qa_system  # Make qa_system available to ask_question function
        qa_system = setup_qa_system(vectorstore, llm)
        
        # Create and launch Gradio interface
        logger.info("Creating Gradio interface...")
        app = create_gradio_interface()
        
        logger.info("Launching web interface...")
        app.launch(share=True)
        
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        raise

if __name__ == "__main__":
    main()