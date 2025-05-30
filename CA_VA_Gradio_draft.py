# ========== COLLECTIVE AGREEMENT VIRTUAL ASSISTANT WITH GRADIO UI ==========
# This script creates a web-based Q&A interface using Gradio
# CHANGES MADE: Added Gradio UI components to transform console script into web app

import os
from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
# ========== UPDATED IMPORTS (FIX FOR LANGCHAIN DEPRECATION) ==========
# CHANGE: Updated to use langchain_community for deprecated imports
from langchain_community.document_loaders import DirectoryLoader
from langchain_community.document_loaders import UnstructuredWordDocumentLoader
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
# ========== NEW IMPORT FOR UI ==========
# CHANGE 1: Added Gradio import to enable web-based user interface. Remember to pip install gradio in the terminal within the virtual/conda environment before running the script.
import gradio as gr

import time
from tqdm import tqdm

os.environ["ALLOW_RESET"] = "TRUE"

load_dotenv('.env', override=True)

# # Explicitly set the deployment name to match your Azure deployment
embedding_model = AzureOpenAIEmbeddings(
    model=os.getenv('AZURE_OPENAI_EMBEDDING_DEPLOYMENT'),
    azure_endpoint=os.getenv('AZURE_OPENAI_EMBEDDING_ENDPOINT'),
    api_version=os.getenv('AZURE_OPENAI_EMBEDDING_API_VERSION'),
    api_key=os.getenv('AZURE_OPENAI_EMBEDDING_API_KEY')
    )

# Instantiate an AzureOpenAI LLM:
llm = AzureChatOpenAI(
    azure_deployment=os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT_NAME"),
    api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
)

#load web page from MOM PWM URLS:
loader_multiple_pages = WebBaseLoader(["https://www.mom.gov.sg/employment-practices/progressive-wage-model/cleaning-sector",
                                      "https://www.mom.gov.sg/employment-practices/progressive-wage-model/retail-sector",
                                      "https://www.mom.gov.sg/employment-practices/progressive-wage-model/food-services-sector",
                                      "https://www.mom.gov.sg/employment-practices/progressive-wage-model/occupational-pws-for-administrators-and-drivers",
                                      "https://www.mom.gov.sg/passes-and-permits/work-permit-for-foreign-worker/sector-specific-rules/non-traditional-sources-occupation-list"
                                      ])

web_documents = loader_multiple_pages.load()

## List of all pdf files (Tripartite Guidelines and)
loader = PyPDFDirectoryLoader("./Tripartite Guidelines")
pdf_documents = loader.load()


folder_path="./wst"
txt_loader = DirectoryLoader(folder_path, glob="*.docx", loader_cls=UnstructuredWordDocumentLoader)
txt_documents = txt_loader.load()

# Split content into chunks.

# Instantiate the splitter:
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
# Do the splitting:
splits = text_splitter.split_documents(pdf_documents+txt_documents+web_documents)


# Create an empty Chroma DB first
vectorstore = Chroma(
    embedding_function=embedding_model
)

# Process in smaller batches
batch_size = 10
for i in tqdm(range(0, len(splits), batch_size)):
    batch = splits[i:i+batch_size]
    try:
        vectorstore.add_documents(documents=batch)
    except Exception as e:
        print(f"Error on batch {i}-{i+batch_size}: {e}")
        # Wait before retrying
        time.sleep(60)
        vectorstore.add_documents(documents=batch)
    # Small delay between successful batches
    time.sleep(2)

# Instantiate a retriever based on the vector store:
retriever = vectorstore.as_retriever() 

# ========== MINOR MODIFICATION FOR UI ==========
# Set up a retrieval chain:
# CHANGE 2: Kept return_source_documents=True to show source citations in UI
qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 5}
    ),
    return_source_documents=True  # Essential for source citations in web UI
)

# ========== MAJOR CHANGE: REPLACED HARDCODED EXECUTION WITH INTERACTIVE FUNCTION ==========
# ORIGINAL CODE REMOVED:
# query = "In the event that retrenchment is inevitable, what should companies do?"
# completion = qa.invoke({"query": query})
# print(completion['result'])

# CHANGE 3: Created interactive function to handle user questions
def ask_question(question):
    """
    Function to process user questions and return answers with sources
    
    CHANGES MADE FOR GRADIO UI:
    - Takes user input instead of hard-coded query
    - Validates input (checks for empty questions)  
    - Formats response with source citations for better web display
    - Includes error handling for robust UI experience
    - Returns formatted string suitable for Gradio text display
    """
    if not question.strip():
        return "Please enter a question."
    
    try:
        # Run the retrieval chain and get a response from the LLM
        completion = qa.invoke({"query": question})
        
        # Format the response with sources
        answer = completion['result']
        sources = completion.get('source_documents', [])
        
        # NEW FEATURE: Add source citations to answer (not in original script)
        if sources:
            source_info = "\n\n**Sources:**\n"
            for i, doc in enumerate(sources[:3], 1):  # Show top 3 sources
                source_info += f"{i}. {doc.metadata.get('source', 'Unknown source')}\n"
            answer += source_info
        
        return answer
    except Exception as e:
        return f"An error occurred: {str(e)}"

# ========== CHANGE 4: COMPLETE GRADIO WEB INTERFACE (ENTIRELY NEW) ==========
# This entire section replaces the simple print statement with a full web UI
with gr.Blocks(title="Collective Agreement Virtual Assistant") as app:
    # Professional header with description
    gr.Markdown(
        """
        # 🤖 Collective Agreement Virtual Assistant
        
        Ask questions about collective agreements, employment practices, and progressive wage models.
        The assistant draws information from tripartite guidelines, WST documents, and MOM resources.
        """
    )
    
    # Two-column layout for clean interface
    with gr.Row():
        with gr.Column():
            # Question input area with helpful placeholder
            question_input = gr.Textbox(
                label="Your Question",
                placeholder="e.g., In the event that retrenchment is inevitable, what should companies do?",
                lines=3
            )
            # Primary submit button
            submit_btn = gr.Button("Ask Question", variant="primary")
            
        with gr.Column():
            # Answer display area with appropriate sizing
            answer_output = gr.Textbox(
                label="Answer",
                lines=15,
                max_lines=20
            )
    
    # Example questions to guide users
    gr.Examples(
        examples=[
            ["In the event that retrenchment is inevitable, what should companies do?"],
            ["What are the progressive wage model requirements for the cleaning sector?"],
            ["What assistance should companies provide during retrenchment?"],
            ["What are the key considerations for implementing no-pay leave?"]
        ],
        inputs=question_input
    )
    
    # Connect the interactive function to UI elements
    # Two ways to submit questions for better user experience
    submit_btn.click(
        fn=ask_question,
        inputs=question_input,
        outputs=answer_output
    )
    
    # Allow Enter key submission
    question_input.submit(
        fn=ask_question,
        inputs=question_input,
        outputs=answer_output
    )

# ========== CHANGE 5: LAUNCH WEB INTERFACE INSTEAD OF ONE-TIME EXECUTION ==========
# REPLACED: Single execution with print output
# NEW: Persistent web server that stays active for continuous use
if __name__ == "__main__":
    # share=True creates a public shareable link
    # Server runs continuously until stopped
    app.launch(share=True)