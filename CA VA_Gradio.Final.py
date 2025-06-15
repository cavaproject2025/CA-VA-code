# Install software packages:
%pip install langchain-openai==0.2.0
%pip install python-dotenv==1.0.0
%pip install langchain==0.3.4
%pip install langchainhub==0.1.13
%pip install pypdf==3.17.0
%pip install tiktoken==0.5.1
%pip install chromadb
%pip install langchain-community==0.3.20
%pip install unstructured
%pip install python-docx
%pip install unstructured langchain-community
%pip install selenium
%pip install "unstructured[all-docs]"
%pip install gradio

# %%
import os
from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings

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

# %%
#load web page from MOM PWM URLS:
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
# ========== UPDATED IMPORTS (FIX FOR LANGCHAIN DEPRECATION) ==========
# CHANGE: Updated to use langchain_community for deprecated imports
from langchain_community.document_loaders import DirectoryLoader
from langchain_community.document_loaders import UnstructuredWordDocumentLoader
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA

from langchain.document_loaders import WebBaseLoader
loader_multiple_pages = WebBaseLoader(["https://www.mom.gov.sg/employment-practices/progressive-wage-model/cleaning-sector",
                                      "https://www.mom.gov.sg/employment-practices/progressive-wage-model/retail-sector",
                                      "https://www.mom.gov.sg/employment-practices/progressive-wage-model/food-services-sector",
                                      "https://www.mom.gov.sg/employment-practices/progressive-wage-model/occupational-pws-for-administrators-and-drivers",
                                      "https://www.mom.gov.sg/employment-practices/leave/shared-parental-leave"
                                      "https://sso.agc.gov.sg/act/ema1968"
                                      "https://sso.agc.gov.sg/Act/CDCSA2001"
                                      "https://sso.agc.gov.sg/Act/IRA1960"
                                      "https://www.mom.gov.sg/-/media/mom/documents/employment-practices/guidelines/tripartite-advisory-on-managing-excess-manpower-and-responsible-retrenchment.pdf"
                                      ])



web_documents = loader_multiple_pages.load()

# %%
#load CAs in docx format
from langchain.document_loaders import DirectoryLoader
from langchain.document_loaders import UnstructuredWordDocumentLoader

folder_path="./wst"
txt_loader = DirectoryLoader(folder_path, glob="*.docx", loader_cls=UnstructuredWordDocumentLoader)
txt_documents = txt_loader.load()

# %%
# Import a splitter module called RecursiveCharacterTextSplitter:
from langchain.text_splitter import RecursiveCharacterTextSplitter
# Instantiate the splitter:
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
# Do the splitting:
splits = text_splitter.split_documents(loader.load())

# Embed and store splits in Chroma database.
# Import the Chroma and embeddings modules:
from langchain.vectorstores import Chroma
# from langchain.embeddings import OpenAIEmbeddings
# Instantiate the Chroma vector store with arguments:

# Comment out the following lines to process in smaller batches with delay to avoid hitting the rate limit when creating the embeddings:
# vectorstore = Chroma.from_documents(
#     documents=splits,
#     embedding=embedding_model
# )


import time
from tqdm import tqdm

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

# %%
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

# %%
# ========== CHANGE 4: COMPLETE GRADIO WEB INTERFACE (ENTIRELY NEW) ==========
# This entire section replaces the simple print statement with a full web UI
# CHANGE 1: Added Gradio import to enable web-based user interface. Remember to pip install gradio in the terminal within the virtual/conda environment before running the script.
import gradio as gr

import time
from tqdm import tqdm

os.environ["ALLOW_RESET"] = "TRUE"

load_dotenv('.env', override=True)
with gr.Blocks(title="Collective Agreement Virtual Assistant") as app:
    # Professional header with description
    gr.Markdown(
        """
        # 🤖 Collective Agreement Virtual Assistant
        
        Ask questions about collective agreements, salary ranges, and progressive wage models.
        The assistant draws information from existing legislations, tripartite guidelines, Collective Agreeement (CA) documents, and MOM resources.
        """
    )
        
    # Two-column layout for clean interface
    with gr.Row():
        with gr.Column():
            # Question input area with helpful placeholder
            question_input = gr.Textbox(
                label="Your Question",
                placeholder="e.g., What is the salary of drivers across food manufacturing CAs?",
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
            ["What is the salary ranges for drivers across food manufactuirng CAs?"],
            ["What is the current progressive wage model wage requirements for restroom cleaners?"],
            ["What is the salary ranges for housekeepers across hotels CAs?"],
            ["What is the current entitlement for shared parental leave?"],
            ["What is the current entitlement for childcare leave?"],
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

#PROMPT TEMPLATE
from langchain.schema.runnable import RunnablePassthrough
from langchain import hub
from langchain_core.prompts import PromptTemplate
template = """
You are a virtual assistant for Industrial Relations Officers in Food, Drinks and Allied Workers Union who have to benchmark Employees Agreement to draft Agreements for negotiations with unionised companies in Singapore.
For prompts relating to Progressive Wage Model or Occupational Progressive Wages, refer to the webpages loaded. For these prompts, to also include if the wage requirements refers monthly basic or gross wages.
Prompts with the word "PWM" refers to the Progressive Wage Model, while prompts with the word "OPW" refers to Occupational Progressive Wages. For these prompts, refer to the Ministry of Manpower webpages loaded for progressive wage models and occupational progressive wage.For these prompts, to also include if the wage requirements refers to monthly basic or gross wages.
For prompts on PWM wage requirements for cleaners or housekeepers, refer to the loaded webpage from MOM on Progressive Wage Model for Cleaning Sector: https://www.mom.gov.sg/employment-practices/progressive-wage-model/cleaning-sector
For prompts on OPW or PWM wage requirements for drivers, refer to the loaded webpage from MOM on Occupational Progressive Wages for administrators and drivers:https://www.mom.gov.sg/employment-practices/progressive-wage-model/occupational-pws-for-administrators-and-drivers
For prompts on PWM wage requirements for cooks or waiters, refer to the loaded webpage from MOM on Progressive Wage Model for Food Services Sector:https://www.mom.gov.sg/employment-practices/progressive-wage-model/food-services-sector
For prompts on OPW or PWM wage requirements for admin assistant, admin executive, concierge or storekeeper, refer to the loaded webpage from MOM on Occupational Progressive Wages for adminstrator and drivers:https://www.mom.gov.sg/employment-practices/progressive-wage-model/occupational-pws-for-administrators-and-drivers
For prompts on PWM wage requirements for cashier or retail assistant, refer to the loaded webpage from MOM on Progressive Wage Model for Retail Sector:https://www.mom.gov.sg/employment-practices/progressive-wage-model/retail-sector
For prompts relating to 'Collective Agreements' or 'CAs', it refers to the Employees Agreements loaded. The names of the companies can be found in the first page of the Employees Agreement.
For prompts relating to shared parental leave, paternity leave, childcare leave and maternity leave, refer to the webpage loaded from Child Development Co-Savings Act here:https://sso.agc.gov.sg/Act/CDCSA2001 .
For prompts relating to employment rights such as rest day pay and entitlements, refer to the webpage loaded on employment act: https://sso.agc.gov.sg/act/ema1968
For prompts relating to tripartite guidelines on retrenchments, refer to the MOM webpage loaded for tripartite guidelines on managing excess manpower and responsible retrenchment:https://www.mom.gov.sg/-/media/mom/documents/employment-practices/guidelines/tripartite-advisory-on-managing-excess-manpower-and-responsible-retrenchment.pdf .
For prompts requesting to list salary ranges of a specific job position across an industry (for example, hospitality/hotels and food manufacturing), refer to Employees Agreement of companies in those industries. List all the relevant job positions and indicate which company each salary ranges belongs to.
For prompts requesting to list salary ranges in Hotel CAs, it refers to Collective Agreeements from hotels, such as Conrad Centennial, Four Seasons, Grand Copthorne Waterfront, Grand Hyatt Singapore, Hotel Royal, Parkroyal Beach Rd, Raffles Hotel, RC Hotels, Ritz Carlton and Shangri-La Hotel. Common job positions in hotel CAs include housekeepers, concierge, bellman, F&B captain, waitress, engineering technician, chef and front office. Indicate which documents that the specific salary ranges are extracted from.
For prompts requesting to list salary ranges in Food Manufacturing CAs, it refers to Collective Agreements from food manufacturing companies, such as Asia Pacific Breweries, Lam Soon, Meiji Seika, Mondelez, Olam Cocoa, YHS. Common job positions in food manufacturing CAs include storekeepers, operators, technicians, merchandiser, clerk, general worker, production supervisor and forklift driver. Indicate which documents that the specific salary ranges are extracted from.
Be as detailed as possible with your responses.


{context}

Question: {question}
Helpful Answer:"""
custom_rag_prompt = PromptTemplate.from_template(template)

# Instantiate RAG chain:
custom_rag_chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | custom_rag_prompt
    | llm
)

# ========== CHANGE 5: LAUNCH WEB INTERFACE INSTEAD OF ONE-TIME EXECUTION ==========
# REPLACED: Single execution with print output
# NEW: Persistent web server that stays active for continuous use
if __name__ == "__main__":
    # share=True creates a public shareable link
    # Server runs continuously until stopped
    app.launch(share=True)


