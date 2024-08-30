# ********************************************************************************************** #
# File name: newmain.py
# Created by: Satya 
# Creation Date: 25-Jun-2024
# Application Name: INTELLIDOC_NEW.AI
#
# Change Details:
# Version No:     Date:        Changed by     Changes Done         
# 01             25-Jun-2024   Satya          Initial Creation
# 01             04-Jul-2024   Satya          Added logic for HR and Legal  -Deployed over Azure
# 02             23-Jul-2024   Satya          Config file and path variables is set
# 03             01-Aug-2024   Satya          Package issue resolved -Deployed over Azure
# 04             17-Aug-2024   Satya          Hybrid Retriver 
# ********************************************************************************************** #

#Added by Aruna for chromaDB SQLite version error
# __import__('pysqlite3')
# import sys
# sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import streamlit as st
import os
import chromadb
from llama_index.core import VectorStoreIndex
from llama_index.core import StorageContext
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.llms.huggingface import HuggingFaceInferenceAPI
from llama_index.core.llms import ChatMessage, MessageRole
from llama_index.core.prompts import ChatPromptTemplate
from llama_parse import LlamaParse
from llama_index.core import Document
from PIL import Image
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core.query_engine import CitationQueryEngine
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core import Settings
from dotenv import load_dotenv
from llama_index.core.storage.docstore import SimpleDocumentStore
import json
from unstructured.partition.pdf import partition_pdf
from streamlit_mic_recorder import speech_to_text
from datetime import datetime
import pandas as pd
import openai



load_dotenv()

LLAMA_API_KEY = os.getenv("LLAMA_API_KEY")
os.environ["LLAMA_CLOUD_API_KEY"] = LLAMA_API_KEY

HF_TOKEN = os.getenv("HF_TOKEN")
MODEL_NAME=os.getenv("MODEL_NAME")
EMBED_MODEL=os.getenv("EMBED_MODEL")
TITLE=os.getenv("TITLE")
ROLE = os.getenv("ROLE").split(',')
PAGE=os.getenv("PAGE").split(",")
ASK_QUESTION=os.getenv("ASK_QUESTION")
ASK=os.getenv("ASK")
UPLOAD_DOC=os.getenv("UPLOAD_DOC")
E_QUESTION=os.getenv("E_QUESTION")
SECTION=os.getenv("SECTION").split(",")
COLLECTION=os.getenv("COLLECTION").split(",")
DATABASE=os.getenv("DATABASE").split(",")
P_QUESTION=os.getenv("P_QUESTION")
INSERT_DOCUMENT=os.getenv("INSERT_DOCUMENT")
ADD_DOC=os.getenv("ADD_DOC")
DOC_ADDED=os.getenv("DOC_ADDED")
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
DELETE_DOC=os.getenv("DELETE_DOC")
C_DELETE=os.getenv("C_DELETE")
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
DOC_DELETED=os.getenv("DOC_DELETED")
N_DOC=os.getenv("N_DOC")
image=os.getenv("image")
imagess=os.getenv("imagess")


Settings.llm = HuggingFaceInferenceAPI(model_name=MODEL_NAME, token=HF_TOKEN)
Settings.embed_model = HuggingFaceEmbedding(model_name=EMBED_MODEL)
# llm = OpenAI(model="gpt-4o")  # Ensure model is set up correctly

#Settings.embed_model =OpenAIEmbedding()


# CSV file to store query results
CSV_FILE_PATH = "record_results.csv"

# Function to append a row to the CSV file
def append_to_csv(database, question, context_str, response, metadata):
    # Check if CSV exists and create a DataFrame
    if os.path.exists(CSV_FILE_PATH):
        df = pd.read_csv(CSV_FILE_PATH)
    else:
        df = pd.DataFrame(columns=[ "Database","Question", "Context", "Response", "Metadata"])

    # Create a new row
    new_row = pd.DataFrame({
        "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "Database": [database],
        "Question": [question],
        "Context": [context_str],
        "Response": [response],
        "Metadata": [metadata]
    })

    # Append the new row and save to CSV
    df = pd.concat([df, new_row], ignore_index=True)
    df.to_csv(CSV_FILE_PATH, index=False)
    
def main():

    img = Image.open(image)
    st.set_page_config(page_title=TITLE, page_icon=img,layout="wide",initial_sidebar_state="expanded")
    
    # Function to hide the sidebar with CSS
    def hide_sidebar():
       hide_style = """
        <style>
        [data-testid="stSidebar"] {
            display: none;
        }
        </style>
        """
       st.markdown(hide_style, unsafe_allow_html=True)


# Function to style radio buttons horizontally
    def style_radio_buttons():
        style = """
        <style>
        .stRadio > div {
            display: flex;
            flex-direction: row;
            justify-content: left-align;
        }
        .stRadio > div > label {
        }
        </style>
        """
        st.markdown(style, unsafe_allow_html=True)


# Apply the custom style for radio buttons
    style_radio_buttons()
    col1, col2 = st.columns([1, 9])

    with col1:
        st.image(imagess, width=130)

    with col2:
        st.title(TITLE)
        # Use HTML to add spacing between the icon and title

    tabs = st.radio("Choose your tab", ["**Admin**", "**User**"],label_visibility="hidden")
    # Check which tab is selected
    if tabs == "**User**":
        # Show sidebar in Tab 1
        with st.sidebar:
            st.subheader(UPLOAD_DOC)
            llm = HuggingFaceInferenceAPI(model_name=MODEL_NAME, token=HF_TOKEN)
            embed_model = HuggingFaceEmbedding(model_name=EMBED_MODEL)
            # Define options without Markdown syntax
            parser_options = ['LlamaParse', 'Unstructured.io']

            # Create a selectbox for parser choice
            parser_temp_choice = st.selectbox("Choose a parser:", parser_options,key="temp_select")

            # Display the selected parser in bold
            st.markdown(f"**Selected Parser:** {parser_temp_choice}")
            uploaded_files = st.file_uploader("**Choose PDF files**", type=["pdf", "csv", "xlsx", "docx", "pptx"], accept_multiple_files=True)

            # Function to parse files and create the index
            def initialize_index(uploaded_files, embed_model):
                temp_documents = []
                for uploaded_file in uploaded_files:
                    file_content = uploaded_file.read()
                    temp_file_name = uploaded_file.name
                    if parser_temp_choice == "LlamaParse":
                        parsed_text = use_llamaparse(file_content, temp_file_name)
                    else:
                        parsed_text = use_unstructured(file_content, temp_file_name)

                    # Split the parsed text into chunks
                    chunk_size = 1000  # Set the desired chunk size (e.g., 1000 characters)
                    chunks = [parsed_text[i:i + chunk_size] for i in range(0, len(parsed_text), chunk_size)]

                    # Create document objects from each chunk
                    for chunk in chunks:
                        document = Document(text=chunk)
                        temp_documents.append(document)

                # Create the index from the chunked documents
                temp_index = VectorStoreIndex.from_documents(temp_documents, embed_model=embed_model)
                return temp_index
            
            if uploaded_files:
                # Check if the index is already initialized
                if "temp_index" not in st.session_state:
                    if uploaded_files:
                        # Initialize the index and store it in session state
                        st.session_state.temp_index = initialize_index(uploaded_files, embed_model)

                # Retrieve the index from session state
                temp_index = st.session_state.temp_index

                # Set up a prompt template for the question-answering task
                qa_prompt_str_temp = (
                    "Context information is below.\n"
                    "---------------------\n"
                    "{context_str}\n"
                    "---------------------\n"
                    "Given the context information and not prior knowledge, "
                    "answer the question: {query_str}\n"
                )

                # Set up a retriever to get relevant nodes
                retriever = temp_index.as_retriever(similarity_top_k=3)

                # Option to select between text input or voice input
                input_option = st.radio("**How would you like to ask your question?**", ('**Text**', '**Voice**'))

                if input_option == '**Text**':
                    st.text_input(ASK_QUESTION, key="temp_question")

                else:
                    c1, c2 = st.columns(2)
                    with c1:
                        st.write("**Convert speech to text:**")
                    with c2:
                        speech_to_text(language='en', use_container_width=True, just_once=True, key='STT', callback=callback)
                    temp_question = st.session_state.get("temp_question", "")
                    # Display the question
                    if temp_question:
                        st.write("**You asked:**", temp_question)

                if st.button(ASK, key="temp_ask"):
                    question_temp = st.session_state.get("temp_question", "")
                    if question_temp:
                        # Retrieve nodes relevant to the question
                        retrieved_nodes = retriever.retrieve(question_temp)
                        context_str = "\n\n".join([r.get_content()[:4000] for r in retrieved_nodes])

                        # Format the QA prompt
                        fmt_qa_prompt = qa_prompt_str_temp.format(context_str=context_str, query_str=question_temp)

                        chat_text_qa_msgs = [
                            ChatMessage(
                                role=MessageRole.SYSTEM,
                                content="You strictly answer the question from the context of the given document only and if you do not know just say 'OUT OF CONTEXT QUESTION'.\n"
                                        "You do not give any kind of information out of the document given by the user.\n"
                            ),
                            ChatMessage(
                                role=MessageRole.USER,
                                content=fmt_qa_prompt
                            ),
                        ]
                        text_qa_template = ChatPromptTemplate(chat_text_qa_msgs)

                        # Query the index using the formatted prompt
                        result = temp_index.as_query_engine(text_qa_template=text_qa_template, llm=llm).query(question_temp)

                        # Display the result
                        st.write(result.response)
                    else:
                        st.warning(P_QUESTION)
        user_page()

    elif tabs == "**Admin**":
    # Hide sidebar in Tab 2
        hide_sidebar()
        admin_page()


def admin_page():
     #Initialize session state for section if not already set
    if "section" not in st.session_state:
        st.session_state.section = SECTION[0]  # Default to the first section

    # Create a selectbox for section selection
    selected_section = st.selectbox("**Select section**", SECTION, key="section_selectbox")
    # Call admin_operations based on the selected section
    if selected_section in SECTION:
        index = SECTION.index(selected_section)
        admin_operations(COLLECTION[index], DATABASE[index])



def callback():
    if st.session_state.STT_output:
        st.session_state["temp_question"] = st.session_state.STT_output


if 'text_received' not in st.session_state:

    st.session_state.text_received =[]                 

def admin_operations(collection_name, db_path):

    # Initialize Streamlit app
    st.write(f"**Managing documents for collection: {collection_name}**")
    collection = init_chroma_collection(db_path, collection_name)

    # Initialize session state for document mapping and list
    if "doc_name_to_id" not in st.session_state:
        st.session_state.doc_name_to_id = {}
    if "doc_list" not in st.session_state:
        docs = collection.get()['metadatas']
        ids = collection.get()['ids']  # Get IDs separately to avoid multiple calls
        for doc_id, meta in zip(ids, docs):
            if 'source' in meta:
                doc_name = meta['source'].split('\\')[-1]
                if doc_name not in st.session_state.doc_name_to_id:
                    st.session_state.doc_name_to_id[doc_name] = []
                st.session_state.doc_name_to_id[doc_name].append(doc_id)  # Append all IDs
        st.session_state.doc_list = list(st.session_state.doc_name_to_id.keys())

    if st.button("**Show Document**"):
        docs = collection.get()['metadatas']
        ids = collection.get()['ids']  # Get IDs separately
        st.session_state.doc_name_to_id = {}
        for doc_id, meta in zip(ids, docs):
            if 'source' in meta:
                doc_name = meta['source'].split('\\')[-1]
                if doc_name not in st.session_state.doc_name_to_id:
                    st.session_state.doc_name_to_id[doc_name] = []
                st.session_state.doc_name_to_id[doc_name].append(doc_id)  # Append all IDs
        st.session_state.doc_list = list(st.session_state.doc_name_to_id.keys())
        st.selectbox("Documents", st.session_state.doc_list, key="doc_select") 

    if "show_uploader" not in st.session_state:
        st.session_state.show_uploader = False       

    if st.button("**Insert Document**"):
        st.session_state.show_uploader = True
    if st.session_state.show_uploader:
        
        # Define options without Markdown syntax
        parser_options = ['LlamaParse', 'Unstructured.io']

        # Create a selectbox for parser choice
        parser_choice = st.selectbox("Choose a parser:", parser_options)

        # Display the selected parser in bold
        st.markdown(f"**Selected Parser:** {parser_choice}")
        # File upload for adding documents

        files = st.file_uploader("**Choose PDF files**", key="admin_upload", accept_multiple_files=True)


        if files and st.button("**Add Document**"):
            # Ensure that files is a list
            for file in files:
                file_content = file.read()
                file_name = file.name

                if parser_choice == "LlamaParse":
                    parsed_text = use_llamaparse(file_content, file_name)
                    #print(parsed_text)
                else:
                    parsed_text = use_unstructured(file_content, file_name)
                    #print(parsed_text)



                base_splitter = SentenceSplitter(chunk_size=512)
                nodes = base_splitter.get_nodes_from_documents([Document(text=parsed_text)])

                # Initialize storage context (by default it's in-memory)
                storage_context = StorageContext.from_defaults()

                # Use the file name as the base ID
                base_file_name = os.path.basename(file.name)
                chunk_ids = []
                metadatas = []

                # Prepare data for storage and collection
                for i, node in enumerate(nodes):
                    chunk_id = f"{base_file_name}_{i + 1}"  # Create chunk IDs like "file.pdf_1", "file.pdf_2", etc.
                    chunk_ids.append(chunk_id)

                    metadata = {"type": base_file_name, "source": file.name}
                    metadatas.append(metadata)

                    # Create a new Document or Node object with text and metadata
                    document = Document(text=node.text, metadata=metadata, id_=chunk_id)

                    # Store the document in the storage context
                    storage_context.docstore.add_documents([document])

                # Load existing documents from the .json file if it exists
                existing_documents = {}
                if os.path.exists(f"./docstore_{collection_name}.json"):
                    with open(f"./docstore_{collection_name}.json", "r") as f:
                        existing_documents = json.load(f)

                    # Persist the storage context (if necessary)
                    storage_context.docstore.persist(f"./docstore_{collection_name}.json")

                    # Load new data from the same file (or another source)
                    with open(f"./docstore_{collection_name}.json", "r") as f:
                        st_data = json.load(f)

                    # Update existing_documents with st_data
                    for key, value in st_data.items():
                        if key in existing_documents:
                            # Ensure the existing value is a list before extending
                            if isinstance(existing_documents[key], list):
                                existing_documents[key].extend(value)  # Merge lists if key exists
                            else:
                                # If it's not a list, you can choose to replace it or handle it differently
                                existing_documents[key] = [existing_documents[key]] + value if isinstance(value, list) else [existing_documents[key], value]
                        else:
                            existing_documents[key] = value  # Add new key-value pair

                    merged_dict = {}
                    for d in existing_documents["docstore/data"]:
                        merged_dict.update(d)
                    final_dict = {}
                    final_dict["docstore/data"] = merged_dict

                    # Write the updated documents back to the JSON file
                    with open(f"./docstore_{collection_name}.json", "w") as f:
                        json.dump(final_dict, f, indent=4)

                else:
                    # Persist the storage context if the file does not exist
                    storage_context.docstore.persist(f"./docstore_{collection_name}.json")
                embed_model = HuggingFaceEmbedding(model_name=EMBED_MODEL)
                # Create the vector store index
                VectorStoreIndex(nodes, storage_context=storage_context, embed_model=embed_model)

                # Add the chunks to the collection
                collection.add(
                    documents=[node.text for node in nodes],
                    metadatas=metadatas,
                    ids=chunk_ids
                )

                # Update session state
                if base_file_name not in st.session_state.doc_name_to_id:
                    st.session_state.doc_name_to_id[base_file_name] = []  # Initialize if not present
                st.session_state.doc_name_to_id[base_file_name].extend(chunk_ids)  # Store all chunk IDs under the base file name
                st.session_state.doc_list.append(base_file_name)  # Store only the base file name

            st.success("Documents added successfully!")
            st.session_state.show_uploader = False

    # Deletion of documents
    if "show_delete" not in st.session_state:
         st.session_state.show_delete = False
        
    if st.button("**Delete Document**"):
         st.session_state.show_delete = True
        
    if st.session_state.show_delete:
        # Ensure the document list is populated

        if not st.session_state.doc_list:
            st.warning("No documents available to delete.")
        else:
            selected_doc_to_delete = st.selectbox("Select Document to Delete", st.session_state.doc_list, key="doc_delete_select")
            # print("Selected document for deletion:", selected_doc_to_delete)
            
            if st.button("**Confirm Delete**"):
                if selected_doc_to_delete:
                    
                    # Retrieve all chunk IDs associated with the selected document name
                    ids_to_delete = st.session_state.doc_name_to_id.get(selected_doc_to_delete, [])
                    # print("IDs to delete:", ids_to_delete)

                    if ids_to_delete:
                        try:
                            current_ids = collection.get()['ids']
                            # print("Current IDs in collection:", current_ids)
                            
                            # Check for existing IDs to delete
                            ids_to_delete_existing = [id for id in ids_to_delete if id in current_ids]
                            # print("Valid IDs to delete:", ids_to_delete_existing)

                            # Step 1: Read the JSON file
                            with open(f'docstore_{collection_name}.json', 'r') as file:
                                data = json.load(file)["docstore/data"]

                            for i in ids_to_delete_existing:
                                del data[i]

                            final_dict={}
                            final_dict["docstore/data"]=data

                            with open(f'docstore_{collection_name}.json', 'w') as file:
                                json.dump(final_dict, file, indent=4)

                            
                            if ids_to_delete_existing:
                                collection.delete(ids=ids_to_delete_existing)
                                
                                # Update session state after deletion
                                del st.session_state.doc_name_to_id[selected_doc_to_delete]
                                st.session_state.doc_list.remove(selected_doc_to_delete)
                                
                                st.success(f"Document '{selected_doc_to_delete}' deleted successfully!")
                        except Exception as e:
                            st.error(f"Error deleting document: {str(e)}")
                    else:
                        st.warning(f"No chunks found for document '{selected_doc_to_delete}'.")
                else:
                    st.warning("No document selected for deletion.")
            

def user_page():
    
    if "selected_role" not in st.session_state:
       st.session_state.selected_role = ROLE[0]

    # Create a selectbox for role selection
    selected_role = st.selectbox("**Select your role**", ROLE)

    # Update the session state with the selected role
    st.session_state.selected_role = selected_role

    
    if st.session_state.selected_role == ROLE[0]:
        hr_team_page()
    elif st.session_state.selected_role == ROLE[1]:
        legal_team_page()
    elif st.session_state.selected_role == ROLE[2]:
        finance_team_page()
    elif st.session_state.selected_role == ROLE[3]:
        operations_team_page()
    elif st.session_state.selected_role == ROLE[4]:
        medical_team_page()
    elif st.session_state.selected_role == ROLE[5]:
        insurance_team_page()
    elif st.session_state.selected_role == ROLE[6]:
        LD_team_page()
    elif st.session_state.selected_role == ROLE[7]:
        others_team_page()                                        


def hr_team_page():
    st.subheader(PAGE[1])
    query_page(COLLECTION[0], DATABASE[0], admin=False)

def legal_team_page():
    st.subheader(PAGE[2])
    query_page(COLLECTION[1], DATABASE[1], admin=False)

def finance_team_page():
    st.subheader(PAGE[3])
    query_page(COLLECTION[2], DATABASE[2], admin=False)

def operations_team_page():
    st.subheader(PAGE[4])
    query_page(COLLECTION[3], DATABASE[3], admin=False)

def medical_team_page():
    st.subheader(PAGE[5])
    query_page(COLLECTION[4], DATABASE[4], admin=False)

def insurance_team_page():
    st.subheader(PAGE[6])
    query_page(COLLECTION[5], DATABASE[5], admin=False)

def LD_team_page():
    st.subheader(PAGE[7])
    query_page(COLLECTION[6], DATABASE[6], admin=False)

def others_team_page():
    st.subheader(PAGE[8])
    query_page(COLLECTION[7], DATABASE[7], admin=False)

def show_documents(collection, key_prefix):
    # Initialize or update session state for doc_name_to_id and doc_list
    if f"{key_prefix}_doc_name_to_id" not in st.session_state:
        st.session_state[f"{key_prefix}_doc_name_to_id"] = {}
    if f"{key_prefix}_doc_list" not in st.session_state:
        docs = collection.get()['metadatas']
        st.session_state[f"{key_prefix}_doc_name_to_id"] = {meta['source'].split('\\')[-1]: doc_id for doc_id, meta in zip(collection.get()['ids'], docs) if 'source' in meta}
        st.session_state[f"{key_prefix}_doc_list"] = list(st.session_state[f"{key_prefix}_doc_name_to_id"].keys())

    if st.button("Show Document", key=f"{key_prefix}_show_document"):
        # Refresh doc_name_to_id and doc_list in session state
        docs = collection.get()['metadatas']
        st.session_state[f"{key_prefix}_doc_name_to_id"] = {meta['source'].split('\\')[-1]: doc_id for doc_id, meta in zip(collection.get()['ids'], docs) if 'source' in meta}
        st.session_state[f"{key_prefix}_doc_list"] = list(st.session_state[f"{key_prefix}_doc_name_to_id"].keys())
        st.selectbox("Documents", st.session_state[f"{key_prefix}_doc_list"], key=f"{key_prefix}_doc_select", placeholder="See the document...",)


def query_page(collection_name, db_path, admin):
    llm = Settings.llm
    embed_model = Settings.embed_model

    # Initialize Chroma collection
    collection = init_chroma_collection(db_path, collection_name)
    # show_documents(collection, key_prefix="key_prefix")
        

    # # Initialize chat history and input mode
    # if "message" not in st.session_state:
    #     st.session_state.message = []
    # if "input_mode" not in st.session_state:
    #     st.session_state.input_mode = "Text"
    # if "voice_question" not in st.session_state:
    #     st.session_state.voice_question = ""

    # # Function to reset chat
    # def reset_chat():
    #     st.session_state.message = []

    # # Function to handle input mode switch
    # def switch_input_mode():
    #     # Clear the previous question when switching modes
    #     if st.session_state.input_mode == "Text":
    #         st.session_state.voice_question = ""
    #     else:
    #         st.session_state.text_input = ""

    # # Create columns for buttons
    # col1, col2 = st.columns([1, 1])  # Adjust the ratio if needed

    # # Place buttons in columns
    # with col1:
    #     show_documents(collection, key_prefix="key_prefix")

    # with col2:
    #     if st.button("Reset Chat"):
    #         reset_chat()

    # # Option to select between text input or voice input
    # input_option = st.radio("**How would you like to ask your question?**", 
    #                         ('Text', 'Voice'), 
    #                         key="input_mode", 
    #                         on_change=switch_input_mode)

    # if st.session_state.input_mode == 'Text' and input_option=='Text':
    #     single_question = st.chat_input("Enter your question:", key="text_input")

    # else:
    #     c1, c2 = st.columns(2)
    #     with c1:
    #         st.write("**Convert speech to text:**")
    #     with c2:
    #         speech_to_text(language='en', use_container_width=True, just_once=True, key='ST', callback=callback)
    #     single_question = st.session_state.get("temp_questions", "")
    #     print(single_question)
    
    def callback():
        if st.session_state.STT_output:
            st.session_state["temp_question"] = st.session_state.STT_output
    # Initialize chat history
    if "message" not in st.session_state:
        st.session_state.message= []

    # Function to reset chat
    def reset_chat():
        st.session_state.message = []

    # Create columns for buttons
    col1, col2 = st.columns([1, 1])  # Adjust the ratio if needed

    # Place buttons in columns
    with col1:
        show_documents(collection, key_prefix="key_prefix")

    with col2:
        if st.button("Reset Chat"):
            reset_chat()

    # Option to select between text input or voice input
    input_option = st.radio("**How would you like to ask your question?**", ('**Text**', '**Voice**'),key="kk")

    if input_option == '**Text**':
        single_question = st.chat_input("Enter your question:")

    else:
        c1, c2 = st.columns(2)
        with c1:
            st.write("**Convert speech to text:**")
        with c2:
            speech_to_text(language='en', use_container_width=True, just_once=True, key='STT', callback=callback)
            single_question = st.session_state.get("temp_question", "")



    # Display chat messages from history on app rerun
    for mes in st.session_state.message:
        with st.chat_message(mes["role"]):
            st.markdown(mes["content"])

    if single_question:
        st.chat_message("user").markdown(single_question)
        st.session_state.message.append({"role": "user", "content": single_question})

        # Process the single question if provided
        if 'documents' in collection.get() and len(collection.get()['documents']) > 0:
            vector_store = ChromaVectorStore(chroma_collection=collection)
            docstore = SimpleDocumentStore.from_persist_path(f"./docstore_{collection_name}.json")
            storage_context = StorageContext.from_defaults(docstore=docstore, vector_store=vector_store)
            vector_index = VectorStoreIndex(nodes=[], storage_context=storage_context, embed_model=embed_model)

            # Create the BM25 retriever
            bm25_retriever = BM25Retriever.from_defaults(docstore=docstore, similarity_top_k=2)

            # Function to perform hybrid retrieval
            def hybrid_retrieve(query, alpha=0.5):
                # Get results from BM25
                bm25_results = bm25_retriever.retrieve(query)
                # Get results from the vector store
                vector_results = vector_index.as_retriever(similarity_top_k=2).retrieve(query)

                # Combine results with weighting
                combined_results = {}
                # Weight BM25 results
                for result in bm25_results:
                    combined_results[result.id_] = combined_results.get(result.id_, 0) + (1 - alpha)
                # Weight vector results
                for result in vector_results:
                    combined_results[result.id_] = combined_results.get(result.id_, 0) + alpha

                # Sort results based on the combined score
                sorted_results = sorted(combined_results.items(), key=lambda x: x[1], reverse=True)
                # Return the top N results
                return [docstore.get_document(doc_id) for doc_id, _ in sorted_results[:4]]

            # Set up a prompt template for the question-answering task
            qa_prompt_str = (
                "Context information is below.\n"
                "---------------------\n"
                "{context_str}\n"
                "---------------------\n"
                "Given the context information and not prior knowledge, "
                "answer the question: {query_str}\n"
            )

            alpha = 0.7  # Adjust alpha as needed
            retrieved_nodes = hybrid_retrieve(single_question, alpha)

            context_str = "\n\n".join([r.get_content()[:4000] for r in retrieved_nodes])
            fmt_qa_prompt = qa_prompt_str.format(context_str=context_str, query_str=single_question)

            chat_text_qa_msgs = [
                ChatMessage(
                    role=MessageRole.SYSTEM,
                    content="You strictly answer the question from the context of the given document only and if you do not know just say 'OUT OF CONTEXT QUESTION'.\n"
                            "You do not give any kind of information out of the document given by user.\n"
                ),
                ChatMessage(
                    role=MessageRole.USER,
                    content=fmt_qa_prompt
                ),
            ]
            text_qa_template = ChatPromptTemplate(chat_text_qa_msgs)
            query_engine = vector_index.as_query_engine(
                text_qa_template=text_qa_template,
                llm=llm,
            )
            response = query_engine.query(single_question)

            # Citation query
            query_engine = CitationQueryEngine.from_args(
                vector_index,
                citation_chunk_size=1024,
                similarity_top_k=3
            )
            citation = query_engine.query(single_question)

            if "out of context" in str(response.response).lower():
                source = "NO METADATA"
            else:
                metadata = citation.metadata
                for key, value in metadata.items():
                    if "source" in value:
                        source = value["source"]
                        break
                else:
                    source = "Source not found in metadata."
            
            # Display the response for the single question
            st.chat_message("assistant").markdown(f"{response.response}"+"  "+f"{source}")
            st.session_state.message.append({"role": "assistant", "content": response.response})
            append_to_csv(collection_name, single_question, context_str, str(response), str(metadata))
            single_question=""


    


def init_chroma_collection(db_path, collection_name):
    db = chromadb.PersistentClient(path=db_path)
    return db.get_or_create_collection(collection_name)

def query_index(index, question, qa_prompt_str, llm):
    chat_text_qa_msgs = [
        ChatMessage(
            role=MessageRole.SYSTEM,
            content="You strictly answer the question from the context of given document only and if you do not know just say 'OUT OF CONTEXT QUESTION'.\n"
                    "You do not give any kind of information out of the document given by user. \n"
        ),
        ChatMessage(
            role=MessageRole.USER, 
            content=qa_prompt_str),
    ]
    text_qa_template = ChatPromptTemplate(chat_text_qa_msgs)
    
    result = index.as_query_engine(
        text_qa_template=text_qa_template,
        llm=llm,
    ).query(question)
    return result


def use_llamaparse(file_content, file_name):
    with open(file_name, "wb") as f:
        f.write(file_content)
    
    parser = LlamaParse(result_type='text', verbose=True, language="en", num_workers=2)
    documents = parser.load_data([file_name])
    
    os.remove(file_name)
    
    res = ''
    for i in documents:
        res += i.text + " "
    return res

def use_unstructured(file_content, file_name):
    with open(file_name, "wb") as f:
        f.write(file_content)
    
    elements = partition_pdf(file_name)
    
    os.remove(file_name)
    
    return "\n\n".join([str(el) for el in elements])


if __name__ == "__main__":
    main()
