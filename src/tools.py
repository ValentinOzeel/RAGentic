import os
import time
import regex as re
import copy
from typing import List, Dict, Union, Literal

from contextlib import contextmanager
import pandas as pd
import numpy as np

import yaml
import json
import sqlite3

import easyocr

from llama_parse import LlamaParse
from langchain_community.document_loaders import PDFMinerLoader, PDFPlumberLoader

from sentence_transformers import SentenceTransformer
from langchain_core.documents import Document
from langchain.embeddings.base import Embeddings
from langchain.indexes import SQLRecordManager, index
from langchain_community.document_loaders import TextLoader
## Vector databases
from qdrant_client import QdrantClient, models
from qdrant_client.models import Distance, VectorParams, Filter, FieldCondition, MatchValue, MatchAny
## Langchain integreation
from langchain_qdrant import QdrantVectorStore
from langchain_qdrant import FastEmbedSparse, RetrievalMode
## Rerank
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import FlashrankRerank
## LLM
from langchain_ollama import ChatOllama
## Promps
#from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_core.output_parsers import BaseOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.prompts.chat import ChatPromptTemplate, MessagesPlaceholder
## RRF
#from langchain.load import dumps, loads
## Chat history
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.messages import trim_messages
from langchain.schema import StrOutputParser
## Messages
from langchain_core.messages import BaseMessage, AIMessage, HumanMessage, SystemMessage
## Callbacks (streaming repsonse)
from langchain.callbacks.manager import CallbackManager

from constants import (
    credentials_yaml_path, 
    image_to_text_languages,
    entry_id_col_name, chunked_entry_id_col_name, date_col_name, main_tags_col_name, sub_tags_col_name, doc_type_col_name, text_col_name,
    sqlite_database_path, sqlite_tags_separator,
    recursive_character_text_splitter,
    device,
    vdb, qdrant_database_path, 
    sql_record_manager_path, 
    embeddings_model_name, embeddings_query_prompt, vector_dimensions, sparse_embeddings_model_name,
    retrieval_mode, retrieval_rerank, retrieval_search_type, filter_strictness_choices, k_outputs_retrieval, relevance_threshold, mmr_fetch_k, mmr_lambda_mult,
    llm_name, llm_temperature, max_chat_history_tokens
)

from prompts import (
    multi_query_prompt, 
    chat_history_contextualize_q_system_prompt, 
    doc_ids_used_in_rag_response_system_prompt, 
    rag_system_prompt
)
            

class YamlManagment():
    '''YAML credential managment'''
    
    @staticmethod
    def create_cred_yaml_file():
        #Initialize the YAML file with an empty dictionary.
        if not os.path.isfile(credentials_yaml_path):
            with open(credentials_yaml_path, 'w') as file:
                yaml.dump({'user_creds': {}}, file)
            
    @staticmethod
    def add_user_credentials(email, password):
        '''
        Add user credentials upon account creation
        ''' 
        # Load yaml creadentials
        with open(credentials_yaml_path, 'r') as file:
            data = yaml.safe_load(file) 
        # Create user_creds dict if doesnt exist
        if not data or not data.get('user_creds', None):
            data = {'user_creds': {}} 
        # Check if provided email already exists
        if email in data['user_creds']:
            return False
        # If it does not, then add the credentials
        else:
            data['user_creds'][email] = password 
            # Add credentials in user_creds Yaml Dict 
            with open(credentials_yaml_path, 'w') as file:
                yaml.dump(data, file)
            return True
        
    @staticmethod
    def check_user_credentials(email, password):
        with open(credentials_yaml_path, 'r') as file:
            data = yaml.safe_load(file)  
            
        # Check if provided email exists
        if not data['user_creds'].get(email, None):
            return False
        # Check if provided password match with the password associated to email
        if data['user_creds'][email] != password:
            return False
        # Return True if user credentials are valid
        return True

    @staticmethod
    def _get_user_n_entry_id(user_id):
        '''
        Add user credentials upon account creation
        ''' 
        # Load yaml credentials
        with open(credentials_yaml_path, 'r') as file:
            data = yaml.safe_load(file)  
        # Create user_n_entry_id dict if doesnt exist
        if not data.get('user_n_entry_id', None):
            data['user_n_entry_id'] = {}
        # Init user_n_entry_id for user_if as 0 if it doesnt exists already
        if not data['user_n_entry_id'].get(user_id, None):
            data['user_n_entry_id'][user_id] = 1
        # Persist modifs in yaml
        with open(credentials_yaml_path, 'w') as file:
            yaml.dump(data, file)
        # Return value
        return data['user_n_entry_id'][user_id]

    @staticmethod
    def _increment_user_n_entry_id(user_id, n):
        # Load yaml credentials
        with open(credentials_yaml_path, 'r') as file:
            data = yaml.safe_load(file)  
        # increment user' entries number
        data['user_n_entry_id'][user_id] = data['user_n_entry_id'][user_id] + n
        # Persist data in yaml
        with open(credentials_yaml_path, 'w') as file:
            yaml.dump(data, file)

    @staticmethod
    def _add_user_pdf(user_id, pdf_file_name, doc_id):
        # Load yaml credentials
        with open(credentials_yaml_path, 'r') as file:
            data = yaml.safe_load(file)  
        # Create user_loaded_pdfs dict if doesnt exist
        if not data.get('user_loaded_pdfs', None):
            data['user_loaded_pdfs'] = {}
        # Init user_loaded_pdfs for user_if if it doesnt exists already
        if not data['user_loaded_pdfs'].get(user_id, None):
            data['user_loaded_pdfs'][user_id] = {}
        # Append pdf file name
        data['user_loaded_pdfs'][user_id][pdf_file_name] = doc_id
        # Persist data in yaml
        with open(credentials_yaml_path, 'w') as file:
            yaml.dump(data, file)

    @staticmethod
    def get_user_pdf_names_ids(user_id):
        # Load yaml credentials
        with open(credentials_yaml_path, 'r') as file:
            data = yaml.safe_load(file)  
        if not data.get('user_loaded_pdfs', None):
            data['user_loaded_pdfs'] = {}
        return data['user_loaded_pdfs'].get(user_id, {})

    @staticmethod
    def get_entry_id_and_increment(user_id):
        entry_id = YamlManagment._get_user_n_entry_id(user_id)            
        # Actualize n entries in yaml
        YamlManagment._increment_user_n_entry_id(user_id, 1)
        return str(entry_id) 
            
                     
class SignLog():
    ###                 ###               
    ### Sign in methods ###
    ###                 ###
    @staticmethod
    def email_check(email_input:str):
        email_regex = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,7}\b'
        return True if re.match(email_regex, email_input) else False

    @staticmethod
    def send_email_as_verif():
        #import smtplib
        #from email.mime.text import MIMEText
        #
        #subject = "Email Subject"
        #body = "This is the body of 
        #the text message"
        #sender = "sender@gmail.com"
        #recipients = ["recipient1@gmail.com", "recipient2@gmail.com"]
        #password = "password"
        #
        #
        #def send_email(subject, body, sender, recipients, password):
        #    msg = MIMEText(body)
        #    msg['Subject'] = subject
        #    msg['From'] = sender
        #    msg['To'] = ', '.join(recipients)
        #    with smtplib.SMTP_SSL('smtp.gmail.com', 465) as smtp_server:
        #       smtp_server.login(sender, password)
        #       smtp_server.sendmail(sender, recipients, msg.as_string())
        #    print("Message sent!")
        #
        #
        #send_email(subject, body, sender, recipients, password)
        pass



### OCR FUNCTION ###
### OCR FUNCTION ###
### OCR FUNCTION ###
def image_to_text_conversion(selected_languages, cpu_or_gpu, image_path):
    '''Image (IN) to text (OUT) OCR function'''

    gpu = True if cpu_or_gpu == 'GPU' else False
    languages = [image_to_text_languages[selected_languages]] if isinstance(selected_languages, str) else [
        image_to_text_languages[language] for language in selected_languages
        ] 

    reader = easyocr.Reader(languages, gpu=gpu)
    result = reader.readtext(image_path)
    whole_text = ' '.join([text for (bbox, text, prob) in result])
    return whole_text




### SQLITE DB MANAGMENT CLASS ###
### SQLITE DB MANAGMENT CLASS ###
### SQLITE DB MANAGMENT CLASS ###
class SQLiteManagment:
    '''SQLite DATABASE MANAGMENT'''

    @staticmethod
    @contextmanager
    def get_db_connection():
        """Get a connection to the SQLite database and close it when context ends (after user action)."""
        conn = sqlite3.connect(sqlite_database_path)
        try:
            yield conn
        finally:
            conn.close()
             
    @staticmethod
    def initialize_db():
        """Initialize the SQLite database with the necessary table if not already exists."""
        with SQLiteManagment.get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(f'''
                CREATE TABLE IF NOT EXISTS entries (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT NOT NULL,
                    {entry_id_col_name} TEXT NOT NULL,
                    {chunked_entry_id_col_name} TEXT NOT NULL,
                    {date_col_name} TEXT,
                    {main_tags_col_name} TEXT,
                    {sub_tags_col_name} TEXT,
                    {doc_type_col_name} TEXT, 
                    {text_col_name} TEXT NOT NULL
                )
            ''')
            conn.commit()
            
    @staticmethod   
    def _get_number_of_entries(user_id):
        """Get the number of entries in the database for a specific user."""
        with SQLiteManagment.get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT COUNT(*) FROM entries WHERE user_id = ?', (user_id,))
            count = cursor.fetchone()[0]
            return count
    
    @staticmethod 
    def add_entry_to_sqlite(user_id, single_entry=None, multiple_entries: List[Dict] = None):
        """Add an entry or multiple entries to the SQLite database."""
        def ignore_if_empty(entry):
            return not entry.get(text_col_name, None)
        
        max_retries = 5
        delay = 0.1

        for attempt in range(max_retries):
            try:
                with SQLiteManagment.get_db_connection() as conn:
                    cursor = conn.cursor()
                    
                    if single_entry:
                        if ignore_if_empty(single_entry):
                            return
                        # Validate and prepare the values for insertion
                        values = (
                            user_id,
                            single_entry.get(entry_id_col_name, None),
                            single_entry.get(chunked_entry_id_col_name, None),
                            single_entry.get(date_col_name, None),
                            single_entry.get(main_tags_col_name, None),
                            single_entry.get(sub_tags_col_name, None),
                            single_entry.get(doc_type_col_name, None),
                            single_entry.get(text_col_name, None)
                        )
                        cursor.execute(f'''
                            INSERT INTO entries (user_id, {entry_id_col_name}, {chunked_entry_id_col_name}, {date_col_name}, {main_tags_col_name}, {sub_tags_col_name}, {doc_type_col_name}, {text_col_name})
                            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                        ''', values)
      
                    elif multiple_entries:                        
                        entries_to_add = [
                            (
                                user_id,
                                entry.get(entry_id_col_name, None),
                                entry.get(chunked_entry_id_col_name, None),
                                entry.get(date_col_name, None),
                                entry.get(main_tags_col_name, None),
                                entry.get(sub_tags_col_name, None),
                                entry.get(doc_type_col_name, None),
                                entry.get(text_col_name, None)
                            )
                            for entry in multiple_entries if not ignore_if_empty(entry)
                        ]
                        cursor.executemany(f'''
                            INSERT INTO entries (user_id, {entry_id_col_name}, {chunked_entry_id_col_name}, {date_col_name}, {main_tags_col_name}, {sub_tags_col_name}, {doc_type_col_name}, {text_col_name})
                            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                        ''', entries_to_add)
                    conn.commit()
                break  # Exit the retry loop if successful
            
            except sqlite3.OperationalError as e:
                if "database is locked" in str(e):
                    time.sleep(delay)  # Wait before retrying
                    delay *= 2  # Exponential backoff
                else:
                    raise  # Re-raise other operational errors

    @staticmethod  
    def sqlite_to_dataframe(user_id):
        """Convert the database entries to a Pandas DataFrame for a specific user."""
        try:
            with SQLiteManagment.get_db_connection() as conn:
                query = f'''
                    SELECT id as {entry_id_col_name}, {chunked_entry_id_col_name}, {date_col_name}, {main_tags_col_name}, {sub_tags_col_name}, {doc_type_col_name}, {text_col_name} 
                    FROM entries 
                    WHERE user_id = ? 
                    ORDER BY id
                '''
                df = pd.read_sql_query(query, conn, params=(user_id,))
                df[date_col_name] = pd.to_datetime(df[date_col_name], format='%m/%d/%Y')
                df[main_tags_col_name] = df[main_tags_col_name].apply(lambda value: ' '.join(value.split(sqlite_tags_separator)))
                df[sub_tags_col_name] = df[sub_tags_col_name].apply(lambda value: ' '.join(value.split(sqlite_tags_separator)))
                return df
        except Exception as e:
            print(e)
            return None
        


### EMBEDDINGS CLASS ###
### EMBEDDINGS CLASS ###
### EMBEDDINGS CLASS ###
class SentenceTransformersEmbeddings(Embeddings):
    def __init__(self, model_name: str = embeddings_model_name):
        # Initialize the SentenceTransformer model using GPU or CPU
        self.model = SentenceTransformer(model_name, trust_remote_code=True).cuda() if device == 'cuda' else SentenceTransformer(model_name, trust_remote_code=True)
        
    def embed_documents(self, documents):
        """
        Embed a list of documents (sentences or paragraphs) using the SentenceTransformer model.
        Parameters:
        documents (list of str): List of text documents to embed.
        Returns:
        numpy.ndarray: Array of embeddings.
        """
        # Use the model to encode the documents
        embeddings = self.model.encode(documents)
        # Convert embeddings to numpy array
        return embeddings
    
    def embed_query(self, query, task=None, mode='query'):
        """
        Embed a single query (sentence) using the SentenceTransformer model.
        Parameters:
        query (str): The query to embed.
        task (str): Embeddings prompt intructions 
        mode (str): Embeddings prompt mode for models that don't take task
        Returns:
        numpy.ndarray: The query embedding.
        """
        # Some model requiere a task
        if task:
            embeddings = self.model.encode(f'Instruct: {task}\nQuery: {query}')
        # Some models requiere a prompt name
        elif mode:
            embeddings = self.model.encode(query, prompt_name=embeddings_query_prompt(mode))
        # Convert embedding to numpy array
        return embeddings



### VECTOR DATABASE MANAGMENT CLASS ###
### VECTOR DATABASE MANAGMENT CLASS ###
### VECTOR DATABASE MANAGMENT CLASS ###
class LangVdb:
    '''VECTOR DATABASE MANAGMENT CLASS'''
    
    _vdb : Literal['qdrant'] = vdb
    
    _pdf_parsers = {
        "llama_parse": LlamaParse(
                       api_key=os.environ.get('LLAMA_CLOUD_API_KEY'),
                       result_type="text",  # "markdown" and "text" are available
                       num_workers=2,  # if multiple files passed, split in `num_workers` API calls
                       verbose=True,
                       language="en",  # Optionally you can define a language, default=en
                       ),
        "langchain_pdfminer": PDFMinerLoader,
        "langchain_pdfplumber": PDFPlumberLoader
        }
    
    ### USE LOCAL ON SIDK STORAGE FOR NOW, SWITCH TO DOCKER SERVER LATER ### 
    ### USE LOCAL ON SIDK STORAGE FOR NOW, SWITCH TO DOCKER SERVER LATER ###
    ### USE LOCAL ON SIDK STORAGE FOR NOW, SWITCH TO DOCKER SERVER LATER ###
    @staticmethod
    def _initialize_vdb_collection(user_id):        
        if LangVdb._vdb == 'qdrant':
            LangVdb._init_qdrant_collection(user_id)

    @staticmethod
    def _init_qdrant_collection(user_id):
        if not QdrantClient(path=qdrant_database_path).collection_exists(collection_name=user_id):
            QdrantClient(path=qdrant_database_path).create_collection(
                collection_name=user_id,
                vectors_config={"dense": VectorParams(size=vector_dimensions, distance=Distance.COSINE)},
                sparse_vectors_config={"sparse": models.SparseVectorParams()}
            )  
             
    @staticmethod
    def access_vdb(user_id):
        if LangVdb._vdb == 'qdrant':
            return LangVdb._access_qdrant_vdb(user_id)
            
    @staticmethod
    def _access_qdrant_vdb(user_id):
        # Build vector store parameter dictionnary
        vector_store_params = {
            'path': qdrant_database_path,
            'collection_name':user_id,
            'embedding': SentenceTransformersEmbeddings() if retrieval_mode in ['dense', 'hybrid'] else None,
            'vector_name': 'dense', 
            'sparse_embedding': FastEmbedSparse(model_name=sparse_embeddings_model_name) if retrieval_mode in ['sparse', 'hybrid'] else None,
            'sparse_vector_name': 'sparse',
            'retrieval_mode': getattr(RetrievalMode, retrieval_mode.upper())
        }
        # Create collection if doesnt exist yet
        if not QdrantClient(path=qdrant_database_path).collection_exists(collection_name=user_id):
            LangVdb._initialize_vdb_collection(user_id)
        return QdrantVectorStore.from_existing_collection(**vector_store_params)
        
         
        
    ### TEXT ENTRY HANDLER METHODS ###
    ### TEXT ENTRY HANDLER METHODS ###
    ### TEXT ENTRY HANDLER METHODS ###
    @staticmethod
    def text_to_db(user_id, vdb, 
                   single_entry_load_kwargs:Dict=None, text_file_load_kwargs:Dict=None,
                   vdb_add=True, sqlite_add=True
                   ):
        # Format text entry
        if single_entry_load_kwargs:
            entry = LangVdb.format_text_entry(user_id, **single_entry_load_kwargs)
            docs = LangVdb.formatted_text_entries_to_docs([entry])
        # Format entries from text file
        elif text_file_load_kwargs:
            entries = LangVdb.txt_file_to_formatted_entries(user_id, **text_file_load_kwargs)
            docs = LangVdb.formatted_text_entries_to_docs(entries)
        
        # Add in vdb
        if vdb_add:
            LangVdb._index_docs_to_vdb(user_id, vdb, docs, source_id_key=entry_id_col_name)
        # Add in sqlite
        if sqlite_add:
            LangVdb._docs_to_sqlite(user_id, docs)
    
    @staticmethod
    def format_text_entry(user_id, text_date, main_tags, sub_tags, text_entry):
        """FORMAT SINGLE ENTRY (USER TEXT) TO SQL OR VDB FORMAT"""
        # Get doc's id and increment
        entry_id = YamlManagment.get_entry_id_and_increment(user_id)

        return {     
            entry_id_col_name: entry_id,   
            date_col_name: text_date,
            main_tags_col_name: main_tags,
            sub_tags_col_name: sub_tags,
            doc_type_col_name: 'text',
            text_col_name: text_entry,
        }

    @staticmethod
    def txt_file_to_formatted_entries(user_id, file_path, entry_delimiter, file_tags_separator, date_delimiter, main_tags_delimiter, sub_tags_delimiter, text_delimiter):
        """FORMAT ENTRIES (USER TEXT) FROM TEXT FILE TO SQL OR VDB FORMAT"""
        def _process_entry(entry):
            date, text = '', ''
            main_tags, sub_tags = [], []

            # Split the entry into lines
            lines = entry.strip().split('\n')

            for i in range(len(lines)):
                # Remove date_delimiter as well as white spaces and grab the date
                if date_delimiter in lines[i]:
                    date = lines[i].replace(date_delimiter, '').replace(' ', '')
                # Remove main_tags delimiter, white spaces and split tags by their file_tags_separator
                elif main_tags_delimiter in lines[i]:
                    main_tags = lines[i].replace(main_tags_delimiter, '').split(file_tags_separator)
                # Remove sub_tags delimiter, white spaces and split tags by their file_tags_separator  
                elif sub_tags_delimiter in lines[i]:
                    sub_tags = lines[i].replace(sub_tags_delimiter, '').split(file_tags_separator)
                # Remove text_delimiter and grab the text
                elif text_delimiter in lines[i]:
                    text = ' '.join(lines[i:]).replace(text_delimiter, '')
            # Format entry
            return LangVdb.format_text_entry(user_id, date, main_tags, sub_tags, text)

        # Open and read the file
        with open(file_path, 'r') as file:
            content = file.read()

        # Split the content by the delimiter
        entries = content.split(entry_delimiter)
        # Process each entry to retrieve data
        return [_process_entry(entry) for entry in entries]


    @staticmethod
    def formatted_text_entries_to_docs(formatted_entries:List):
        """CONVERT VDB FORMATTED TEXT ENTRIES TO DOCUMENTS"""
        # Get non empty entries
        entries = [entry for entry in formatted_entries if entry.get(text_col_name, None)]
        
        if not entries: 
            return None
        
        # Generate chunked Documents from formatted entries
        formatted_chunked_documents = [] 
        for entry_dict in entries:
            # Create Document with metadata
            generated_document = Document(
                page_content=entry_dict[text_col_name], 
                metadata={
                    entry_id_col_name: entry_dict[entry_id_col_name],
                    date_col_name: entry_dict[date_col_name], 
                    main_tags_col_name: entry_dict[main_tags_col_name],
                    sub_tags_col_name: entry_dict[sub_tags_col_name],
                    doc_type_col_name: entry_dict[doc_type_col_name]
                }
            )
            # Chunk initial document
            chunked_docs = recursive_character_text_splitter.split_documents([generated_document])
            # Add chunk ID in chunked document's metadata
            for i, doc in enumerate(chunked_docs, start=0):
                doc.metadata[chunked_entry_id_col_name] = f"{entry_dict[entry_id_col_name]}.{i}"
                formatted_chunked_documents.append(doc)
        # Return created Documents   
        return formatted_chunked_documents
   

    ### PDF HANDLER METHODS ###
    ### PDF HANDLER METHODS ###
    ### PDF HANDLER METHODS ###
    @staticmethod
    def pdf_to_db(user_id, vdb, pdf_path, pdf_date, main_tags, sub_tags, vdb_add=True, sqlite_add=True):
        """CONVERT PDF FILE TO DOCUMENT ENTRIES AND INDEX THESE IN VECTOR DATABASE"""
        
        # Parse pdf and get chunked documents
        chunked_docs = LangVdb._parse_PDF(pdf_path)
        # Get doc's id and increment
        doc_id = YamlManagment.get_entry_id_and_increment(user_id)
        # Format documents
        docs = LangVdb._format_chunked_docs(doc_id, chunked_docs, pdf_date, main_tags, sub_tags, doc_type = 'pdf')
        # Index in vdb
        if vdb_add:
            LangVdb._index_docs_to_vdb(user_id, vdb, docs)
        # Add in sqlite
        if sqlite_add:
            LangVdb._docs_to_sqlite(user_id, docs)
        # Add pdf in yml user's pdf list
        YamlManagment._add_user_pdf(user_id, os.path.basename(pdf_path), doc_id)
        
        
    @staticmethod
    def _parse_PDF(pdf_path, parser_type:str = 'langchain_pdfminer'):
        # Parse PDF
        try:
            # Get selected PDF parser
            parser = LangVdb._pdf_parsers[parser_type]
            # If using Llamaindex's llama_parse
            if parser_type == "llama_parse":
                parsed_pdf = parser.load_data(pdf_path) 
                # Load parsing output in temporary file 
                txt_file_path = 'output.txt'
                with open(txt_file_path, 'w', encoding='utf-8') as f:
                    for doc in parsed_pdf:
                        f.write(doc.text + '\n')
                # Load txt file with TextLoader
                loader = TextLoader(txt_file_path, autodetect_encoding=True)
                documents = loader.load()
                # Delete the file
                os.remove(txt_file_path)
            # Else using one of langchain's pdf loader
            else:
                loader = parser(pdf_path,
                                concatenate_pages=True)
                documents = loader.load()
            # Split loaded documents into chunks
            return recursive_character_text_splitter.split_documents(documents)

        except Exception as e:
            print(f"_parse_PDF fail: {e}")



    ### METHODS COMMON TO TEXT AND PDF HANDLERS (FORMAT DOCS + INDEX IN VDB + LOAD IN SQLITE) ###
    ### METHODS COMMON TO TEXT AND PDF HANDLERS (FORMAT DOCS + INDEX IN VDB + LOAD IN SQLITE) ###
    ### METHODS COMMON TO TEXT AND PDF HANDLERS (FORMAT DOCS + INDEX IN VDB + LOAD IN SQLITE) ###
    @staticmethod
    def _format_chunked_docs(entry_id, chunked_docs:list, doc_date, main_tags, sub_tags, doc_type):
        '''Format chunked documents' metadata'''
        for i, doc in enumerate(chunked_docs, start=0):
            # Add metadata
            doc.metadata[entry_id_col_name] = f"{entry_id}"
            doc.metadata[chunked_entry_id_col_name] = f"{entry_id}.{i}"
            doc.metadata[date_col_name] = doc_date
            doc.metadata[main_tags_col_name] = main_tags
            doc.metadata[sub_tags_col_name] = sub_tags
            doc.metadata[doc_type_col_name] = doc_type
        return chunked_docs
         
    @staticmethod
    def _index_docs_to_vdb(user_id, vdb, docs, source_id_key="source"):
        '''Index documents entry in vector store'''
        # Get user's vector store entries record manager
        namespace = f"{LangVdb._vdb}/{user_id}"
        record_manager = SQLRecordManager(
            namespace, db_url=sql_record_manager_path
        )
        record_manager.create_schema()
        # Index documents in vector store
        index(
            docs,
            record_manager,
            vdb,
            cleanup="incremental",
            source_id_key=source_id_key,
        )

    @staticmethod
    def _format_tags_list_to_str_sql_entry(tags_list):
        '''Format tags for sqlite entry'''
        return f'{sqlite_tags_separator}'.join([tag.replace(' ', '') for tag in tags_list]) if isinstance(tags_list, List) else tags_list
    
    @staticmethod
    def _docs_to_sqlite(user_id, docs):
        '''Add documents in sqlite database'''
        list_entries = [
            {
                entry_id_col_name: doc.metadata[entry_id_col_name],   
                chunked_entry_id_col_name: doc.metadata[chunked_entry_id_col_name],  
                date_col_name: doc.metadata[date_col_name],
                main_tags_col_name: LangVdb._format_tags_list_to_str_sql_entry(doc.metadata[main_tags_col_name]),
                sub_tags_col_name: LangVdb._format_tags_list_to_str_sql_entry(doc.metadata[sub_tags_col_name]),
                doc_type_col_name: doc.metadata[doc_type_col_name],
                text_col_name: doc.page_content,
            } for doc in docs
        ]
        SQLiteManagment.add_entry_to_sqlite(user_id, multiple_entries=list_entries)



### RAG CLASS ###
### RAG CLASS ###
### RAG CLASS ###
class RAGentic():
    """
    A class to implement a Retrieval-Augmented Generation (RAG) system using a local Chat Language Model (LLM), local embedding model and local vector database (VDB).

Initialization:
    - The class initializes with a local language model (LLM) using ChatOllama.
    - It sets up various components for chat history, document retrieval, engeenered system prompt templates and chains.

Flexible Parameter Adjustment:
    - Allows modification of LLM parameters on the fly.
    
Engeenered system prompts and corresponding chains:
    - Contextualization of query: A prompt and chain are set up to contextualize the user's query based on previous chat history.
    - Document ID retrieval: A prompt and chain to specifically identify which documents from the retrieved set the LLM would use to answer the contextualized query.
    - RAG System Prompt: Core prompt and chain responsible for integrating the retrieved documents with the LLM's response generation process.
    
RAG Chain:
    - The core RAG functionality is implemented in the rag_call method.
    - It contextualizes the query, retrieves relevant documents, determines which documents are used to answer the query and then uses these documents to generate a response using the LLM.

Chat History Management:
    - The class maintains a chat history and can trim it to manage token limits.

Query Contextualization:
    - The _chat_history_contextualize_human_query method uses the chat history to contextualize the user's query, potentially making it more relevant and specific.

Document Retrieval:
    - The retrieval method is responsible for retrieving relevant documents based on a query.
    - It supports dense, sparse and hybrid retrieval mode.
    - It supports different search types (e.g., similarity threshold, MMR) and can apply filters to the search.
    - The method can use reranking techniques like FlashRank or RAG Fusion to improve retrieval results.
    - Supports filtering of retrieved documents based on metadata, with adjustable strictness.
    - Can format retrieval results in different ways (string, RAG format, or raw documents).
    
Reranking Techniques:
    - Implements FlashRank and Reciprocal Ranking Fusion for potentially improving retrieval quality.

Document Usage Tracking:
    - The class tracks which retrieved documents are actually used in generating the response.

Streaming Support:
    - Supports streaming of LLM responses through callback mechanisms.



    Attributes:
    -----------
    llm : ChatOllama
        An instance of the local Chat Language Model used to generate responses.
    k_docs : int
        The number of documents to retrieve from the vector database.
    chat_dict : dict
        Dictionary to store incremental chat strings for app display.
    chat_history : InMemoryChatMessageHistory
        An object to maintain the chat history for the session.
    rag_retrieved_docs : str
        Stores the formatted string of documents retrieved in the last RAG call.
    rag_retrieved_doc_IDs : str
        Stores the string of document IDs retrieved in the last RAG call.
    rag_docs_used_to_generate_llm_response : str
        Stores the document IDs that were actually used to generate the final LLM response of the last RAG call.

    Prompts & Chains:
    -----------------
    chat_history_contextualize_q_prompt : ChatPromptTemplate
        An engeenered system prompt used to contextualize user queries based on chat history.
    chat_history_contextualize_q_chain : Pipeline
        A pipeline to contextualize the query by integrating chat history and generating the final LLM response.
        
    docs_used_in_response_system_prompt : ChatPromptTemplate
        An engeenered prompt to identify the document IDs that the LLM will use to answer the query.
    list_docs_used_in_response_chain : Pipeline
        A pipeline to list the document IDs from the retrieved documents that will be used in generating the response.
        
    rag_prompt : ChatPromptTemplate
        The engeenered system prompt for the final RAG process, incorporating both the query and the retrieved documents.
    rag_chain : Pipeline
        The final RAG pipeline that processes the contextualized query with the retrieved documents to generate a response.

    Methods:
    --------
    _get_chat_history_content():
        Extracts and returns the content from the chat history messages.
        
    _modify_llm_params(params: Dict):
        Dynamically updates the LLM parameters with the provided dictionary.
        
    retrieval(query, lang_vdb, search_type='similarity', k_outputs=None, rerank=False, filter_strictness='strict', filters={}, format_results=None):
        Retrieves documents from the vector database based on the query and provided parameters.
        
    _convert_filters_to_qdrant_filter(filters: Dict, filter_strictness: str) -> Union[Filter, None]:
        Converts the given filters to a format compatible with the Qdrant vector database.
        
    reranking(rerank, base_retriever, query):
        Executes the re-ranking process based on the selected method (FlashRank or RAG Fusion).
        
    _flashrank_reranking(base_retriever, query):
        Performs re-ranking of documents using the FlashRank algorithm.
        
    _reciprocal_ranking_fusion(multi_query_results: List[List], k=60):
        Merges results from multiple queries using reciprocal ranking fusion.
        
    _get_multi_query_llm_chain():
        Constructs the chain responsible for generating multiple queries from the original query.
        
    _build_rag_fusion_chain(base_retriever):
        Builds the RAG fusion chain that includes multiple queries and result merging.
        
    _get_k_best_results(retrieved_docs, k_outputs):
        Returns the top `k_outputs` documents from the retrieved documents list.
        
    _retrieval_results_str_format(retrieved_docs):
        Formats the retrieved documents into a string format suitable for display or further processing.
        
    _retrieval_results_to_rag_format(retrieved_docs, print_output=False):
        Formats the retrieved documents into a string format specifically for RAG processing.
        
    _build_str_retrieved_doc_IDs(retrieved_docs, print_output=False):
        Constructs a string of document IDs from the retrieved documents.
        
    _trim_chat_history(chat_history=None):
        Trims the chat history to fit within a specified token limit, prioritizing the most recent exchanges.
        
    _chat_history_contextualize_human_query(human_query: str, temp_llm: int = 0, print_output: bool = True):
        Contextualizes the human query by integrating relevant chat history and adjusting the LLM's temperature for processing.
        
    _which_docs_used_in_rag_response(human_query: str, retrieved_docs_rag: str, temp_llm: int = 0):
        Determines and returns the document IDs used by the LLM to generate the final response.
        
    rag_call(lang_vdb, human_query, llm_params, retrieval_params, streaming_callback_llm_response=None):
        Orchestrates the complete RAG process: contextualizes the query, retrieves documents, determines the used documents, and generates the final response.

    Usage:
    ------
    Instantiate the `RAGentic` class and utilize the `rag_call` method to perform retrieval-augmented generation.
    Ensure the proper setup of the language model (LLM) and vector database (VDB) prior to use.

    Example:
    --------
    ```python
    rag = RAGentic()
    response = rag.rag_call(lang_vdb, "What is the Schrödinger Equation?", llm_params={}, retrieval_params={})
    print(response)
    ```
    """
    def __init__(self):
        # Local Chat LLM
        self.llm = ChatOllama(
            model=llm_name,
            temperature=llm_temperature
        )
        
        # n docs retrieval
        self.k_docs = k_outputs_retrieval
        
        # Chat strings for incremental display in app
        self.chat_dict = {}
        
        # Chat history
        self.chat_history = InMemoryChatMessageHistory()
        
        # Retriever to use
        self.rag_retrieved_docs = ''
        # Last retrieved_doc_IDs_str
        self.rag_retrieved_doc_IDs = ''
        # Last retrieved documents used to formulate rag response
        self.rag_docs_used_to_generate_llm_response = ''
        
        ############################################################################
        # Chat history query contextualization system prompt
        self.chat_history_contextualize_q_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", chat_history_contextualize_q_system_prompt),
                MessagesPlaceholder("chat_history"),
                #("human", "{human_query}"),
            ]
        )
        # Chat history query contextualization chain
        self.chat_history_contextualize_q_chain = (
            self.chat_history_contextualize_q_prompt
            | self.llm
            | StrOutputParser()
        )
        ############################################################################
        # Documents IDs (from retrieved documents) that the llm use to answer the query
        self.docs_used_in_response_system_prompt = ChatPromptTemplate.from_template(doc_ids_used_in_rag_response_system_prompt)
        # Retrieved data relevant to answer the query chain
        self.list_docs_used_in_response_chain = (
            self.docs_used_in_response_system_prompt
            | self.llm
            | StrOutputParser()
        )
        ############################################################################
        # RAG system prompt
        self.rag_prompt = ChatPromptTemplate.from_template(rag_system_prompt)
        # RAG call with contextualized query + retrieved documents
        self.rag_chain = (
            self.rag_prompt
            | self.llm 
            | StrOutputParser()
        )
        
    def _get_chat_history_content(self):
        return [msg.content for msg in self.chat_history.messages]
    
    def _modify_llm_params(self, params:Dict): 
        for param_name, param_value in params.items():
            try:    
                setattr(self.llm, param_name, param_value)
            except Exception as e:
                print(f"_modify_llm_params fail for param {param_name}: {e}")
                  
    def retrieval(self, query, lang_vdb,
                  search_type:str=retrieval_search_type, k_outputs:int=k_outputs_retrieval, rerank:Literal['flashrank', 'rag_fusion', False]=retrieval_rerank,
                  filter_strictness=filter_strictness_choices[0], filters:Dict={}, 
                  format_results:Literal['str', 'rag', False, None]=False):

        # Set n as number of documents to retrieve
        self.k_docs = k_outputs if k_outputs else self.k_docs
        # Build initial search_kwargs
        search_kwargs = {'k': self.k_docs}
        # kwargs specific to "similarity_score_threshold" and "mmr"
        if search_type == "similarity_score_threshold":
            search_kwargs['score_threshold'] = relevance_threshold
        elif search_type == "mmr":
            search_kwargs['fetch_k'] = mmr_fetch_k
            search_kwargs['lambda_mult'] = mmr_lambda_mult     
            
        # Metadata embeddings filtering      
        if filters:
            # If there is filters value
            if any([value for key, value in filters.items()]):
                # Convert filter for qdrant
                if LangVdb._vdb == 'qdrant':
                    filters = self._convert_filters_to_qdrant_filter(filters, filter_strictness)
                # Add filter in search_kwargs
                search_kwargs['filter'] = filters     

        # Get retrieval engine from vector database
        retriever = lang_vdb.as_retriever(
            search_type=search_type,
            search_kwargs=search_kwargs
            )
       
        # Get retriever results using reranking (flashrank or rag fusion)
        if rerank:
            reranked_docs = self.reranking(rerank, retriever, query)
            retrieved_docs = self._get_k_best_results(reranked_docs, self.k_docs) 
        # Get retriever results
        else:
            retrieved_docs = retriever.get_relevant_documents(query)
        
        if format_results == 'str':  
            return self._retrieval_results_str_format(retrieved_docs), 
        elif format_results == 'rag':
            return self._retrieval_results_to_rag_format(retrieved_docs)
        else:
            return retrieved_docs

    def _convert_filters_to_qdrant_filter(self, filters: Dict, filter_strictness:str) -> Union[Filter, None]:
        # Return None if there if no filters
        if not filters:
            return None
        
        # For now we only add filter for main_tags and sub_tags (which are lists)
        available_filters = [main_tags_col_name, sub_tags_col_name, entry_id_col_name]
        must_conditions = []

        def _create_field_condition(filter_name: str, filter_value: str, value_or_any:str, field_is_list:bool=True) -> 'FieldCondition':
            # Field metadata key 
            metadata_key = f"metadata.{filter_name}[]" if field_is_list else f"metadata.{filter_name}"
            # Return FieldCondition
            if value_or_any == 'match_value':
                return FieldCondition(key=metadata_key, match=MatchValue(value=filter_value))
            elif value_or_any == 'match_any':
                return FieldCondition(key=metadata_key, match=MatchAny(any=filter_value))
            
        for filter_name, filter_value in filters.items():
            # Check if filter_name is available and filter_value is a non empty List
            if filter_name in available_filters and isinstance(filter_value, list) and filter_value:
                
                # If filter based on pdf file ID
                if filter_name == entry_id_col_name:
                    # Must match one of the provided IDs
                    must_conditions.append(_create_field_condition(filter_name, filter_value, 'match_any', field_is_list=False))
                            
                # Else filter based on TAGs
                else:    
                    # If strictness = any tags
                    if filter_strictness == filter_strictness_choices[0]:
                        # Must match at least one of the values in filter
                        must_conditions.append(_create_field_condition(filter_name, filter_value, 'match_any'))

                    # Elif all tags must be present
                    elif filter_strictness == filter_strictness_choices[1]:
                        # iterate over each tag and check if all of these are present in metadata
                        for value in filter_value:
                            must_conditions.append(_create_field_condition(filter_name, value, 'match_value'))
            
        return Filter(must=must_conditions)


    def reranking(self, rerank:Literal['flashrank', 'rag_fusion'], base_retriever, query):
        if rerank == 'flashrank':
            return self._flashrank_reranking(base_retriever, query)
            
        elif rerank == 'rag_fusion':
            rag_fusion_chain = self._build_rag_fusion_chain(base_retriever)
            return rag_fusion_chain.invoke({"original_query": query})
        
    def _flashrank_reranking(self, base_retriever, query):
        # Perform reranking with FlashRank
        compressor = FlashrankRerank(top_n=self.k_docs)
        compression_retriever = ContextualCompressionRetriever(
            base_compressor=compressor, base_retriever=base_retriever
        )       
        retrieved_docs = compression_retriever.get_relevant_documents(query)
        return retrieved_docs
    
    def _reciprocal_ranking_fusion(self, multi_query_results: List[List], k=60):    
        fused_scores, map_id_to_doc = {}, {}
         
        for docs in multi_query_results:
            # Assumes the docs are returned in sorted order of relevance (as expected since we provide outputs of a retriever)
            for rank, doc in enumerate(docs):
                doc_id = doc.metadata[entry_id_col_name]
                map_id_to_doc[doc_id] = doc
                if doc_id not in fused_scores:
                    fused_scores[doc_id] = 0
                fused_scores[doc_id] += 1 / (rank + k)

        reranked_docs = []
        for doc_id, score in sorted(fused_scores.items(), key=lambda x: x[1], reverse=True):
            doc = map_id_to_doc[doc_id]
            reranked_docs.append(doc)

        return reranked_docs
    
    def _get_multi_query_llm_chain(self):
        
        class LineQueryParser(BaseOutputParser[List[str]]):
                #Output parser for a list of lines: will split the LLM user' query rewrite into a list of queries
                def parse(self, text: str) -> List[str]:
                    lines = text.strip().split("\n")
                    return list(filter(None, lines))  # Remove empty lines
        # llm output parser to generate queries
        query_parser = LineQueryParser()
        # Multi query prompt for llm original query rewrite
        query_prompt_template = PromptTemplate(
            input_variables=["original_query"],
            template=multi_query_prompt,
        )
        # Chain to generate multi queries
        llm_multi_query_chain = (
            query_prompt_template | self.llm | query_parser
        )
        return llm_multi_query_chain
    
    def _build_rag_fusion_chain(self, base_retriever):
        return self._get_multi_query_llm_chain() | base_retriever.map() | self._reciprocal_ranking_fusion
        
        
    def _get_k_best_results(self, retrieved_docs, k_outputs):
        return retrieved_docs[:k_outputs] if len(retrieved_docs) > k_outputs else retrieved_docs
    
    
    def _retrieval_results_str_format(self, retrieved_docs):
        return f"\n{'-'*100}\n".join(
                    [
                        f"Document {i+1}:\n{doc.page_content}\n" +\
                        f"Metadata:\n" + "\n".join([f"{key}: {str(value)}" for key, value in doc.metadata.items()])
                        for i, doc in enumerate(retrieved_docs)
                    ]
                )
    
    def _retrieval_results_to_rag_format(self, retrieved_docs, print_output=False):
        # Format retrieved documents in rag format
        formatted_results = f"\n\n{'-'*20}\n\n".join(
                                [f"**Document ID {doc.metadata[chunked_entry_id_col_name]}**:\n{doc.page_content}" for doc in retrieved_docs]
                            )
        # Print formatted results if necessary
        if print_output:
            print(f'RETRIEVED DOCS:\n{formatted_results}\n\n')
        return formatted_results
        
    def _build_str_retrieved_doc_IDs(self, retrieved_docs, print_output=False):
        # Format retrieved doc IDs
        retrieved_doc_IDs = ", ".join([doc.metadata[chunked_entry_id_col_name] for doc in retrieved_docs]) if retrieved_docs else 'No retrieved documents.'
        # Print if necessary
        if print_output:
            print(f'RETRIEVED DOC IDS:\n{retrieved_doc_IDs}\n\n')
        return retrieved_doc_IDs
    
    def _trim_chat_history(self, chat_history=None):
        messages = chat_history if chat_history else self.chat_history.messages
        if not messages:
            return []
        return trim_messages(
                messages = messages,
                max_tokens=max_chat_history_tokens,
                strategy="last",
                token_counter=self.llm,
                include_system=True,
                allow_partial=False,
                start_on="human",
            )
        
    def _chat_history_contextualize_human_query(self, human_query:str, temp_llm:int=0, print_output:bool=True):
        # Get current llm temperature
        original_temperature = copy.copy(getattr(self.llm, 'temperature'))
        # Set llm temperature to temp_llm
        self._modify_llm_params({'temperature' : temp_llm})
        # Trim the chat history before passing it to the chain
        trimmed_history = self._trim_chat_history(self.chat_history.messages)
        # Format the chat_history_contextualize_q_prompt with variables
        formatted_prompt = self.chat_history_contextualize_q_prompt.format(
            chat_history=trimmed_history, 
            human_query=human_query
            )
        # Call LLM with the formatted prompt to generate new query
        chat_history_contextualized_human_query = self.llm.invoke(formatted_prompt)
        # Print if necessary
        if print_output:
            print(f'\n\nRECONTEXT HUMAN QUERY: {chat_history_contextualized_human_query.content}\n\n')      
        # Set temperature to original value
        self._modify_llm_params({'temperature' : original_temperature})
        # Return contextualized query
        return chat_history_contextualized_human_query.content

    def _which_docs_used_in_rag_response(self, human_query:str, retrieved_docs_rag:str, temp_llm:int=0):
        # Get current llm temperature
        original_temperature = copy.copy(getattr(self.llm, 'temperature'))
        # Set llm temperature to temp_llm
        self._modify_llm_params({'temperature' : temp_llm})
        # Chain call
        used_docs = self.list_docs_used_in_response_chain.invoke({
            "question": human_query,
            "context": retrieved_docs_rag
        })
        # Set temperature to original value
        self._modify_llm_params({'temperature' : original_temperature})
        # Return contextualized query
        return used_docs
    
    def rag_call(self, lang_vdb, human_query, llm_params, retrieval_params, streaming_callback_llm_response=None):
        # Update llm params
        if isinstance(llm_params, Dict) and llm_params:
            self._modify_llm_params(llm_params)
            
        ## Contextualize query based on chat history
        contextualized_human_query = self._chat_history_contextualize_human_query(human_query, print_output=True)
        
        # Retrieve documents associated to the contextualized query
        retrieved_documents = self.retrieval(
            contextualized_human_query, 
            lang_vdb, 
            **retrieval_params
        )
        
        # Format the retrieved documents in rag format
        self.rag_retrieved_docs = self._retrieval_results_to_rag_format(retrieved_documents, print_output=True)
        # Build a string of retrieved doc's IDs
        self.rag_retrieved_doc_IDs = self._build_str_retrieved_doc_IDs(retrieved_documents, print_output=True)

        # Determine which retrieved documents are used to formulate the answer
        # Will be added at the end of the llm answer via the LLMResponseStreamingHandlerTaipy's method --on_llm_end--
        self.rag_docs_used_to_generate_llm_response = self._which_docs_used_in_rag_response(contextualized_human_query, self.rag_retrieved_docs)
        
        # Activate streaming callback
        self._modify_llm_params({'callbacks' : CallbackManager([streaming_callback_llm_response] if streaming_callback_llm_response else None)})

        # Call rag
        ai_response_str = self.rag_chain.invoke({
            "question": contextualized_human_query,
            "context": self.rag_retrieved_docs
        })

        # Reset callbacks for normal llm usage 
        self._modify_llm_params({'callbacks':None})

        # Add human and ai messages in chat history
        self.chat_history.add_messages([HumanMessage(content=human_query), AIMessage(content=ai_response_str)])
        return 
        