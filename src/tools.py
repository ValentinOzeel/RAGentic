import os
import time
import regex as re
from typing import List, Dict, Union, Literal

from contextlib import contextmanager
import pandas as pd
import numpy as np

import yaml
import json
import sqlite3

import easyocr

from llama_parse import LlamaParse
from langchain_community.document_loaders import PDFPlumberLoader

from sentence_transformers import SentenceTransformer
from langchain_core.documents import Document
from langchain.embeddings.base import Embeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.indexes import SQLRecordManager, index
from langchain_community.document_loaders import TextLoader
## Vector databases
#from pymilvus import MilvusClient
from qdrant_client import QdrantClient, models
from qdrant_client.models import Distance, VectorParams, Filter, FieldCondition, MatchValue, MatchAny
## Langchain integreation
#from langchain_milvus.vectorstores import Milvus
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
from operator import itemgetter
from langchain.schema import StrOutputParser
## Messages
from langchain_core.messages import HumanMessage
from langchain_core.messages import AIMessage
## Callbacks (streaming repsonse)
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.base import BaseCallbackHandler

from constants import (credentials_yaml_path, 
                       image_to_text_languages,
                       entry_id_col_name, chunked_entry_id_col, date_col_name, main_tags_col_name, sub_tags_col_name, doc_type_col_name, text_col_name,
                       sqlite_database_path, sqlite_tags_separator,
                       chunk_size, chunk_overlap,
                       device,
                       vdb, milvus_database_path, qdrant_database_path, retrieval_mode, retrieval_rerank,
                       sql_record_manager_path, embeddings_model_name, embeddings_query_prompt, vector_dimensions, sparse_embeddings_model_name,
                       retrieval_search_type, filter_strictness_choices, k_outputs_retrieval, relevance_threshold, mmr_fetch_k, mmr_lambda_mult,
                       llm_name, llm_temperature, max_chat_history_tokens
)

from prompts import multi_query_prompt, chat_history_contextualize_q_system_prompt, rag_system_prompt
            

class YamlManagment():

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
        return entry_id 
            
            
            
                
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








class SQLiteManagment:
    '''
    In an application where multiple users can access the database, 
    it's generally better to open and close the database connection for each action. 
    SQLite, while lightweight and convenient, is not designed for high-concurrency scenarios like a full-fledged database server such as PostgreSQL or MySQL.
    That's why contextmanager is used here to close connect and close the db after action.
    '''

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
                            single_entry.get(date_col_name, None),
                            single_entry.get(main_tags_col_name, None),
                            single_entry.get(sub_tags_col_name, None),
                            single_entry.get(doc_type_col_name, None),
                            single_entry.get(text_col_name, None)
                        )
                        cursor.execute(f'''
                            INSERT INTO entries (user_id, {entry_id_col_name}, {date_col_name}, {main_tags_col_name}, {sub_tags_col_name}, {doc_type_col_name}, {text_col_name})
                            VALUES (?, ?, ?, ?, ?, ?, ?)
                        ''', values)
                        
                    elif multiple_entries:
                        entries_to_add = [
                            (
                                user_id,
                                entry.get(entry_id_col_name, None),
                                entry.get(date_col_name, None),
                                entry.get(main_tags_col_name, None),
                                entry.get(sub_tags_col_name, None),
                                entry.get(doc_type_col_name, None),
                                entry.get(text_col_name, None)
                            )
                            for entry in multiple_entries if not ignore_if_empty(entry)
                        ]
                        cursor.executemany(f'''
                            INSERT INTO entries (user_id, {entry_id_col_name}, {date_col_name}, {main_tags_col_name}, {sub_tags_col_name}, {doc_type_col_name}, {text_col_name})
                            VALUES (?, ?, ?, ?, ?, ?, ?)
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
                    SELECT id as text_id, {entry_id_col_name}, {date_col_name}, {main_tags_col_name}, {sub_tags_col_name}, {doc_type_col_name}, {text_col_name} 
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
        



        
        
        
def image_to_text_conversion(selected_languages, cpu_or_gpu, image_path):

    gpu = True if cpu_or_gpu == 'GPU' else False
    languages = [image_to_text_languages[selected_languages]] if isinstance(selected_languages, str) else [
        image_to_text_languages[language] for language in selected_languages
        ] 

    reader = easyocr.Reader(languages, gpu=gpu)
    result = reader.readtext(image_path)

    whole_text = ' '.join([text for (bbox, text, prob) in result])
    
    return whole_text








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
    
    def embed_query(self, query, task=None, mode='semantic'):
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







class LangVdb:
    _vdb : Literal['qdrant', 'milvus'] = vdb
    
    _pdf_parsers = {
        "llama_parse": LlamaParse(
                       api_key=os.environ.get('LLAMA_CLOUD_API_KEY'),
                       result_type="text",  # "markdown" and "text" are available
                       num_workers=2,  # if multiple files passed, split in `num_workers` API calls
                       verbose=True,
                       language="en",  # Optionally you can define a language, default=en
                       ),
        "langchain_pdfplumber": PDFPlumberLoader
        }
    #    ### USE LOCAL ON SIDK STORAGE FOR NOW, SWITCH TO DOCKER SERVER LATER ### 
    #    ### USE LOCAL ON SIDK STORAGE FOR NOW, SWITCH TO DOCKER SERVER LATER ###
    #    ### USE LOCAL ON SIDK STORAGE FOR NOW, SWITCH TO DOCKER SERVER LATER ###

        

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
        
        
        
        
    ### TEXT ENTRY HANDLER ###
    ### TEXT ENTRY HANDLER ###
    ### TEXT ENTRY HANDLER ###
    
    @staticmethod
    def format_text_entry(user_id, text_date, main_tags, sub_tags, text_entry, format:Literal['sqlite', 'vdb']='sqlite'):
        """
        FORMAT SINGLE ENTRY (USER TEXT) TO SQL OR VDB FORMAT
        """
        if format == 'sqlite':
            main_tags = f'{sqlite_tags_separator}'.join(main_tags) if isinstance(main_tags, List) else main_tags
            sub_tags = f'{sqlite_tags_separator}'.join(sub_tags) if isinstance(sub_tags, List) else sub_tags

        # Get doc's id and increment
        entry_id = YamlManagment.get_entry_id_and_increment(user_id)

        return {     
            entry_id_col_name: entry_id,   
            chunked_entry_id_col: f"{entry_id}.0",
            date_col_name: text_date,
            main_tags_col_name: main_tags,
            sub_tags_col_name: sub_tags,
            doc_type_col_name: 'text',
            text_col_name: text_entry,
        }

    @staticmethod
    def txt_file_to_formatted_entries(user_id, file_path, entry_delimiter, file_tags_separator, date_delimiter, main_tags_delimiter, sub_tags_delimiter, text_delimiter, 
                                      format:Literal['sqlite', 'vdb']='sqlite'):
        """
        FORMAT ENTRIES (USER TEXT) FROM TEXT FILE TO SQL OR VDB FORMAT
        """
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

            # Format list as str (sqlite cannot save list)
            if format == 'sqlite':
                 # Format tags to sqlite (string) 
                main_tags_str = f'{sqlite_tags_separator}'.join([tag.replace(' ', '') for tag in main_tags]) if format == 'sqlite' else main_tags
                sub_tags_str = f'{sqlite_tags_separator}'.join([tag.replace(' ', '') for tag in sub_tags]) if format == 'sqlite' else sub_tags
                return LangVdb.format_text_entry(user_id, date, main_tags_str, sub_tags_str, text, format=format)
            # Retain lsit format otherwise
            else:
                return LangVdb.format_text_entry(user_id, date, main_tags, sub_tags, text, format=format)


        # Open and read the file
        with open(file_path, 'r') as file:
            content = file.read()

        # Split the content by the delimiter
        entries = content.split(entry_delimiter)
        # Process each entry to retrieve data
        return [_process_entry(entry) for entry in entries]


    @staticmethod
    def formatted_text_entries_to_docs(formatted_entries:List):
        """
        CONVERT VDB FORMATTED TEXT ENTRIES TO DOCUMENTS
        """
        # Split text and create corresponding entries (and chunked_entry_id)
        splitted_formatted_entries = LangVdb._recursively_split_formatted_entries(formatted_entries)
        # Convert entries to Document
        return LangVdb._texts_to_documents(splitted_formatted_entries)
    
    @staticmethod
    def _recursively_split_formatted_entries(entries):
        def _create_new_entries(entry):
            # Split the entry text
            splitted_text = text_splitter.split_text(entry[text_col_name])
            # Create new entries with splitted text (while combining entry_id and child_id as "entry_id.child_id")
            return [
                {
                    entry_id_col_name: entry[entry_id_col_name],
                    chunked_entry_id_col: f"{entry[entry_id_col_name]}.{child_id}",
                    date_col_name: entry[date_col_name], 
                    main_tags_col_name: entry[main_tags_col_name],
                    sub_tags_col_name: entry[sub_tags_col_name],
                    doc_type_col_name: entry[doc_type_col_name],
                    text_col_name: text
                } for child_id, text in enumerate(splitted_text, start=1)
            ] 
               
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            is_separator_regex=False,
        )
        
        # Iterate over entries input, split each entry's text and accordingly create new entries
        # with splitted text while conserving original entry's metadata. The list of new entries is returned.
        return [new_entry for entry in entries for new_entry in _create_new_entries(entry)] if entries else None

    @staticmethod
    def _texts_to_documents(entries:Union[List[dict], dict, None]):
        if not entries:
            return None
        
        elif isinstance(entries, List):
            entries = [entry for entry in entries if entry.get(text_col_name, None)]
            return [
                Document(
                    page_content=entry_dict[text_col_name], 
                    metadata={
                        entry_id_col_name: entry_dict[entry_id_col_name],
                        chunked_entry_id_col: entry_dict[chunked_entry_id_col],
                        date_col_name: entry_dict[date_col_name], 
                        main_tags_col_name: entry_dict[main_tags_col_name],
                        sub_tags_col_name: entry_dict[sub_tags_col_name],
                        doc_type_col_name: entry_dict[doc_type_col_name]
                        }
                    ) for entry_dict in entries
                ] if entries else None

        else:
            return [
                Document(
                    page_content=entries[text_col_name], 
                    metadata={
                        entry_id_col_name: entries[entry_id_col_name],
                        chunked_entry_id_col: entries[chunked_entry_id_col],
                        date_col_name: entries[date_col_name], 
                        main_tags_col_name: entries[main_tags_col_name],
                        sub_tags_col_name: entries[sub_tags_col_name],
                        doc_type_col_name: entries[doc_type_col_name]
                        }
                    )
            ] if entries.get(text_col_name, None) else None


        

    ### PDF HANDLER ###
    ### PDF HANDLER ###
    ### PDF HANDLER ###
    
    @staticmethod
    def pdf_to_db(user_id, vdb, pdf_path, pdf_date, main_tags, sub_tags, vdb_add=True, sqlite_add=True):
        """
        CONVERT PDF FILE TO DOCUMENT ENTRIES AND INDEX THESE IN VECTOR DATABASE
        """
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
    def _parse_PDF(pdf_path, parser_type:str = 'langchain_pdfplumber'):
        
        try:
            # Parse PDF
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
                #[
                #    Document(
                    # page_content='---\nsidebar_position: 0\n---\nThey optionally implement:\n\n3. "Lazy load": load documents into memory lazily\n', 
                    # metadata={'source': '../docs/docs/modules/data_connection/document_loaders/index.md'})
                #]
                loader = TextLoader(txt_file_path, autodetect_encoding=True)
                documents = loader.load()
                # Delete the file
                os.remove(txt_file_path)
            # Else using one of langchain's pdf loader
            else:
                loader = parser(pdf_path)
                documents = loader.load()
            # Split loaded documents into chunks
            return RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap).split_documents(documents)

            
        except Exception as e:
            print(f"_parse_PDF fail: {e}")


    
    @staticmethod
    def _format_chunked_docs(entry_id, chunked_docs:list, doc_date, main_tags, sub_tags, doc_type):
            
        for i, doc in enumerate(chunked_docs, start=1):
            # Add metadata
            doc.metadata[entry_id_col_name] = f"{entry_id}.{i}"
            doc.metadata[date_col_name] = doc_date
            doc.metadata[main_tags_col_name] = main_tags
            doc.metadata[sub_tags_col_name] = sub_tags
            doc.metadata[doc_type_col_name] = doc_type

        return chunked_docs

                     
    @staticmethod
    def _index_docs_to_vdb(user_id, vdb, docs, source_id_key="source"):
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
    def _docs_to_sqlite(user_id, docs):
        list_entries = [
            {
                entry_id_col_name: doc.metadata[entry_id_col_name],   
                date_col_name: doc.metadata[date_col_name],
                main_tags_col_name: doc.metadata[main_tags_col_name],
                sub_tags_col_name: doc.metadata[sub_tags_col_name],
                text_col_name: doc.page_content,
            } for doc in docs
        ]
        SQLiteManagment.add_entry_to_sqlite(user_id, multiple_entries=list_entries)




        
    ### DOCUMENTS INDEXING IN VDB ###
    ### DOCUMENTS INDEXING IN VDB ###
    ### DOCUMENTS INDEXING IN VDB ###

    @staticmethod
    def add_docs_to_vdb(user_id, vdb, docs:List):
        """
        INDEX DOCUMENTS IN VECTOR DATABASE
        """
        if docs:
            ## Add documents in vector store
            #vdb.add_documents(documents=documents_entries)
            
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
                source_id_key=entry_id_col_name,
            )
 
 
 
 

#RAG:
#KEEP TRACK OF LOADED PDF FOR ADD FILTER "filename" in RAG
#ADD PDF DOCS IN SQL TOOO
#MOVE RETRIEVAL IN RAG CLASS


class RAGentic():
    
    def __init__(self):

        self.llm = ChatOllama(
            model=llm_name,
            temperature=llm_temperature
        )
        

        # Chat strings for incremental display in app
        self.chat_dict = {}
        
        # Chat history
        self.chat_history = InMemoryChatMessageHistory()
        
        self.chat_history_contextualize_q_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", chat_history_contextualize_q_system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{human_query}"),
            ]
        )
        # Create the chain
        self.chat_history_contextualize_q_chain = (
            self.chat_history_contextualize_q_prompt
            | self.llm
            | StrOutputParser()
        )
    
        self.rag_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", rag_system_prompt),
                ("human", "{retrieved_docs_rag}"),
                ("human", "{chat_history_contextualized_human_query}"),
            ]
        )
        
        self.rag_chain = (
            self.rag_prompt
            | self.llm 
            | StrOutputParser()
        )
        
        self.k_docs = k_outputs_retrieval


    def _get_chat_history_content(self):
        return [msg.content for msg in self.chat_history.messages]
    
    def _modify_llm_params(self, params:Dict): 
        for param_name, param_value in params.items():
            try:    
                setattr(self.llm, param_name, param_value)
            except Exception as e:
                print(f"_modify_llm_params fail for param {param_name}: {e}")
                
                
    def retrieval(self, query, lang_vdb,
                  llm_params:Dict=None, streaming_callback_llm_response:BaseCallbackHandler=None,
                  search_type:str=retrieval_search_type, k_outputs:int=k_outputs_retrieval, rerank:Literal['flashrank', 'rag_fusion', False]=retrieval_rerank,
                  filter_strictness=filter_strictness_choices[0], filters:Dict={}, 
                  format_results:Literal['str', 'rag', False, None]=False):

        self.k_docs = k_outputs if k_outputs else self.k_docs
        
        print(
            f'''
            \n\n\n
            query: {query},
            llm_params: {llm_params}
            search_type: {search_type}
            k_outputs: {self.k_docs}
            rerank: {rerank}
            \n\n\n
            '''
        )
        # Update llm params
        if isinstance(llm_params, Dict) and llm_params:
            self._modify_llm_params(llm_params)
        
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
            retrieved_docs = retriever.invoke(query)

        # Activate llm response streaming if callback provided
        if streaming_callback_llm_response:
            self._modify_llm_params({'callbacks':CallbackManager([streaming_callback_llm_response])})
            
        print('RETRIEVED DOCS:', self._retrieval_results_str_format(retrieved_docs))
            
        if format_results == 'str':  
            return self._retrieval_results_str_format(retrieved_docs) 
        elif format_results == 'rag':
            return self._retrieval_results_to_rag_format(retrieved_docs)
        else:
            return retrieved_docs

    def _convert_filters_to_qdrant_filter(self, filters: Dict, filter_strictness:str) -> Union[Filter, None]:
        
        def _create_field_condition(filter_name: str, filter_value: str, value_or_any:str) -> 'FieldCondition':
            if value_or_any == 'match_value':
                return FieldCondition(key=f"metadata.{filter_name}[]", match=MatchValue(value=filter_value))
            elif value_or_any == 'match_any':
                return FieldCondition(key=f"metadata.{filter_name}[]", match=MatchAny(any=filter_value))
        
        if not filters:
            return None
        
        # For now we only add filter for main_tags and sub_tags (which are lists)
        available_filters = [main_tags_col_name, sub_tags_col_name, entry_id_col_name]
        must_conditions = []
        
        
        for filter_name, filter_value in filters.items():
            if filter_name in available_filters and filter_value:
                # If strictness = any tags
                if filter_strictness == filter_strictness_choices[0]:
                    
                    if isinstance(filter_value, list):
                        # Must match at least one of the values in filter
                        must_conditions.append(_create_field_condition(filter_name, filter_value, 'match_any'))
                    # For other typer of filters such as date
                    #else:
                    #    must_conditions.append(_create_field_condition(filter_name, filter_value))

                # Elif all tags must be present
                elif filter_strictness == filter_strictness_choices[1]:
                    if isinstance(filter_value, list):
                        # iterate over each tag and check if all of these are present in metadata
                        for value in filter_value:
                            must_conditions.append(_create_field_condition(filter_name, value, 'match_value'))
            
        return Filter(must=must_conditions)
    

    def _flashrank_reranking(self, base_retriever, query):
        # Perform reranking with FlashRank
        compressor = FlashrankRerank(top_n=self.k_docs)
        compression_retriever = ContextualCompressionRetriever(
            base_compressor=compressor, base_retriever=base_retriever
        )       
        retrieved_docs = compression_retriever.invoke(query)
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
            template=multi_query_prompt ,
        )
        # Chain to generate multi queries
        llm_multi_query_chain = (
            query_prompt_template | self.llm | query_parser
        )
        return llm_multi_query_chain
    
    def _build_rag_fusion_chain(self, base_retriever):
        return self._get_multi_query_llm_chain() | base_retriever.map() | self._reciprocal_ranking_fusion
        
    def reranking(self, rerank:Literal['flashrank', 'rag_fusion'], base_retriever, query):
        if rerank == 'flashrank':
            return self._flashrank_reranking(base_retriever, query)
            
        elif rerank == 'rag_fusion':
            rag_fusion_chain = self._build_rag_fusion_chain(base_retriever)
            return rag_fusion_chain.invoke({"original_query": query})
        
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
    
    def _retrieval_results_to_rag_format(self, retrieved_docs):
        return "\n\n".join(
                [
                    f"Doc_ID: {doc.metadata[entry_id_col_name]}, Doc_content: {doc.page_content}"
                    for doc in retrieved_docs
            ]
        )

    
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
        
    def _chat_history_contextualize_human_query(self, human_query):
        # Trim the chat history before passing it to the chain
        trimmed_history = self._trim_chat_history(self.chat_history.messages)
        # Use self.history_contextualize_q_chain to generate new query
        return self.chat_history_contextualize_q_chain.invoke({
            "chat_history": trimmed_history,
            "human_query": human_query
        })
        
    
    def _rag_query(self, history_contextualized_query, lang_vdb, llm_params, retrieval_params, streaming_callback_llm_response=None):
        
        return self.rag_chain.invoke({
            "retrieved_docs_rag": self.retrieval(
                history_contextualized_query, 
                lang_vdb, 
                llm_params=llm_params, 
                streaming_callback_llm_response=streaming_callback_llm_response,
                **retrieval_params
            ),
            "chat_history_contextualized_human_query": history_contextualized_query
        })
        
    
    def send_user_query_to_rag(self, lang_vdb, human_query, llm_params, retrieval_params, streaming_callback_llm_response=None):


        chat_history_contextualized_human_query = self._chat_history_contextualize_human_query(human_query)
        
        print('\n\n\nRECONTEXT HUMAN QUERY :', chat_history_contextualized_human_query, '\n\n\n')
        
        ai_response = self._rag_query(
            chat_history_contextualized_human_query,
            lang_vdb,
            llm_params,
            retrieval_params,
            streaming_callback_llm_response=streaming_callback_llm_response
        )   
        
         
        self.chat_history.add_messages([HumanMessage(content=human_query), AIMessage(content=ai_response)])
        
        # Reset callbacks for normal llm usage 
        self._modify_llm_params({'callbacks':None})
        
        return ai_response
