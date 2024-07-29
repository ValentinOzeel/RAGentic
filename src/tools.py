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

from sentence_transformers import SentenceTransformer
from langchain_core.documents import Document
from langchain.embeddings.base import Embeddings
# Vector databases
from pymilvus import MilvusClient
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, Filter, FieldCondition, MatchValue, MatchAny, Match
# Langchain integreation
from langchain_milvus.vectorstores import Milvus
from langchain_qdrant import QdrantVectorStore
from langchain_qdrant import FastEmbedSparse, RetrievalMode
# Rerank
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import FlashrankRerank



from constants import (credentials_yaml_path, 
                       image_to_text_languages,
                       date_col_name, main_tags_col_name, sub_tags_col_name, text_col_name,
                       sqlite_database_path, sqlite_tags_separator,
                       device,
                       vdb, milvus_database_path, qdrant_database_path, retrieval_mode, retrieval_rerank_flag,
                       embeddings_model_name, embeddings_query_prompt, vector_dimensions, sparse_embeddings_model_name,
                       retrieval_search_type, filter_strictness_choices, k_outputs_retrieval, relevance_threshold, mmr_fetch_k, mmr_lambda_mult
)


def create_cred_yaml_file():
    #Initialize the YAML file with an empty dictionary.
    if not os.path.isfile(credentials_yaml_path):
        with open(credentials_yaml_path, 'w') as file:
            yaml.dump({'user_creds': {}}, file)
            
                
class SignLog():
    ###                 ###               
    ### Sign in methods ###
    ###                 ###
    @staticmethod
    def email_check(email_input:str):
        email_regex = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,7}\b'
        return True if re.match(email_regex, email_input) else False

    @staticmethod
    def send_email_as_verif(self):
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
    
    @staticmethod
    def add_user_credentials(email, password):
        '''
        Add user credentials upon account creation
        ''' 
        # Load yaml creadentials
        with open(credentials_yaml_path, 'r') as file:
            data = yaml.safe_load(file)  
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
        
    ###                 ###               
    ### Log in methods  ###
    ###                 ###
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
                    {date_col_name} TEXT,
                    {main_tags_col_name} TEXT,
                    {sub_tags_col_name} TEXT,
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
                            single_entry.get(date_col_name, None),
                            single_entry.get(main_tags_col_name, None),
                            single_entry.get(sub_tags_col_name, None),
                            single_entry.get(text_col_name, None)
                        )
                        cursor.execute(f'''
                            INSERT INTO entries (user_id, {date_col_name}, {main_tags_col_name}, {sub_tags_col_name}, {text_col_name})
                            VALUES (?, ?, ?, ?, ?)
                        ''', values)
                        
                    elif multiple_entries:
                        entries_to_add = [
                            (
                                user_id,
                                entry.get(date_col_name, None),
                                entry.get(main_tags_col_name, None),
                                entry.get(sub_tags_col_name, None),
                                entry.get(text_col_name, None)
                            )
                            for entry in multiple_entries if not ignore_if_empty(entry)
                        ]
                        cursor.executemany(f'''
                            INSERT INTO entries (user_id, {date_col_name}, {main_tags_col_name}, {sub_tags_col_name}, {text_col_name})
                            VALUES (?, ?, ?, ?, ?)
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
                    SELECT id as text_id, {date_col_name}, {main_tags_col_name}, {sub_tags_col_name}, {text_col_name} 
                    FROM entries 
                    WHERE user_id = ? 
                    ORDER BY id
                '''
                df = pd.read_sql_query(query, conn, params=(user_id,))
                df[date_col_name] = pd.to_datetime(df[date_col_name], format='%m/%d/%Y')
                df[main_tags_col_name] = df[main_tags_col_name].apply(lambda value: value.split(sqlite_tags_separator))
                df[sub_tags_col_name] = df[sub_tags_col_name].apply(lambda value: value.split(sqlite_tags_separator))
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



def format_entry(text_date, main_tags, sub_tags, text_entry, format:Literal['sqlite', 'vdb']='sqlite'):
    if format == 'sqlite':
        main_tags = f'{sqlite_tags_separator}'.join(main_tags) if isinstance(main_tags, List) else main_tags
        sub_tags = f'{sqlite_tags_separator}'.join(sub_tags) if isinstance(sub_tags, List) else sub_tags
    print('FORMAT ENTRY: ', {
        date_col_name: text_date,
        main_tags_col_name: main_tags,
        sub_tags_col_name: sub_tags,
        text_col_name: text_entry,
    }, '\n')
    return {
        date_col_name: text_date,
        main_tags_col_name: main_tags,
        sub_tags_col_name: sub_tags,
        text_col_name: text_entry,
    }

def txt_file_to_formatted_entries(file_path, entry_delimiter, file_tags_separator, date_delimiter, main_tags_delimiter, sub_tags_delimiter, text_delimiter, 
                                  format:Literal['sqlite', 'vdb']='sqlite'):
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
            return format_entry(date, main_tags_str, sub_tags_str, text, format=format)
        # Retain lsit format otherwise
        else:
            return format_entry(date, main_tags, sub_tags, text, format=format)


    # Open and read the file
    with open(file_path, 'r') as file:
        content = file.read()

    # Split the content by the delimiter
    entries = content.split(entry_delimiter)
    # Process each entry to retrieve data
    return [_process_entry(entry) for entry in entries]



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
    _vdb_client = None
    
    
    
    @staticmethod
    def initialize_vdb():
        ### USE LOCAL ON SIDK STORAGE FOR NOW, SWITCH TO DOCKER SERVER LATER ### 
        ### USE LOCAL ON SIDK STORAGE FOR NOW, SWITCH TO DOCKER SERVER LATER ###
        ### USE LOCAL ON SIDK STORAGE FOR NOW, SWITCH TO DOCKER SERVER LATER ###
        
        if LangVdb._vdb == 'milvus':
            LangVdb._vdb_client = MilvusClient(milvus_database_path)
            
        elif LangVdb._vdb == 'qdrant':
            LangVdb._vdb_client = QdrantClient(path=qdrant_database_path)

        
    @staticmethod
    def initialize_vdb_collection(user_id):        
        if LangVdb._vdb == 'milvus':
            LangVdb._init_milvus_collection(user_id)

        elif LangVdb._vdb == 'qdrant':
            LangVdb._init_qdrant_collection(user_id)

    @staticmethod
    def _init_milvus_collection(user_id):
        if not LangVdb._vdb_client.has_collection(collection_name=user_id):
            LangVdb._vdb_client.create_collection(
                collection_name=user_id,
                dimension=vector_dimensions,  # The dimension of vectors that will be stored
            )

    @staticmethod
    def _init_qdrant_collection(user_id):
        if not LangVdb._vdb_client.collection_exists(collection_name="{user_id}"):
            LangVdb._vdb_client.create_collection(
                collection_name=user_id,
                vectors_config=VectorParams(size=vector_dimensions, distance=Distance.COSINE),
            )

    @staticmethod
    def access_vdb(user_id):
        
        if LangVdb._vdb == 'milvus':
            return LangVdb._access_milvus_vdb(user_id)
            
        elif LangVdb._vdb == 'qdrant':
            return LangVdb._access_qdrant_vdb(user_id)
        


        
    @staticmethod
    def _access_milvus_vdb(user_id):
        return Milvus(
            SentenceTransformersEmbeddings(),
            connection_args={"uri": milvus_database_path},
            collection_name=user_id,
        )
        
    @staticmethod
    def _access_qdrant_vdb(user_id):
        return QdrantVectorStore(
            client=LangVdb._vdb_client,
            collection_name=user_id,
            embedding=SentenceTransformersEmbeddings() if retrieval_mode in ['dense', 'hybrid'] else None,
            sparse_embedding = FastEmbedSparse(model_name=sparse_embeddings_model_name) if retrieval_mode in ['sparse', 'hydrid'] else None,
            retrieval_mode = getattr(RetrievalMode, retrieval_mode.upper())
        )
            

    @staticmethod
    def _texts_to_documents(entries:Union[List[dict], dict]):
        if isinstance(entries, List):
            entries = [entry for entry in entries if entry.get(text_col_name, None)]
            return [
                Document(
                    page_content=entry_dict[text_col_name], 
                    metadata={
                        date_col_name: entry_dict[date_col_name], 
                        main_tags_col_name: entry_dict[main_tags_col_name],
                        sub_tags_col_name: entry_dict[sub_tags_col_name]
                        }
                    ) for entry_dict in entries
                ] 

        else:
            return [
                Document(
                    page_content=entries[text_col_name], 
                    metadata={
                        date_col_name: entries[date_col_name], 
                        main_tags_col_name: entries[main_tags_col_name],
                        sub_tags_col_name: entries[sub_tags_col_name]
                        }
                    )
            ] if entries.get(text_col_name, None) else None


    @staticmethod
    def add_entry_to_vdb(vdb, formatted_entries:List):
        documents_entries = LangVdb._texts_to_documents(formatted_entries)
        if documents_entries:
            for doc in documents_entries:
               # print(doc)
               pass
            vdb.add_documents(documents=documents_entries)
        
    @staticmethod
    def retrieval(query, lang_vdb, 
                  rerank:bool=retrieval_rerank_flag, filters:Dict={}, filter_strictness=filter_strictness_choices[0], k_outputs:int=k_outputs_retrieval, search_type:str=retrieval_search_type, 
                  str_format_results:bool=True):
        
        # Build initial search_kwargs
        search_kwargs = {'k': k_outputs}
                
        if filters:
            # If there is filters value
            if any([value for key, value in filters.items()]):
                # Convert filter for qdrant
                if LangVdb._vdb == 'qdrant':
                    filters = LangVdb._convert_filters_to_qdrant_filter(filters, filter_strictness)
                # Add filter in search_kwargs
                search_kwargs['filter'] = filters     

        # kwargs specific to "similarity_score_threshold" and "mmr"
        if search_type == "similarity_score_threshold":
            search_kwargs['score_threshold'] = relevance_threshold
        elif search_type == "mmr":
            search_kwargs['fetch_k'] = mmr_fetch_k
            search_kwargs['lambda_mult'] = mmr_lambda_mult       
        
        # Get retrieval engine
        retriever = lang_vdb.as_retriever(
            search_type=search_type,
            search_kwargs=search_kwargs
            )
        
        print('\n\n\n\n\n\mSEARCH PARAMS', search_kwargs)
        
        if not rerank:
            retrieved_docs = retriever.invoke(query)
        else:
            # Perform rerank with FlashRank
            compressor = FlashrankRerank()
            compression_retriever = ContextualCompressionRetriever(
                base_compressor=compressor, base_retriever=retriever
            )       
            retrieved_docs = compression_retriever.invoke(query)
        
        print('DOCS:', LangVdb._retrieval_results_str_format(retrieved_docs))
        
        
        return LangVdb._retrieval_results_str_format(retrieved_docs) if str_format_results else retrieved_docs

    @staticmethod
    def _convert_filters_to_qdrant_filter(filters: Dict, filter_strictness:str) -> Union[Filter, None]:
        
        def _create_field_condition(filter_name: str, filter_value: str, value_or_any:str) -> 'FieldCondition':
            if value_or_any == 'match_value':
                return FieldCondition(key=f"metadata.{filter_name}[]", match=MatchValue(value=filter_value))
            elif value_or_any == 'match_any':
                return FieldCondition(key=f"metadata.{filter_name}[]", match=MatchAny(any=filter_value))
        
        if not filters:
            return None
        
        # For now we only add filter for main_tags and sub_tags (which are lists)
        available_filters = [main_tags_col_name, sub_tags_col_name]
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
    
    
    
    @staticmethod
    def _retrieval_results_str_format(retrieved_docs):
 
        return f"\n{'-'*50}\n".join(
                    [
                        f"Document {i+1}:\n{doc.page_content}\n" +\
                        f"Metadata:\n" + "\n".join([f"{key}: {str(value)}" for key, value in doc.metadata.items()])
                        for i, doc in enumerate(retrieved_docs)
                    ]
                )
    
