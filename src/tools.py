import os
import time
import regex as re
import yaml
import json
import sqlite3
from contextlib import contextmanager
import pandas as pd

from typing import List, Dict

import easyocr

from constants import (credentials_yaml_path, sqlite_database_path, image_to_text_languages,
                       sqlite_tags_separator, date_col_name, main_tags_col_name, sub_tags_col_name, text_col_name)

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
    @contextmanager
    def get_db_connection():
        """Get a connection to the SQLite database and close it when context ends (after user action)."""
        conn = sqlite3.connect(sqlite_database_path)
        try:
            yield conn
        finally:
            conn.close()


    @staticmethod   
    def _get_number_of_entries(user_id):
        """Get the number of entries in the database for a specific user."""
        with SQLiteManagment.get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT COUNT(*) FROM entries WHERE user_id = ?', (user_id,))
            count = cursor.fetchone()[0]
            return count
    
    @staticmethod
    def format_entry(text_date, main_tags, sub_tags, text_entry):
        main_tags = f'{sqlite_tags_separator}'.join(main_tags) if isinstance(main_tags, List) else main_tags
        sub_tags = f'{sqlite_tags_separator}'.join(sub_tags) if isinstance(sub_tags, List) else sub_tags
        return {
            date_col_name: text_date,
            main_tags_col_name: main_tags,
            sub_tags_col_name: sub_tags,
            text_col_name: text_entry,
        }
    
    @staticmethod 
    def add_entry_to_db(user_id, single_entry=None, multiple_entries: List[Dict] = None):
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
    def db_to_dataframe(user_id):
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



def txt_file_to_formatted_entries(file_path, entry_delimiter, file_tags_separator, date_delimiter, main_tags_delimiter, sub_tags_delimiter, text_delimiter):
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
                main_t = lines[i].replace(main_tags_delimiter, '').split(file_tags_separator)
                main_tags = f'{sqlite_tags_separator}'.join([tag.replace(' ', '') for tag in main_t])
            # Remove sub_tags delimiter, white spaces and split tags by their file_tags_separator  
            elif sub_tags_delimiter in lines[i]:
                sub_t = lines[i].replace(sub_tags_delimiter, '').split(file_tags_separator)
                sub_tags = f'{sqlite_tags_separator}'.join([tag.replace(' ', '') for tag in sub_t])
            # Remove text_delimiter and grab the text
            elif text_delimiter in lines[i]:
                text = ' '.join(lines[i:]).replace(text_delimiter, '')

        return SQLiteManagment.format_entry(date, main_tags, sub_tags, text)


    # Open and read the file
    with open(file_path, 'r') as file:
        content = file.read()

    # Split the content by the delimiter
    entries = content.split(entry_delimiter)
    # Process each entry to retrieve data
    return [_process_entry(entry) for entry in entries]



def get_user_tags(state):
    # Get all unique user's main and sub-tags values
    if isinstance(state.user_table, pd.DataFrame):
        state.user_main_tags = sorted(list(set(tag for main_tag_list in state.user_table[main_tags_col_name] if main_tag_list is not None for tag in main_tag_list if tag)))
        state.user_sub_tags = sorted(list(set(tag for sub_tag_list in state.user_table[sub_tags_col_name] if sub_tag_list is not None for tag in sub_tag_list if tag))) 
    return state


    ##### PSEUDO CO DE FOR RETRIEVAL
    
    
#    
#    
#at login :
#    check if all doc text in sqlite are in milvus otherwise add them
#    If user add text / file, add them in milvus too 
#    
#  embedding:
#      # ！The default dimension is 1024, if you need other dimensions, please clone the model and modify `modules.json` to replace `2_Dense_1024` with another dimension, e.g. `2_Dense_256` or `2_Dense_8192` !
#model = SentenceTransformer("infgrad/stella_en_400M_v5", trust_remote_code=True).cuda()
#query_embeddings = model.encode(queries, prompt_name=query_prompt_name)
#doc_embeddings = model.encode(docs)
#print(query_embeddings.shape, doc_embeddings.shape)
## (2, 1024) (2, 1024)
#
#similarities = model.similarity(query_embeddings, doc_embeddings)
#print(similarities)   
#    
#class LangMilvRAG():
#    def __init__(self):
#        