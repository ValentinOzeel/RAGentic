import os
import regex as re
import yaml
import json
import pandas as pd

from typing import List, Dict

import easyocr

from constants import (credentials_yaml_path, json_data_folder_path, image_to_text_languages,
                       date_col_name, main_tags_col_name, sub_tags_col_name, text_col_name)

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



class DataManagment():
    @staticmethod
    def _get_user_json_file_path(user_id):
        return os.path.join(json_data_folder_path, ''.join([user_id, '.json']))
    
    @staticmethod
    def initialize_json_data_file_if_not_already(user_id):
        """Initialize the JSON file with an empty dictionary."""
        json_path = DataManagment._get_user_json_file_path(user_id)
        if not os.path.isfile(json_path):
            with open(json_path, 'w') as file:
                json.dump({}, file, indent=4)
    
    @staticmethod   
    def _get_number_of_entries(user_id):
        """Get the number of entries in the JSON file."""
        json_path = DataManagment._get_user_json_file_path(user_id)
        with open(json_path, 'r') as file:
            data = json.load(file)
        return len(data)
    
    
    @staticmethod
    def format_entry(text_date, main_tags, sub_tags, text_entry):
        return {
            date_col_name: text_date,
            main_tags_col_name: main_tags,
            sub_tags_col_name: sub_tags,
            text_col_name: text_entry,
        }
    
    @staticmethod 
    def add_entry_to_json(user_id, single_entry=None, multiple_entries:List[Dict]=None):
        """Add an entry to the JSON file."""
        def ignore_if_empty(entry):
            if not entry.get(text_col_name, None):
                return True 
            return False
            
        
        json_path = DataManagment._get_user_json_file_path(user_id)
        with open(json_path, 'r') as file:
            json_data = json.load(file) or {}

        if single_entry:
            if ignore_if_empty(single_entry): 
                return
            new_entry_id = DataManagment._get_number_of_entries(user_id) + 1
            json_data[new_entry_id] = single_entry  
        
        elif multiple_entries:
            last_entry_id = DataManagment._get_number_of_entries(user_id)
            
            counter = 1
            for entry in multiple_entries:
                if ignore_if_empty(entry): 
                    continue
                json_data[last_entry_id + counter] = entry  
                counter += 1
                

        with open(json_path, 'w') as file:
            json.dump(json_data, file, indent=4)
        

    
    @staticmethod  
    def json_to_dataframe(user_id):
        """Convert JSON file to a Pandas DataFrame."""
        try:
            json_path = DataManagment._get_user_json_file_path(user_id)
            with open(json_path, 'r') as file:
                data = json.load(file)

            # Transform the data into a DataFrame
            df = pd.DataFrame.from_dict(data, orient='index')
            df.index.name = 'text_id'
            df.reset_index(inplace=True)
            df[date_col_name] = pd.to_datetime(df[date_col_name], format='%m/%d/%Y')
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
                main_tags = [tag.replace(' ', '') for tag in main_t]
            # Remove sub_tags delimiter, white spaces and split tags by their file_tags_separator  
            elif sub_tags_delimiter in lines[i]:
                sub_t = lines[i].replace(sub_tags_delimiter, '').split(file_tags_separator)
                sub_tags = [tag.replace(' ', '') for tag in sub_t]
            # Remove text_delimiter and grab the text
            elif text_delimiter in lines[i]:
                text = ' '.join(lines[i:]).replace(text_delimiter, '')

        return DataManagment.format_entry(date, main_tags, sub_tags, text)


    # Open and read the file
    with open(file_path, 'r') as file:
        content = file.read()

    # Split the content by the delimiter
    entries = content.split(entry_delimiter)
    # Process each entry to retrieve data
    return [_process_entry(entry) for entry in entries]





#################### WORKING ON REPLACING JSON DATA MANAGMERNT WITH SQLITE DATABASE
#################### WORKING ON REPLACING JSON DATA MANAGMERNT WITH SQLITE DATABASE
#################### WORKING ON REPLACING JSON DATA MANAGMERNT WITH SQLITE DATABASE
#################### WORKING ON REPLACING JSON DATA MANAGMERNT WITH SQLITE DATABASE

What is needed to work on now : Efficient tags retrival 
Storing Lists in SQLite:
Storing as Strings: Store the list as a comma-separated string.
Converting on Retrieval: Convert the string back to a list when reading from the database.
Test to be carrried out
    

import os
import sqlite3
import pandas as pd
from typing import List, Dict

# Specify the path to the SQLite database file
database_file_path = 'data.db'

class DataManagment:
    
    @staticmethod
    def _get_db_connection():
        """Get a connection to the SQLite database."""
        conn = sqlite3.connect(database_file_path)
        return conn
    
    @staticmethod
    def initialize_db():
        """Initialize the SQLite database with the necessary table if not already exists."""
        conn = DataManagment._get_db_connection()
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS entries (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL,
                text_date TEXT,
                main_tags TEXT,
                sub_tags TEXT,
                text_entry TEXT
            )
        ''')
        conn.commit()
        conn.close()

    @staticmethod   
    def _get_number_of_entries(user_id):
        """Get the number of entries in the database for a specific user."""
        conn = DataManagment._get_db_connection()
        cursor = conn.cursor()
        cursor.execute('SELECT COUNT(*) FROM entries WHERE user_id = ?', (user_id,))
        count = cursor.fetchone()[0]
        conn.close()
        return count
    
    @staticmethod
    def format_entry(text_date, main_tags, sub_tags, text_entry):
        return {
            date_col_name: text_date,
            main_tags_col_name: main_tags,
            sub_tags_col_name: sub_tags,
            text_col_name: text_entry,
        }
    
    @staticmethod 
    def add_entry_to_db(user_id, single_entry=None, multiple_entries:List[Dict]=None):
        """Add an entry or multiple entries to the SQLite database."""
        def ignore_if_empty(entry):
            if not entry.get(text_col_name, None):
                return True 
            return False
            
        conn = DataManagment._get_db_connection()
        cursor = conn.cursor()

        if single_entry:
            if ignore_if_empty(single_entry): 
                return
            cursor.execute('''
                INSERT INTO entries (user_id, text_date, main_tags, sub_tags, text_entry)
                VALUES (?, ?, ?, ?, ?)
            ''', (user_id, single_entry[date_col_name], single_entry[main_tags_col_name], 
                  single_entry[sub_tags_col_name], single_entry[text_col_name]))
        
        elif multiple_entries:
            entries_to_add = [
                (user_id, entry[date_col_name], entry[main_tags_col_name], entry[sub_tags_col_name], entry[text_col_name])
                for entry in multiple_entries if not ignore_if_empty(entry)
            ]
            cursor.executemany('''
                INSERT INTO entries (user_id, text_date, main_tags, sub_tags, text_entry)
                VALUES (?, ?, ?, ?, ?)
            ''', entries_to_add)

        conn.commit()
        conn.close()
    
    @staticmethod  
    def db_to_dataframe(user_id):
        """Convert the database entries to a Pandas DataFrame for a specific user."""
        try:
            conn = DataManagment._get_db_connection()
            query = 'SELECT id as text_id, text_date, main_tags, sub_tags, text_entry FROM entries WHERE user_id = ?'
            df = pd.read_sql_query(query, conn, params=(user_id,))
            df[date_col_name] = pd.to_datetime(df[date_col_name], format='%m/%d/%Y')
            conn.close()
            return df
        except Exception as e:
            print(e)
            return None

# Example usage
if __name__ == '__main__':
    DataManagment.initialize_db()
    
    # Adding single entry
    single_entry = DataManagment.format_entry('01/01/2023', 'Tag1,Tag2', 'SubTag1', 'Sample text entry')
    DataManagment.add_entry_to_db('user1', single_entry=single_entry)
    
    # Adding multiple entries
    multiple_entries = [
        DataManagment.format_entry('02/01/2023', 'Tag1,Tag3', 'SubTag2', 'Another text entry'),
        DataManagment.format_entry('03/01/2023', 'Tag2', 'SubTag1,SubTag3', 'Yet another text entry')
    ]
    DataManagment.add_entry_to_db('user1', multiple_entries=multiple_entries)
    
    # Convert database to DataFrame
    df = DataManagment.db_to_dataframe('user1')
    print(df)