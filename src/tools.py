import os
import regex as re
import yaml
import json
import pandas as pd

from constants import credentials_yaml_path, json_data_folder_path

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
        return os.path.join(json_data_folder_path, ''.join(user_id, '.json'))
    
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
    def add_entry_to_json(user_id, data):
        """Add an entry to the JSON file."""
        json_path = DataManagment._get_user_json_file_path(user_id)
        with open(json_path, 'r') as file:
            data = json.load(file) or {}

        entry_id = DataManagment._get_number_of_entries() + 1
        
        data[entry_id] = data  

        with open(json_path, 'w') as file:
            json.dump(data, file, indent=4)
    
    @staticmethod  
    def json_to_dataframe(user_id):
        """Convert JSON file to a Pandas DataFrame."""
        json_path = DataManagment._get_user_json_file_path(user_id)
        with open(json_path, 'r') as file:
            data = json.load(file)

        # Transform the data into a DataFrame
        df = pd.DataFrame.from_dict(data, orient='index')
        df.index.name = 'text_id'
        df.reset_index(inplace=True)
        return df