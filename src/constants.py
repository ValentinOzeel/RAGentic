import os 

###                    ###               
### SignLog constants  ###
###                    ###
user_email = None 
user_password = None 
verify_password = None 

###                    ###               
###   Data constants   ###
###                    ###
text_date = None
tags_separator = None
main_tags = None
sub_tags = None
text_entry = None

###                         ###               
### Miscellanous constants  ###
###                         ###

notify_duration = 5000 #mseconds

# Assuming we are in src\constants.py
root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
credentials_yaml_path = os.path.join(root_path, 'conf', 'app_credentials.yaml')
json_data_folder_path = os.path.join(root_path, 'conf', 'user_files')