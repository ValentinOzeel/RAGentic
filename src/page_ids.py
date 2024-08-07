import random

pages_names = [
    'welcome',
    'init',
    'sign_in',
    'log_in',
    'root_page', 
    'chose_task',
    'manage_data',
    'retrieve_data',
    'rag_app'
]

######## PAGE ID BUILDER ########
def make_random_page_id(n_numbers:str=15):
    # Used to prevent user to access pages by knowing their ID
    return ''.join([str(random.randint(1, 9)) for n in range(n_numbers)])

page_ids = {page_name : make_random_page_id() for page_name in pages_names}
######## PAGE ID BUILDER ########