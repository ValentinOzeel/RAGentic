import random

from taipy.gui import Gui
import taipy.gui.builder as tgb

from pages import (
    root_page, 
    succes_authentification, 
    manage_data_p, 
    retrieve_data_p, 
    incorrect
    )


######## PAGE ID BUILDER ########

def make_random_page_id(n_numbers:str=15):
    # Used to prevent user to access pages by knowing their ID
    return ''.join([str(random.randint(1, 9)) for number in range(n_numbers)])

pages_names = [
    'root', 
    'unsuccessful_auth',
    'succesful_auth',
    'manage_data',
    'retrieve_data'
]

page_ids = {page_name : make_random_page_id() for page_name in pages_names}

######## PAGE ID BUILDER ########




pages = {
    page_ids["root"]: root_page,
    page_ids["unsuccessful_auth"]: incorrect,
    page_ids["succesful_auth"]: succes_authentification,
    page_ids["manage_data"]: manage_data_p,
    page_ids["retrieve_data"]: retrieve_data_p
}



if __name__ == "__main__":
    Gui(pages=pages).run(use_reloaders=True, debug=True)