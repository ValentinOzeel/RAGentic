from taipy.gui import Gui
import taipy.gui.builder as tgb

from tools import create_cred_yaml_file
from pages import (
    welcome,
    init,
    sign_in,
    log_in,
    root_page, 
    manage_data, 
    retrieve_data, 
    )
from page_ids import page_ids




if __name__ == "__main__":
    create_cred_yaml_file()
    
    pages = {
        page_ids["init"]: init,
        page_ids['welcome']: welcome,
        page_ids["sign_in"]: sign_in,
        page_ids["log_in"]: log_in,
        page_ids["root_page"]: root_page,
        page_ids["manage_data"]: manage_data,
        page_ids["retrieve_data"]: retrieve_data
    }

    Gui(pages=pages).run(use_reloaders=True, debug=True)