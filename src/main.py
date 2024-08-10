from taipy.gui import Gui

from tools import YamlManagment, SQLiteManagment
from pages import (
    welcome,
    init,
    sign_in,
    log_in,
    root_page, 
    chose_task,
    manage_data, 
    retrieve_data, 
    rag
    )
from page_ids import page_ids


if __name__ == "__main__":
    YamlManagment.create_cred_yaml_file()

    # Initialize the SQlite database (creates the file and table if not exists)
    SQLiteManagment.initialize_db()
    
    pages = {
        page_ids["init"]: init,
        page_ids['welcome']: welcome,
        page_ids["sign_in"]: sign_in,
        page_ids["log_in"]: log_in,
        page_ids["root_page"]: root_page,
        page_ids["chose_task"]: chose_task,
        page_ids["manage_data"]: manage_data,
        page_ids["retrieve_data"]: retrieve_data,
        page_ids["rag"]: rag
    }

    my_theme = {
      "palette": {
        "background": {"default": "#0e0916"},
        "primary": {"main": "#67be86"}
      }
    }
  
    style_kit = {
      'color_primary': "#67be86",
      'color_secondary': "#4fc2b0",
      
      'font_family': "Lato, Arial, sans-serif",
      
      'color_error': "#FF595E",
      'color_warning': "#FAA916",
      'color_success': "#96E6B3",
      
      'input_button_height': "65px"
    }
    
    Gui(pages=pages).run(favicon='src\images\rag.png', theme=my_theme, stylekit=style_kit, use_reloaders=True, debug=True)