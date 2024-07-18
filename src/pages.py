import taipy.gui.builder as tgb
from taipy.gui import Icon

from callbacks import (
            on_login, on_sign_in,
            on_data_entry_add,
            on_image_to_text,
            on_refresh_table
        )

from tools import DataManagment as dm

from page_ids import page_ids
from constants import (
    user_email, user_password, verify_password, 
    text_date, tags_separator, main_tags, sub_tags, text_entry, 
    image_to_text_languages, image_to_text_cpu_or_gpu, selected_languages, selected_image_paths, image_to_text_output,
    user_table
    )


##########              ##########
########## SIGNin/LOGin ##########
##########              ##########

# Init page (choose between log in and sign in and be redirected toward the
# corresponding page)
# ---
with tgb.Page() as init:
    tgb.navbar(lov=[('/'+page_ids['welcome'], 'Welcome page'),
                    ('/'+page_ids['log_in'], 'Log in'), 
                    ('/'+page_ids['sign_in'], 'Sign in')])
    with tgb.layout("1 1 1"):
        tgb.text('\n')
        tgb.text("## Welcome! Please Sign-in or Log-in to access the app.", mode="md")
# Welcoming page
# ---
with tgb.Page() as welcome:
    with tgb.layout("1 1 1"):
        tgb.text('\n')
        tgb.text("## Welcome! Please Sign-in or Log-in to access the app.", mode="md")

# Sign in page
# ---
with tgb.Page() as sign_in:
    with tgb.layout("1 1 1"):
        tgb.text(' ')
        with tgb.part():
            tgb.text("## Fill the following fields and sign in!", mode="md")
            tgb.input("{user_email}", label='Email address*')
            tgb.input("{user_password}", password=True, label='Password*')
            tgb.input("{verify_password}", password=True, label='Password verification*')
            tgb.button("Sign in!", on_action=on_sign_in)
# Log in page
# ---
with tgb.Page() as log_in:
    with tgb.layout("1 1 1"):
        tgb.text(' ')
        with tgb.part():
            tgb.login("Please log in! (reaload page to retry)", on_action=on_login)




##########      ##########
########## DATA ##########
##########      ##########

# Root page (choose action with menu)
with tgb.Page() as root_page:
    tgb.navbar(
             lov=[
                 ('/'+page_ids['chose_task'], '---'),
                 ('/'+page_ids['retrieve_data'], Icon('/images/data_retrieval.png', 'Retrieve tagged data')),
                 ('/'+page_ids['manage_data'], Icon('/images/data_managment.png', 'Manage your data')), 
                ]
             )
    
# Welcoming page
# ---
with tgb.Page() as chose_task:
    with tgb.layout("1 1 1"):
        tgb.text(' ')
        tgb.text('## Welcome! Please choose your task.', mode="md")

with tgb.Page() as manage_data:
    tgb.text("## Add a single entry:", mode="md")
    with tgb.layout("1 1 1 1 1 1"):
        tgb.input("{text_date}", label='Text date')
        tgb.input("{tags_separator}", label='Separator used to fill following fields with several values.')
        tgb.input("{main_tags}", label='Main tags')
        tgb.input("{sub_tags}", label='Sub tags')
        tgb.input("{text_entry}", label='Text to add*')
        tgb.button("Add data entry", on_action=on_data_entry_add)
    
    tgb.text("## Image to text conversion.", mode="md")
    with tgb.layout("1 1 1"):
        tgb.selector(value="{selected_languages}", 
                     lov=';'.join([str(language) for language in image_to_text_languages.keys()]), 
                     multiple=True, dropdown=True)
        tgb.selector(value="{image_to_text_cpu_or_gpu}", lov="CPU;GPU")
        tgb.file_selector("{selected_image_paths}", 
                          label="Upload image", 
                          extensions=".png, .jpg", drop_message="Drop your image here", 
                          on_action=on_image_to_text)
    with tgb.layout("1 1 1"):
        tgb.text("   ")
        tgb.text("# {image_to_text_output}", mode="md")
    
    tgb.text("## Load an entire text file:", mode="md")
    

#user_table = "{dm.json_to_dataframe(user_email)}"
with tgb.Page() as retrieve_data:
    tgb.text("## Your data:", mode="md")
    
    tgb.table("{user_table}")
    tgb.button("Load/Refresh df", on_action=on_refresh_table)
    
    