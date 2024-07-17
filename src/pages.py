import taipy.gui.builder as tgb
from taipy.gui import Icon

from callbacks import SignLogCallbacks as sl_callbacks, on_menu

from page_ids import page_ids
from constants import user_email, user_password, verify_password


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
        tgb.text('Please sign-in or log-in!')
# Welcoming page
# ---
with tgb.Page() as welcome:
    with tgb.layout("1 1 1"):
        tgb.text(' ')
        tgb.text('Welcome! Please Sign-in or Log-in to access the app.')
# Sign in page
# ---
with tgb.Page() as sign_in:
    with tgb.layout("1 1 1"):
        tgb.text(' ')
        with tgb.part():
            tgb.input("{user_email}", label='Email address*')
            tgb.input("{user_password}", password=True, label='Password*')
            tgb.input("{verify_password}", password=True, label='Password verification*')
            tgb.button("Sign in!", on_action=sl_callbacks.on_sign_in)
# Log in page
# ---
with tgb.Page() as log_in:
    with tgb.layout("1 1 1"):
        tgb.text(' ')
        with tgb.part():
            tgb.login("Please log in! (reaload page to retry)", on_action=sl_callbacks.on_login)




##########      ##########
########## DATA ##########
##########      ##########

# Root page (choose action with menu)
with tgb.Page() as root_page:
    tgb.menu(label="Main menu", 
             lov=[('manage_data', Icon('/images/data_managment.png', 'Manage your data')), ('retrieve_data', Icon('/images/data_retrieval.png', 'Retrieve tagged data'))], 
             on_action=on_menu)
        
with tgb.Page() as manage_data:
    tgb.text("## You can either load a single entry or an entire text file.", mode="md")

    with tgb.layout("1 1 1 1 1 1"):
        tgb.input("{text_date}", label='Text date')
        tgb.input("{tags_separator}", label='Separator used to fill following fields with several values.')
        tgb.input("{main_tags}", label='Main tags')
        tgb.input("{sub_tags}", label='Sub tags')
        tgb.input("{text_entry}", label='Text to add*')
        tgb.button("Sign in!", on_action=on_data_entry_add)

    
with tgb.Page() as retrieve_data:
    tgb.text("## This is page 2", mode="md")