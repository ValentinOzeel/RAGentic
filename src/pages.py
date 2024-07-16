import taipy.gui.builder as tgb
from taipy.gui import Icon

from callbacks import on_menu, on_login



# Add a navbar to switch from one page to the other
with tgb.Page() as root_page:
    tgb.login("Log in or create an account below!", on_action=on_login)


with tgb.Page() as incorrect:
    with tgb.layout("1 1 1"):
        
        tgb.text(' ')
        
        with tgb.part():
            tgb.text("## Try again or create an account below!", mode="md")
            tgb.text("## Actualize the page first to try again.", mode="md")

            tgb.login("Try again or create an account below!", on_action=on_login)







with tgb.Page() as succes_authentification:
    tgb.menu(label="Main menu", 
             lov=[('manage_data', Icon('/images/data_managment.png', 'Manage your data')), ('retrieve_data', Icon('/images/data_retrieval.png', 'Retrieve tagged data'))], 
             on_action=on_menu)
        
with tgb.Page() as manage_data_p:
    tgb.text("## This is page 1", mode="md")
    
with tgb.Page() as retrieve_data_p:
    tgb.text("## This is page 2", mode="md")