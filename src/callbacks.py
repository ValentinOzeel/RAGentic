import taipy.gui.builder as tgb
from taipy.gui import navigate, notify

from constants import notify_duration
from page_ids import page_ids
from tools import SignLog as sl, DataManagment as dm, image_to_text_conversion


def on_sign_in(state, id, login_args):    
    # State corresponding values
    acc_creation_values = [state.user_email, state.user_password, state.verify_password]
    # Notify error if all fields are not filled
    if not all(element for element in acc_creation_values):
        notify(state, 'error', 'All fields are required to be filled in!', duration=notify_duration)
    # If they are, check the email adress' validity
    else:
        e_check = sl.email_check(state.user_email)
        if not e_check:
            notify(state, 'error', 'Incorrect email address!', duration=notify_duration)
            
        # Notify error if password and verify_password correspond
        elif state.user_password != state.verify_password:
            notify(state, 'error', "Password and password verification entries don't match!", duration=notify_duration)
            
        ##### SEND AN EMAIL (send_email_as_verif func) WITH RANDOM CODE, ASK USER TO INSERT IT TO VERIFY EMAIL
        ##### SEND AN EMAIL (send_email_as_verif func) WITH RANDOM CODE, ASK USER TO INSERT IT TO VERIFY EMAIL
        ##### SEND AN EMAIL (send_email_as_verif func) WITH RANDOM CODE, ASK USER TO INSERT IT TO VERIFY EMAIL
        
        # If all is good until there
        else:
            # Add credentials except if user already exists
            status = sl.add_user_credentials(state.user_email, state.user_password)
            if not status:
                notify(state, 'error', 'An account is already associated to this email adress.', duration=notify_duration)
            else:
                notify(state, 'success', 'Account successfully created!', duration=notify_duration)
                # Initialize user's json file
                dm.initialize_json_data_file_if_not_already(state.user_email)
                navigate(state, page_ids["log_in"])
                

def on_login(state, id, login_args):
    user_email, user_password = login_args["args"][:2]
    # Check whether fields have been filled
    if user_email is None or user_password is None:
        notify(state, 'error', 'None of email adress and/or password fields should be empty.', duration=notify_duration)
        
    elif not sl.check_user_credentials(user_email, user_password):
        notify(state, 'error', 'Incorrect email adress and/or password.', duration=notify_duration)
        
    else:
        state.user_email, state.user_password = user_email, user_password
        # Set user_table variable as user's json data loaded into df 
        state.user_table = dm.json_to_dataframe(state.user_email)
        notify(state, 'success', 'Successful authentification.')
        navigate(state, page_ids["root_page"])


def on_menu(state, id, payload):
    print(payload)
    #selected_page = info["args"][0]
    #print(selected_page, page_ids, page_ids[selected_page])
    #navigate(state, to=page_ids[selected_page])
    

    
def on_data_entry_add(state, action, info):   
    if not state.text_entry:
        notify(state,'error', 'You must at least fill the text field (indicated with an *).')
    
    if state.tags_separator:
        state.main_tags = [tag for tag in state.main_tags.split(state.tags_separator) if state.tags_separator in state.main_tags] if state.main_tags else state.main_tags
        state.sub_tags = [tag for tag in state.sub_tags.split(state.tags_separator) if state.tags_separator in state.sub_tags] if state.sub_tags else state.sub_tags

    text_and_metadata = {
        'text_date': state.text_date,
        'main_tags': state.main_tags,
        'sub_tags': state.sub_tags,
        'text_entry': state.text_entry,
    }
    # Add entry in json file
    dm.add_entry_to_json(state.user_email, text_and_metadata)
    # Notify success
    notify(state, 'success', 'Text added to database.', duration=notify_duration)
    # Udpate dataframe shown
    state.user_table = dm.json_to_dataframe(state.user_email)
    
     
    
def on_image_to_text(state, id, payload):
    if not all([state.selected_languages, state.image_to_text_cpu_or_gpu,  state.selected_image_paths]):
        notify(state,'error', 'You must select at least CPU or GPU, one (or more) language, and provide an image.', duration=notify_duration)
    
    else:
        state.image_to_text_output = image_to_text_conversion(
            state.selected_languages, 
            state.image_to_text_cpu_or_gpu, 
            state.selected_image_paths
        )
        notify(state, 'success', 'Converted image as text', duration=notify_duration)
         
         
         
def on_refresh_table(state, id, payload):
    state.user_table = dm.json_to_dataframe(state.user_email)