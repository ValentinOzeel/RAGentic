import taipy.gui.builder as tgb
from taipy.gui import navigate, notify
import pandas as pd
from constants import notify_duration, date_col_name
from page_ids import page_ids
from tools import SignLog as sl, DataManagment as dm, image_to_text_conversion, txt_file_to_formatted_entries


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
    

    
def on_data_entry_add(state, action, info):   
    if not state.text_entry:
        notify(state,'error', 'You must at least fill the text field (indicated with an *).')
    
    if state.tags_separator:
        main_tags = [tag for tag in state.main_tags.split(state.tags_separator) if state.tags_separator in state.main_tags] if state.main_tags else state.main_tags
        sub_tags = [tag for tag in state.sub_tags.split(state.tags_separator) if state.tags_separator in state.sub_tags] if state.sub_tags else state.sub_tags

    # Format entry
    formatted_entry = dm.format_entry(state.text_date, main_tags, sub_tags, state.text_entry)
    # Add entry in json file
    dm.add_entry_to_json(state.user_email, single_entry=formatted_entry)
    # Notify success
    notify(state, 'success', 'Text added to database.', duration=notify_duration)
    # Udpate dataframe shown
    state.user_table = dm.json_to_dataframe(state.user_email)
    state.text_entry = ''
    
    
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
         
def on_txt_file_load(state, id, payload):
    if not all([state.entry_delimiter, state.text_delimiter, state.text_file_to_load]):
        return notify(state,'error', 'You must at least provide entry_delimiter, text_delimiter and the text file.', duration=notify_duration)
    
    if not all([state.date_delimiter, state.file_tags_separator, state.main_tags_delimiter, state.sub_tags_delimiter]):
        notify(state,'warning', 'Some fields were left empty.', duration=notify_duration)
    
    try:
        formatted_entries = txt_file_to_formatted_entries(
            file_path=state.text_file_to_load,
            entry_delimiter=state.entry_delimiter,
            file_tags_separator=state.file_tags_separator,
            date_delimiter=state.date_delimiter,
            main_tags_delimiter=state.main_tags_delimiter,
            sub_tags_delimiter=state.sub_tags_delimiter,
            text_delimiter=state.text_delimiter
        )
        # Add entry in json file
        dm.add_entry_to_json(state.user_email, multiple_entries=formatted_entries)
        # upadte table
        state.user_table = dm.json_to_dataframe(state.user_email)
        
    except Exception as e:
        print(e)
        return notify(state,'error', "The file couldn't be loaded", duration=notify_duration)
    
    notify(state,'success', 'Loaded text file as entries', duration=notify_duration)
    
def on_reset_filters(state, id, payload):
    state.user_table = dm.json_to_dataframe(state.user_email)
    
def on_filter_date(state, id, payload):
    start_date, end_date = pd.to_datetime(state.filter_dates[0]), pd.to_datetime(state.filter_dates[1])
    
    if start_date and end_date:
        if start_date > end_date:
            return notify(state,'error', 'End date cannot be inferior to Start date.', duration=notify_duration)

        fresh_df = dm.json_to_dataframe(state.user_email)
        filtered_table = fresh_df[fresh_df[[date_col_name]].notnull().all(axis=1)]
        state.user_table = filtered_table[
            (filtered_table[date_col_name] >= start_date) & (filtered_table[date_col_name] <= end_date)
            ]
    