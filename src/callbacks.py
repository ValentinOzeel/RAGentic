import taipy.gui.builder as tgb
from taipy.gui import navigate, notify
import pandas as pd
from constants import notify_duration, date_col_name, main_tags_col_name, sub_tags_col_name, filter_strictness_choices, base_type_search, k_outputs

from page_ids import page_ids
from tools import SignLog as sl, SQLiteManagment as sm, image_to_text_conversion, format_entry, txt_file_to_formatted_entries, LangMilvusManagment as lmm

def _get_user_tags(state):
    # Get all unique user's main and sub-tags values
    if isinstance(state.user_table, pd.DataFrame):
        state.user_main_tags = sorted(list(set(tag for main_tag_list in state.user_table[main_tags_col_name] if main_tag_list is not None for tag in main_tag_list if tag)))
        state.user_sub_tags = sorted(list(set(tag for sub_tag_list in state.user_table[sub_tags_col_name] if sub_tag_list is not None for tag in sub_tag_list if tag))) 
    return state

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
                # Create a Milvus database collection for the new user
                lmm.initialize_milvus_db_collection(state.user_email)
                notify(state, 'success', 'Account successfully created!', duration=notify_duration)
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
        # Set user_table variable as user's sqlite data loaded into df 
        state.user_table = sm.sqlite_to_dataframe(state.user_email)
        # Get all unique user's main and sub-tags values
        state = _get_user_tags(state)
        # Access Milvus collection
        state.lang_milvus_user_vdb = lmm.access_lang_milvus_vdb(state.user_email)
        notify(state, 'success', 'Successful authentification.')
        navigate(state, page_ids["root_page"])
    

    
def on_data_entry_add(state, action, info):   
    if not state.text_entry:
        notify(state,'error', 'You must at least fill the text field (indicated with an *).')
    
    if state.tags_separator:
        main_tags = [tag for tag in state.main_tags.split(state.tags_separator) if state.tags_separator in state.main_tags] if state.main_tags else state.main_tags
        sub_tags = [tag for tag in state.sub_tags.split(state.tags_separator) if state.tags_separator in state.sub_tags] if state.sub_tags else state.sub_tags

    # Add entry in sqlite
    sm.add_entry_to_sqlite(
        state.user_email, 
        single_entry=format_entry(state.text_date, main_tags, sub_tags, state.text_entry, format='sqlite')
        )
    # Add embedded entry in Milvus
    lmm.add_entry_to_milvus(
        state.lang_milvus_user_vdb, 
        format_entry(state.text_date, main_tags, sub_tags, state.text_entry, format='milvus')
        )
    # Notify success
    notify(state, 'success', 'Text added to database.', duration=notify_duration)
    # Udpate dataframe shown
    state.user_table = sm.sqlite_to_dataframe(state.user_email)
    # Get all unique user's main and sub-tags values
    state = _get_user_tags(state)
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
        format_kwargs = {
            'file_path':state.text_file_to_load,
            'entry_delimiter':state.entry_delimiter,
            'file_tags_separator':state.file_tags_separator,
            'date_delimiter':state.date_delimiter,
            'main_tags_delimiter':state.main_tags_delimiter,
            'sub_tags_delimiter':state.sub_tags_delimiter,
            'text_delimiter':state.text_delimiter
        }

        # Add entry in sqlite
        sm.add_entry_to_sqlite(
            state.user_email, 
            multiple_entries=txt_file_to_formatted_entries(**format_kwargs, format='sqlite')
            )
        # Add embedded entry in Milvus
        lmm.add_entry_to_milvus(
            state.lang_milvus_user_vdb, 
            txt_file_to_formatted_entries(**format_kwargs, format='milvus')
            )
        # upadte table
        state.user_table = sm.sqlite_to_dataframe(state.user_email)
        # Get all unique user's main and sub-tags values
        state = _get_user_tags(state)
        
    except Exception as e:
        print(e)
        return notify(state,'error', "The file couldn't be loaded", duration=notify_duration)
    
    notify(state,'success', 'Loaded text file as entries', duration=notify_duration)
        
    
def on_filter_date(state, id, payload, use_fresh_df=True):
    start_date, end_date = pd.to_datetime(state.filter_dates[0]), pd.to_datetime(state.filter_dates[1])
    
    if start_date and end_date:
        if start_date > end_date:
            return notify(state,'error', 'End date cannot be inferior to Start date.', duration=notify_duration)

        df = sm.sqlite_to_dataframe(state.user_email) if use_fresh_df else state.user_table
        filtered_table = df[df[[date_col_name]].notnull().all(axis=1)]
        state.user_table = filtered_table[
            (filtered_table[date_col_name] >= start_date) & (filtered_table[date_col_name] <= end_date)
            ]
    

def on_filter_tags(state, id, payload):
    def filter_df(list_filter, column_name):
        # Dynamic tag filling so need to start from fresh df
        fresh_df = sm.sqlite_to_dataframe(state.user_email)
        # If non-strict filter
        if state.filter_strictness == filter_strictness_choices[0]:
            return fresh_df[fresh_df[column_name].apply(
                lambda tags: any(tag in list_filter for tag in tags) if tags else False
                )]
        # If strict filter
        elif state.filter_strictness == filter_strictness_choices[1]:
            return fresh_df[fresh_df[column_name].apply(
                lambda tags: all(tag in list_filter for tag in tags) if tags else False
                )]    
    
    main_tags_filtered_df, sub_tags_filtered_df = None, None
    # Filter main tags
    if state.filter_main_tags:
        main_tags_filtered_df = filter_df(state.filter_main_tags, main_tags_col_name)
    # Filter sub tags
    if state.filter_sub_tags:
        sub_tags_filtered_df = filter_df(state.filter_sub_tags, sub_tags_col_name)
    # Build final df
    if main_tags_filtered_df is not None and sub_tags_filtered_df is None:
        state.user_table = main_tags_filtered_df
    elif main_tags_filtered_df is None and sub_tags_filtered_df is not None:
        state.user_table = sub_tags_filtered_df
    else:
        state.user_table = pd.concat([main_tags_filtered_df, sub_tags_filtered_df]).sort_index()
    # re-filter date if needed
    on_filter_date(state, None, None, use_fresh_df=False)

        
def on_reset_filters(state, id, payload):
    state.user_table = sm.sqlite_to_dataframe(state.user_email)
    state.filter_dates[0], state.filter_dates[1] = None, None
    state.filter_main_tags = []
    state.filter_sub_tags = []
    
    
def on_retrieval_query(state, id, payload):
    
    retrieved_docs = lmm.retrieval(
        query=state.retrieval_query,
        lang_milvus_vdb = state.lang_milvus_user_vdb,
        filters= {main_tags_col_name:state.retrieval_main_tags, sub_tags_col_name:state.retrieval_sub_tags},
        k_outputs= state.k_outputs_retrieval if state.k_outputs_retrieval else k_outputs,
        search_type= state.retrieval_search_type if state.retrieval_search_type else base_type_search,
        )
    
    state.retrieval_results = '\n------------------\n'.join(
        [f'{doc.page_content}\n\
            Date: {doc.metadata[date_col_name]}\
            Main tags: {doc.metadata[main_tags_col_name]}\
            Sub tags: {doc.metadata[sub_tags_col_name]}'
        ] for doc in retrieved_docs
    )
    

