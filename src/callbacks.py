import taipy.gui.builder as tgb
from taipy.gui import navigate, notify

import os
import pandas as pd
import copy

from langchain.callbacks.base import BaseCallbackHandler

from constants import (
    notify_duration, 
    entry_id_col_name, date_col_name, main_tags_col_name, sub_tags_col_name, doc_type_col_name, 
    sqlite_tags_separator,
    filter_strictness_choices, retrieval_search_type, rag_retrieval_search_type,
    rag_column_shown_in_table, ai_role, human_role
    )
from page_ids import page_ids
from tools import SignLog as sl, YamlManagment as ym, SQLiteManagment as sm, image_to_text_conversion, LangVdb as lvdb, RAGentic



class LLMResponseStreamingHandler(BaseCallbackHandler):
    def __init__(self, state):
        self.state = state
        self.current_llm_response_tokens = []
    
    def _update_table(self, token=None, final_str=None):
        # Append response token
        if token:
            self.current_llm_response_tokens.append(token)
            # Join all generated tokens up to now in a string
            self.state.rag_ai_response = "".join(self.current_llm_response_tokens)
        elif final_str:
            self.state.rag_ai_response = final_str
        # Append in taipy's state RAG conversation list
        self.state.RAGentic.chat_dict['RAG'][rag_column_shown_in_table].append(self.state.rag_ai_response)
        self.state.RAGentic.chat_dict['RAG']['role'].append(ai_role)
        # Actualize the response shown in the app, in essence enabling streaming
        self.state.rag_conversation_table = pd.DataFrame(self.state.RAGentic.chat_dict['RAG'])
        
    def on_llm_new_token(self, token: str, **kwargs) -> None:
        self._update_table(token=token)
        # Remove the last string added in taipy's state RAG conversation list so that 
        # every added token doesnt count as a complete ai response
        self.state.RAGentic.chat_dict['RAG'][rag_column_shown_in_table].pop()
        self.state.RAGentic.chat_dict['RAG']['role'].pop()
        
    def on_llm_end(self, response: str, **kwargs) -> None:
        ai_response = "".join(self.current_llm_response_tokens)
        final_response = f"{ai_response} ___Retrieved document IDs___: {self.state.RAGentic.last_retrieved_doc_IDs_str}"
        self._update_table(final_str=final_response)



    
    
def style_rag(state, idx: int, row: int) -> str:
    """
    Apply a style to the conversation table depending on the message's author.
    Args:
        - state: The current state of the app.
        - idx: The index of the message in the table.
        - row: The row of the message in the table.
    Returns:
        The style to apply to the message.
    """
    if idx is None: 
        return None
    
    role = state.rag_conversation_table.at[idx, "role"] if state.rag_conversation_table.at[idx, "role"] else None

    if role == human_role:
        return 'human_message'
    elif role == ai_role:
        return 'ai_message'
        
    



    
    
    
    

def _get_user_tags(state):
    print(f"--{tags}--" for tags in state.user_table[main_tags_col_name])
    # Get all unique user's main and sub-tags values
    if isinstance(state.user_table, pd.DataFrame):
        state.user_main_tags = sorted(list(set(tag for main_tag_list in state.user_table[main_tags_col_name] if main_tag_list is not None for tag in main_tag_list.split(' ') if tag)))
        state.user_sub_tags = sorted(list(set(tag for sub_tag_list in state.user_table[sub_tags_col_name] if sub_tag_list is not None for tag in sub_tag_list.split(' ') if tag))) 
    return state

def _delete_loaded_file(file_path):
    os.remove(file_path)

def on_sign_in(state, id, login_args):    
    # State corresponding values
    acc_creation_values = [state.user_email, state.user_password, state.verify_password]

    if all(element == 'dev' for element in acc_creation_values):
        # Add credentials except if user already exists
        status = ym.add_user_credentials(state.user_email, state.user_password)
        if not status:
            notify(state, 'error', 'An account is already associated to this email adress.', duration=notify_duration)
        else:
            # Create a vector database collection for the new user
    #        lvdb.initialize_vdb_collection(state.user_email)
            notify(state, 'success', 'Account successfully created!', duration=notify_duration)
            navigate(state, page_ids["log_in"])
        
    else:
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
                status = ym.add_user_credentials(state.user_email, state.user_password)
                if not status:
                    notify(state, 'error', 'An account is already associated to this email adress.', duration=notify_duration)
                else:
                    # Create a vector database collection for the new user
      #              lvdb.initialize_vdb_collection(state.user_email)
                    notify(state, 'success', 'Account successfully created!', duration=notify_duration)
                    navigate(state, page_ids["log_in"])

                
def on_login(state, id, login_args):
    user_email, user_password = login_args["args"][:2]
    # Check whether fields have been filled
    if user_email is None or user_password is None:
        notify(state, 'error', 'None of email adress and/or password fields should be empty.', duration=notify_duration)
        
    elif not ym.check_user_credentials(user_email, user_password):
        notify(state, 'error', 'Incorrect email adress and/or password.', duration=notify_duration)
        
    else:
        state.user_email, state.user_password = user_email, user_password
        # Set user_table variable as user's sqlite data loaded into df 
        state.user_table = sm.sqlite_to_dataframe(state.user_email)
        # Get all unique user's main and sub-tags values
        state = _get_user_tags(state)
        # Access vdb collection
        state.lang_user_vdb = lvdb.access_vdb(state.user_email)
        # Access user's pdfs
        state.user_pdf_names_ids = ym.get_user_pdf_names_ids(state.user_email)
        # Initiate RAGentic objet in state
        state.RAGentic = RAGentic()
        notify(state, 'success', 'Successful authentification.')
        navigate(state, page_ids["root_page"])
    

    
def on_text_entry_add(state, action, info):   
    if not state.text_entry:
        notify(state,'error', 'You must at least fill the text field (indicated with an *).')
    
    if state.tags_separator:
        main_tags = [tag for tag in state.main_tags.split(state.tags_separator) if state.tags_separator in state.main_tags] if state.main_tags else state.main_tags
        sub_tags = [tag for tag in state.sub_tags.split(state.tags_separator) if state.tags_separator in state.sub_tags] if state.sub_tags else state.sub_tags
    else:
        main_tags, sub_tags = state.main_tags, state.sub_tags

    lvdb.text_to_db(
        state.user_email, 
        state.lang_user_vdb,
        single_entry_load_kwargs={
                'text_date':state.text_date, 
                'main_tags':main_tags, 
                'sub_tags':sub_tags, 
                'text_entry':state.text_entry
                }
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
    if not all([state.entry_delimiter, state.text_delimiter, state.file_path_to_load]):
        return notify(state,'error', 'You must at least provide entry_delimiter, text_delimiter and the text file.', duration=notify_duration)
    
    if not all([state.date_delimiter, state.file_tags_separator, state.main_tags_delimiter, state.sub_tags_delimiter]):
        notify(state,'warning', 'Some fields were left empty.', duration=notify_duration)
    
    try:

        lvdb.text_to_db(
            state.user_email, 
            state.lang_user_vdb,
            text_file_load_kwargs={
                'file_path':state.file_path_to_load,
                'entry_delimiter':state.entry_delimiter,
                'file_tags_separator':state.file_tags_separator,
                'date_delimiter':state.date_delimiter,
                'main_tags_delimiter':state.main_tags_delimiter,
                'sub_tags_delimiter':state.sub_tags_delimiter,
                'text_delimiter':state.text_delimiter
                }
        )

        # upadte table
        state.user_table = sm.sqlite_to_dataframe(state.user_email)
        # Get all unique user's main and sub-tags values
        state = _get_user_tags(state)
        # Delete loaded file
        _delete_loaded_file(state.file_path_to_load)
        
        notify(state,'success', 'Loaded text file in databases', duration=notify_duration)
        
    except Exception as e:
        print(e)
        return notify(state,'error', "The file couldn't be loaded in database", duration=notify_duration)
    

        
        
def on_pdf_file_load(state, id, payload):
    if state.pdf_tags_separator:
        pdf_main_tags = (
            state.pdf_main_tags.split(state.pdf_tags_separator) 
            if state.pdf_tags_separator in state.pdf_main_tags 
            else [state.pdf_main_tags]
        ) if state.pdf_main_tags else []
        
        pdf_sub_tags = (
            state.pdf_sub_tags.split(state.pdf_tags_separator)
            if state.pdf_tags_separator in state.pdf_sub_tags
            else [state.pdf_sub_tags]
        ) if state.pdf_sub_tags else []
        

    else:
        pdf_main_tags, pdf_sub_tags = state.pdf_main_tags, state.pdf_sub_tags

    try:
        # Add pdf in sqlite and vdb
        lvdb.pdf_to_db(
            state.user_email, 
            state.lang_user_vdb, 
            state.pdf_path_to_load, state.pdf_date, pdf_main_tags, pdf_sub_tags, 
            vdb_add=True, sqlite_add=True
            )

        # Udpate dataframe shown
        state.user_table = sm.sqlite_to_dataframe(state.user_email)
        # Get all unique user's main and sub-tags values
        state = _get_user_tags(state)
        # Delete loaded file
        _delete_loaded_file(state.pdf_path_to_load)
        state.pdf_path_to_load = ''
        # Update user's pdfs
        state.user_pdf_names_ids = ym.get_user_pdf_names_ids(state.user_email)
        # Notify success
        notify(state, 'success', 'PDF added to databases.', duration=notify_duration)
        
    except Exception as e:
        print('_on_pdf_load fail: ', e)
        return notify(state,'error', "The file couldn't be loaded in databases", duration=notify_duration)






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
    state.retrieval_results = state.RAGentic.retrieval(
        query= state.retrieval_query,
        lang_vdb= state.lang_user_vdb,
        rerank= False if state.retrieval_rerank.lower() == 'false' else state.retrieval_rerank.lower(),
        filters= {main_tags_col_name:state.retrieval_main_tags, sub_tags_col_name:state.retrieval_sub_tags},
        k_outputs= state.k_outputs_retrieval,
        search_type= state.retrieval_search_type if state.retrieval_search_type else retrieval_search_type,
        filter_strictness= state.retrieval_filter_strictness,
        format_results='str'
        )
    
    
def on_rag_input(state, id, payload):  
    # If llm is generating an answer, prevent the user from adding a new query
    if not state.rag_input_active:
        notify(state,'info', 'Please wait for the llm to generate its answer first.', duration=notify_duration)
        return 
    # If chat_dict is empty, initiate it
    if not state.RAGentic.chat_dict:
        state.RAGentic.chat_dict['RAG'] = {
            rag_column_shown_in_table: [],
            'role': []
        }
    
    # Copy user query and actualize rag_current_user_query so that we can immediately remove it from the app's user input section
    rag_current_user_query = copy.copy(state.rag_current_user_query)
    state.rag_current_user_query = ''
    
    # Inactivate RAG input
    state.rag_input_active = False
    
    # Append user's query in RAG conversation list and show it in the app
    state.RAGentic.chat_dict['RAG'][rag_column_shown_in_table].append(rag_current_user_query)
    state.RAGentic.chat_dict['RAG']['role'].append(human_role)
    state.rag_conversation_table = pd.DataFrame(state.RAGentic.chat_dict['RAG'])
    
    
    # Build LLM params dict  
    llm_params = {
        'model': state.llm_name,
        'temperature': float(state.llm_temperature)
    }
    # Build retrieval filters dict
    filters = {
            main_tags_col_name:state.rag_retrieval_main_tags, 
            sub_tags_col_name:state.rag_retrieval_sub_tags, 
            doc_type_col_name: ['pdf', 'text'] if state.rag_considered_docs == 'all' else state.rag_considered_docs, 
            entry_id_col_name:[ym.get_user_pdf_names_ids(state.user_email)[pdf_name] for pdf_name in state.rag_considered_pdfs] if state.rag_considered_docs in ['pdf', 'all'] else None
        }   
    # Build retrieval parameters dict
    retrieval_params = {
        'search_type': state.rag_retrieval_search_type if state.rag_retrieval_search_type else rag_retrieval_search_type,
        'k_outputs': int(state.rag_k_outputs_retrieval),
        'rerank': False if state.rag_retrieval_rerank.lower() == 'false' else state.rag_retrieval_rerank.lower(),
        'filter_strictness': state.rag_retrieval_filter_strictness,
        'filters': filters,
    }
    
    
    print('\n\n\n', 'RETRIEVAL PARAMS: ', retrieval_params, '\n\n\n')
    
    # Instanciate a fresh streaming callback
    my_stream_callback = LLMResponseStreamingHandler(state)
    # Call the llm with user's query
    ai_response_str, retrieved_doc_IDs_str = state.RAGentic.rag_call(
        state.lang_user_vdb,
        rag_current_user_query,
        llm_params,
        retrieval_params,
        my_stream_callback
    )

    # Activate RAG input
    state.rag_input_active = True
            

    
    
