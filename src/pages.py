import random

import taipy.gui.builder as tgb
from taipy.gui import Icon

from callbacks import (
            on_login, on_sign_in,
            on_text_entry_add,
            on_image_to_text,
            on_txt_file_load,
            on_pdf_file_load,
            on_filter_date,
            on_filter_tags,
            on_reset_filters,
            on_retrieval_query,
        )

from page_ids import page_ids
from constants import (
    user_email, user_password, verify_password, 
    text_date, tags_separator, main_tags, sub_tags, text_entry, 
    image_to_text_languages, image_to_text_cpu_or_gpu, selected_languages, selected_image_paths, image_to_text_output,
    user_table, user_main_tags, user_sub_tags, filter_dates, filter_strictness_choices, filter_strictness, filter_main_tags, filter_sub_tags,
    entry_delimiter, file_tags_separator, date_delimiter, main_tags_delimiter, sub_tags_delimiter, text_delimiter, file_path_to_load, 
    pdf_date, pdf_tags_separator, pdf_main_tags, pdf_sub_tags, pdf_path_to_load, user_pdf_names_ids,
    lang_user_vdb, RAGentic, retrieval_query, k_outputs_retrieval, retrieval_rerank, retrieval_search_type_possibilities, retrieval_search_type, retrieval_filter_strictness_choices, retrieval_filter_strictness, retrieval_main_tags, retrieval_sub_tags, retrieval_results,
    rag_considered_docs_choices, rag_considered_docs, rag_considered_pdfs
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
            tgb.login("Please log in! (reload page to retry)", on_action=on_login)




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
        tgb.input("{tags_separator}", label='Separator (fill fields with several values)')
        tgb.input("{main_tags}", label='Main tags')
        tgb.input("{sub_tags}", label='Sub tags')
        tgb.input("{text_entry}", label='Text to add*')
        tgb.button("Add data entry", on_action=on_text_entry_add)
    
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
        tgb.text("{image_to_text_output}")
    
    
    tgb.text("## Load an entire text file:", mode="md")
    with tgb.layout("1 1 1 1 1 1 1"):
        tgb.input("{entry_delimiter}", label='Entry delimiter*')
        tgb.input("{file_tags_separator}", label='Multi-tags separator')
        tgb.input("{date_delimiter}", label='Date delimiter')
        tgb.input("{main_tags_delimiter}", label='Main tags delimiter')
        tgb.input("{sub_tags_delimiter}", label='Sub tags delimiter')
        tgb.input("{text_delimiter}", label='Text delimiter*')
        tgb.file_selector("{file_path_to_load}", 
                          label="Upload your text file*", 
                          extensions=".txt", drop_message="Drop your txt file here", 
                          on_action=on_txt_file_load)
        
    tgb.text("## Load a PDF file:", mode="md")
    with tgb.layout("1 1 1 1 1"):
        tgb.input("{pdf_date}", label='PDF date')
        tgb.input("{pdf_tags_separator}", label='Separator (fill fields with several values)')
        tgb.input("{pdf_main_tags}", label='Main tags')
        tgb.input("{pdf_sub_tags}", label='Sub tags')
        tgb.file_selector("{pdf_path_to_load}", 
                          label="Upload your PDF file*", 
                          extensions=".pdf", drop_message="Drop your PDF file here", 
                          on_action=on_pdf_file_load)
        
with tgb.Page() as retrieve_data:
    tgb.text("## Your data:", mode="md")
    tgb.table("{user_table}", editable=True, page_size=50, allow_all_rows=True)

    tgb.text("## Filter your data:", mode="md")
    
    with tgb.layout("1 1 1 1 1 1"):
        tgb.date_range("{filter_dates}", label_start="From:", label_end="To:", on_change=on_filter_date)

        tgb.text(' ')

        tgb.selector(value="{filter_strictness}", 
                     lov="{filter_strictness_choices}", 
                     multiple=False, dropdown=True, label='Filter strictness')
        
        tgb.selector(value="{filter_main_tags}", 
                     lov="{user_main_tags}", 
                     multiple=True, dropdown=True, label='Main tag filter',
                     on_change=on_filter_tags)
        
        tgb.selector(value="{filter_sub_tags}", 
                     lov="{user_sub_tags}", 
                     multiple=True, dropdown=True, label='Sub tag filter',
                     on_change=on_filter_tags)
        
        tgb.button("Reset table", on_action=on_reset_filters)
        
        
    tgb.text('\n \n ')
    tgb.text("## Retrieve data with query:", mode="md")
    tgb.text('\n \n ')
    
    with tgb.layout("1 1 1 1 1 1"):
        tgb.selector(value="{retrieval_search_type}", 
                     lov=retrieval_search_type_possibilities, 
                     multiple=False, dropdown=True, label='Retrieval search type')
        
        tgb.selector(value="{retrieval_filter_strictness}", 
                     lov="{retrieval_filter_strictness_choices}", 
                     multiple=False, dropdown=True, label='Retrieval filter strictness')
        
        tgb.selector(value="{retrieval_rerank}", 
                     lov=[(False, 'No rerank'), ('flashrank', 'flashrank'), ('rag_fusion', 'rag_fusion')], 
                     multiple=False, dropdown=True, label='Rerank usage')

        tgb.slider("{k_outputs_retrieval}", min=1, max=10, hover_text='n retrieval outputs', labels={n+1:n+1 for n in range(10)})

        tgb.selector(value="{retrieval_main_tags}", 
                     lov="{user_main_tags}", 
                     multiple=True, dropdown=True, label='Retrieval main tag filter')
        
        tgb.selector(value="{retrieval_sub_tags}", 
                     lov="{user_sub_tags}", 
                     multiple=True, dropdown=True, label='Retrieval sub tag filter')
        
    with tgb.layout("1 1 1"):
        
        tgb.text('     ')
        
        with tgb.part():
            tgb.input("{retrieval_query}", label='Enter your query*')
            tgb.button("Send query", on_action=on_retrieval_query)
        
    tgb.text('{retrieval_results}', mode='pre')
    
  
with tgb.Page() as rag_app:
    with tgb.layout("1 2"):
        with tgb.part("1"):
            # This part acts like a sidebar
            tgb.text("## RAG Parameters", mode="md")
            
            tgb.selector(value="{llm_name}", 
                         lov="{ollama_llms}", 
                         multiple=False, dropdown=True, label='LLM used')
            
            tgb.indicator("LLM temperature", value="{llm_temperature}", min="0", max="1")
            
            
            tgb.selector(value="{rag_considered_docs}", 
                         lov="{rag_considered_docs_choices}", 
                         multiple=False, dropdown=True, label='Documents considered')
            
            if rag_considered_docs == 'PDFs':
                tgb.selector(value="{rag_considered_pdfs}", 
                             lov="{user_pdf_names_ids.keys()}", 
                             multiple=True, dropdown=True, label='PDFs considered')

            tgb.selector(value="{rag_retrieval_search_type}", 
                         lov=retrieval_search_type_possibilities, 
                         multiple=False, dropdown=True, label='Retrieval search type')

            tgb.slider("{rag_k_outputs_retrieval}", min=1, max=10, hover_text='n retrieval outputs', labels={n+1:n+1 for n in range(10)})
            
            tgb.selector(value="{rag_retrieval_rerank}", 
                         lov=[(False, 'No rerank'), ('flashrank', 'flashrank'), ('rag_fusion', 'rag_fusion')], 
                         multiple=False, dropdown=True, label='Rerank usage')

            tgb.selector(value="{rag_retrieval_filter_strictness}", 
                         lov="{retrieval_filter_strictness_choices}", 
                         multiple=False, dropdown=True, label='Retrieval filter strictness')

            tgb.selector(value="{rag_retrieval_main_tags}", 
                         lov="{user_main_tags}", 
                         multiple=True, dropdown=True, label='Retrieval main tag filter')

            tgb.selector(value="{rag_retrieval_sub_tags}", 
                         lov="{user_sub_tags}", 
                         multiple=True, dropdown=True, label='Retrieval sub tag filter')

        with tgb.part("2"):
            # Main content
            tgb.text("## RAG app:", mode="md")
            tgb.table("{rag_conversation_table}", editable=False, width=100, show_all=True)  
            tgb.input("{rag_current_user_query}", label='Enter your message here...')
    
    

                      
#
# LET USER CHOOSE LLM, TEMPERTURE, USE REWRITE USER PROMPT OR NOT

#TABLE ADD FILTER KEYWORDS IN TEXT