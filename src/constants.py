import os 
import pandas as pd
import datetime 
import torch

from langchain_text_splitters import RecursiveCharacterTextSplitter

###                    ###               
### SignLog constants  ###
###                    ###
user_email = '' 
user_password = '' 
verify_password = '' 

###                    ###               
###   Data constants   ###
###                    ###
text_date = ''
tags_separator = '//'
main_tags = ''
sub_tags = ''
text_entry = ''

user_table = pd.DataFrame()
retrieval_data = pd.DataFrame()
entry_id_col_name = 'entry_id'
chunked_entry_id_col_name = 'chunked_entry_id'
date_col_name = 'text_date'
main_tags_col_name = 'main_tags'
sub_tags_col_name = 'sub_tags'
doc_type_col_name = 'doc_type'
text_col_name = 'text_entry'
filter_dates = [None, None]
user_main_tags = []
user_sub_tags = []
filter_strictness_choices = ['any tags', 'all tags']
filter_strictness = filter_strictness_choices[0]

filter_main_tags = []
filter_sub_tags = []

chunk_size = 1024
chunk_overlap = 256
chunk_separators = [
    "\n\n",
    "\n",
    ".",
]
recursive_character_text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            is_separator_regex=False,
            separators=chunk_separators
           )



retrieval_query = ''
retrieval_search_type_possibilities = ['similarity', 'similarity_score_threshold', 'mmr']
retrieval_search_type = retrieval_search_type_possibilities[0]
retrieval_filter_strictness_choices = ['any tags', 'all tags']
retrieval_filter_strictness = retrieval_filter_strictness_choices[0]
retrieval_rerank = 'flashrank'
k_outputs_retrieval = 1
retrieval_main_tags = []
retrieval_sub_tags = []
retrieval_results = ''

lang_user_vdb = None    
RAGentic = None     

entry_delimiter = '----'
file_tags_separator = '//'
date_delimiter = 'DATE: '
main_tags_delimiter = 'MAIN_TAGS: '
sub_tags_delimiter = 'SUB_TAGS: '
text_delimiter = 'TEXT: '
file_path_to_load = None

pdf_date = ''
pdf_tags_separator = '//'
pdf_main_tags = ''
pdf_sub_tags = ''
pdf_path_to_load = ''
user_pdf_names_ids = {}
                
###                                ###               
###   Language for image to text   ###
###                                ###

image_to_text_cpu_or_gpu = 'CPU'
selected_languages = ['English']
selected_image_paths = None

image_to_text_languages = {
    'Abaza': 'abq',
    'Adyghe': 'ady',
    'Afrikaans': 'af',
    'Angika': 'ang',
    'Arabic': 'ar',
    'Assamese': 'as',
    'Avar': 'ava',
    'Azerbaijani': 'az',
    'Belarusian': 'be',
    'Bulgarian': 'bg',
    'Bihari': 'bh',
    'Bhojpuri': 'bho',
    'Bengali': 'bn',
    'Bosnian': 'bs',
    'Simplified Chinese': 'ch_sim',
    'Traditional Chinese': 'ch_tra',
    'Chechen': 'che',
    'Czech': 'cs',
    'Welsh': 'cy',
    'Danish': 'da',
    'German': 'de',
    'English': 'en',
    'Spanish': 'es',
    'Estonian': 'et',
    'Persian': 'fa',
    'Finnish': 'fi',
    'French': 'fr',
    'Irish': 'ga',
    'Goan Konkani': 'gom',
    'Hindi': 'hi',
    'Croatian': 'hr',
    'Hungarian': 'hu',
    'Indonesian': 'id',
    'Ingush': 'inh',
    'Icelandic': 'is',
    'Italian': 'it',
    'Japanese': 'ja',
    'Kabardian': 'kbd',
    'Kannada': 'kn',
    'Korean': 'ko',
    'Kurdish': 'ku',
    'Latin': 'la',
    'Lak': 'lbe',
    'Lezghian': 'lez',
    'Lithuanian': 'lt',
    'Latvian': 'lv',
    'Magahi': 'mah',
    'Maithili': 'mai',
    'Maori': 'mi',
    'Mongolian': 'mn',
    'Marathi': 'mr',
    'Malay': 'ms',
    'Maltese': 'mt',
    'Nepali': 'ne',
    'Newari': 'new',
    'Dutch': 'nl',
    'Norwegian': 'no',
    'Occitan': 'oc',
    'Pali': 'pi',
    'Polish': 'pl',
    'Portuguese': 'pt',
    'Romanian': 'ro',
    'Russian': 'ru',
    'Serbian (cyrillic)': 'rs_cyrillic',
    'Serbian (latin)': 'rs_latin',
    'Nagpuri': 'sck',
    'Slovak': 'sk',
    'Slovenian': 'sl',
    'Albanian': 'sq',
    'Swedish': 'sv',
    'Swahili': 'sw',
    'Tamil': 'ta',
    'Tabassaran': 'tab',
    'Telugu': 'te',
    'Thai': 'th',
    'Tajik': 'tjk',
    'Tagalog': 'tl',
    'Turkish': 'tr',
    'Uyghur': 'ug',
    'Ukranian': 'uk',
    'Urdu': 'ur',
    'Uzbek': 'uz',
    'Vietnamese': 'vi'
}

image_to_text_output = ''


###                         ###               
### Miscellanous constants  ###
###                         ###

notify_duration = 8000 #mseconds

sqlite_tags_separator = ','

###                         ###               
###      LLM constants      ###
###                         ###

ollama_llms = [
    #https://github.com/ollama/ollama#model-library
    'llama3.1', # 8b
    'llama3.1:70b',
    'llama3.1:405b',
    'phi3', # 3.8b
    'phi3:medium', # 14b
    'gemma2', #9b
    'gemma2:27b',
    'mistral', # 7b
    'moondream', # 1.4b
    'neural-chat', # 7b
    'starling-lm', # 7b
    'codellama', # 7b
    'llama2-uncensored', # 7b
    'llava', # 7b
    'solar', # 10.7b        
]

llm_name = 'llama3.1'
llm_temperature = "0.1"

device = 'cuda' if torch.cuda.is_available() else 'cpu'
# ！The default dimension is 1024.
# if you need other dimensions, please clone the model (https://huggingface.co/dunzhang/stella_en_400M_v5) and modify `modules.json` 
# to replace `2_Dense_1024` with another dimension, e.g. `2_Dense_256` or `2_Dense_8192` (512, 768, 1024, 2048, 4096, 6144 and 8192) !
vector_dimensions = 1024
embeddings_model_name = "infgrad/stella_en_400M_v5"
stella_en_embeddings_query_prompt_query = "s2p_query"
stella_en_embeddings_query_prompt_semantic = "s2s_query"
# "infgrad/stella_en_400M_v5" model supports two prompts: "s2p_query" and "s2s_query" for sentence-to-passage and sentence-to-sentence tasks, respectively.
def embeddings_query_prompt(mode:str):
    if mode in ['semantic', 'query']:
        return stella_en_embeddings_query_prompt_semantic if mode == 'semantic' else stella_en_embeddings_query_prompt_query
    else:
        return stella_en_embeddings_query_prompt_query
    
sparse_embeddings_model_name = "Qdrant/bm42-all-minilm-l6-v2-attentions"
vdb = 'qdrant'
retrieval_mode = 'hybrid' # 'dense', 'sparse', 'hybrid'

relevance_threshold = 0.5
mmr_fetch_k = 50 # documents for the MMR algorithm to consider
mmr_lambda_mult = 0.5 #Diversity of results returned by MMR; 1 for minimum diversity and 0 for maximum.

# For langchain indexing
sql_record_manager_path = "sqlite:///conf/record_manager_cache.sql"

## RAG
max_chat_history_tokens = 20000

rag_input_active = True
rag_column_shown_in_table = 'RAG_conversation'
ai_role = 'ai'
human_role = 'human'
rag_dict = {'RAG': {rag_column_shown_in_table:[], 'role':[]}}
rag_conversation_table = pd.DataFrame(columns=[rag_column_shown_in_table, 'role'])
rag_current_user_query = ''
rag_considered_docs_choices = ['pdf', 'text', 'all']
rag_considered_docs = 'all'
rag_considered_pdfs = []
rag_ai_response = ''


rag_retrieval_search_type = retrieval_search_type_possibilities[0]
rag_retrieval_filter_strictness = retrieval_filter_strictness_choices[0]
rag_retrieval_rerank = 'flashrank'
rag_k_outputs_retrieval = "10"
rag_retrieval_main_tags = []
rag_retrieval_sub_tags = []

rag_response_unrelevant_retrieved_docs = "I'm sorry, the retrieved documents are not relevant enought to provide a thorough answer."

# Assuming we are in src\constants.py
root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
credentials_yaml_path = os.path.join(root_path, 'conf', 'app_credentials.yaml')
sqlite_database_path = os.path.join(root_path, 'conf', 'data_sqlite.db')
qdrant_database_path = os.path.join(root_path, 'conf', 'data_qdrant.db')