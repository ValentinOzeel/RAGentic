import os 
import pandas as pd
import datetime 

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
date_col_name = 'text_date'
main_tags_col_name = 'main_tags'
sub_tags_col_name = 'sub_tags'
text_col_name = 'text_entry'
filter_dates = [datetime.datetime(2000, 1, 1).strftime("%m/%d/%Y"),  datetime.datetime.now().strftime("%m/%d/%Y")]

entry_delimiter = '----'
file_tags_separator = '//'
date_delimiter = 'DATE:'
main_tags_delimiter = 'MAIN_TAGS:'
sub_tags_delimiter = 'SUB_TAGS:'
text_delimiter = 'TEXT:'
text_file_to_load = None

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

notify_duration = 5000 #mseconds

# Assuming we are in src\constants.py
root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
credentials_yaml_path = os.path.join(root_path, 'conf', 'app_credentials.yaml')
json_data_folder_path = os.path.join(root_path, 'conf', 'user_files')