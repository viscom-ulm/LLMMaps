import json
import numpy as np
import pickle
import argparse
import time
import os
import random
import pathlib
import re
my_chatgpt_api_key = None
import openai
if my_chatgpt_api_key:
    openai.api_key = my_chatgpt_api_key
else:
    raise ValueError('You selected ChatGPT as model, please also add an API_KEY to use it.')

parser = argparse.ArgumentParser()
parser.add_argument('--path', help='path of the preprocessed QA data file in json format', type=str, default='../data/pubmedqa.qar')
parser.add_argument('--only_bloom', help='generate only the bloom_taxonomy', action='store_true')
parser.add_argument('--only_diff', help='generate only the difficulty ranking', action='store_true')
parser.add_argument('--max_iterations', type=int, default=5, help='Sets max number of iterations over the dataset. Sometimes not all questions result in automatically readable results, these will form the set for the next itearation.', action='store_true')

args = parser.parse_args()

data_path = args.path 
dataset_name = os.path.basename(data_path)[:-4]

q_data = json.load(open(data_path))
q_keys = list(q_data.keys())
q_type = q_data[q_keys[0]]['question_type']

def get_choice_string(choices):
    '''
    Transforms the list of choices into a string with one choice per line, choices numbered starting at 1.
    '''
    return '\n'.join(f'{i+1}. '+choice for i,choice in enumerate(choices))

class define_question_style():
    '''
    This class defines the question_style object. This includes checking for context info in all samples and setting of string bases, which are then formatted with the sample input.
    '''

    def __init__(self, q_data, q_keys, q_type):
        self.q_data = q_data
        self.q_keys = q_keys
        self.q_type = q_type

        #Additional system message necessary for the CHatGPT API
        self.system_message = 'You are a helpful assistant fullfilling the tasks given to him according to how he is instructed.'

        if q_type == 'MC':    
            self.difficulty_base = ('Consider the following multiple choice question with it\'s answer options. Rank the question according to it\'s difficulty level on a scale from 1 to 5, '
                            + 'where 1 is easy and 5 is hard. Answer only with one digit for the difficulty level.\n\n'
                            + 'Question: {}\nChoices:\n {}\n\nDifficulty Level:')
            self.bloom_base = ('Consider Bloom\'s taxonomy of educational objectives, which categorizes the educational objectives of tasks or test questions into 6 main categories:\n'
                            +'Remember\nUnderstand\nApply\nAnalyze\nEvaluate\nCreate\n'
                            +'For a better assessment of these categories, they can be further split down into the following subcategories:\n'
                            +'Remember\n  -Recognizing\n  -Recalling\n'
                            +'Understand\n  -Interpreting\n  -Exemplifying\n  -Classifying\n  -Summarizing\n  -Inferring\n  -Comparing\n  -Explaining\n'
                            +'Apply\n  -Executing\n  -Implementing\n'
                            +'Analyze\n  -Differentiating\n  -Organizing\n  -Attributing\n'
                            +'Evaluate\n  -Checking\n  -Critiquing\n'
                            +'Create\n  -Generating\n  -Planning\n  -Producing\n'
                            +'What is the main category in Bloom\'s taxonomy of the following multiple choice question along with it\'s context information?'
                            +'Question:\n {}\nChoices:\n {}\n\nBloom Category:')

            # Check is context information is available across the dataset. Note that some processed datasets may contain unwanted context for these tasks depending on how import is set up. 
            # The latter is the case when the context info transforms a knowledge QA task into a text understanding task. e.g. SciQ. These should not be incorporated for any task in our work.
            self.is_contextful = True
            for key in q_keys:
                if q_data[key]['context'] is None or q_data[key]['context']=='':
                    self.is_contextful = False

            if self.is_contextful:
                self.difficulty_base = ('Consider the following multiple choice question with it\'s context information. Rank the question according to it\'s difficulty level on a scale from 1 to 5, '
                            + 'where 1 is easy and 5 is hard. Answer only with one digit for the difficulty level.\n\n'
                            + 'Question: {}\n\nContext: {}\nChoices:\n {}\n\nDifficulty Level:')
                self.bloom_base = ('Consider Bloom\'s taxonomy of educational objectives, which categorizes the educational objectives of tasks or test questions into 6 main categories:\n'
                            +'Remember\nUnderstand\nApply\nAnalyze\nEvaluate\nCreate\n'
                            +'For a better assessment of these categories, they can be further split down into the following subcategories:\n'
                            +'Remember\n  -Recognizing\n  -Recalling\n'
                            +'Understand\n  -Interpreting\n  -Exemplifying\n  -Classifying\n  -Summarizing\n  -Inferring\n  -Comparing\n  -Explaining\n'
                            +'Apply\n  -Executing\n  -Implementing\n'
                            +'Analyze\n  -Differentiating\n  -Organizing\n  -Attributing\n'
                            +'Evaluate\n  -Checking\n  -Critiquing\n'
                            +'Create\n  -Generating\n  -Planning\n  -Producing\n'
                            +'What is the main category in Bloom\'s taxonomy of the following multiple choice question along with it\'s context information?'
                            +'Question: {}\n\nContext: {}\nChoices:\n {}\n\nBloom Category:')
            
        else:
            self.difficulty_base = ('Consider the following question. Rank the question according to it\'s difficulty level on a scale from 1 to 5, '
                            + 'where 1 is easy and 5 is hard. Answer only with one digit for the difficulty level.\n\n'
                            + 'Question: {}\n\nDifficulty Level:')
            self.bloom_base = ('Consider Bloom\'s taxonomy of educational objectives, which categorizes the educational objectives of tasks or test questions into 6 main categories:\n'
                            +'Remember\nUnderstand\nApply\nAnalyze\nEvaluate\nCreate\n'
                            +'For a better assessment of these categories, they can be further split down into the following subcategories:\n'
                            +'Remember\n  -Recognizing\n  -Recalling\n'
                            +'Understand\n  -Interpreting\n  -Exemplifying\n  -Classifying\n  -Summarizing\n  -Inferring\n  -Comparing\n  -Explaining\n'
                            +'Apply\n  -Executing\n  -Implementing\n'
                            +'Analyze\n  -Differentiating\n  -Organizing\n  -Attributing\n'
                            +'Evaluate\n  -Checking\n  -Critiquing\n'
                            +'Create\n  -Generating\n  -Planning\n  -Producing\n'
                            +'What is the main category in Bloom\'s taxonomy of the following question along with it\'s context information?'
                            +'Question: {}\n\nBloom Category:')

            #Check is context information is available across the dataset
            self.is_contextful = True
            for key in q_keys:
                if q_data[key]['context'] is None or q_data[key]['context']=='':
                    self.is_contextful = False

            if self.is_contextful:
                self.difficulty_base = ('Consider the following question with it\'s context information. Rank the question according to it\'s difficulty level on a scale from 1 to 5, '
                            + 'where 1 is easy and 5 is hard. Answer only with one digit for the difficulty level.\n\n'
                            + 'Question: {}\n\nContext: {}\n\nDifficulty Level:')
                self.bloom_base = ('Consider Bloom\'s taxonomy of educational objectives, which categorizes the educational objectives of tasks or test questions into 6 main categories:\n'
                            +'Remember\nUnderstand\nApply\nAnalyze\nEvaluate\nCreate\n'
                            +'For a better assessment of these categories, they can be further split down into the following subcategories:\n'
                            +'Remember\n  -Recognizing\n  -Recalling\n'
                            +'Understand\n  -Interpreting\n  -Exemplifying\n  -Classifying\n  -Summarizing\n  -Inferring\n  -Comparing\n  -Explaining\n'
                            +'Apply\n  -Executing\n  -Implementing\n'
                            +'Analyze\n  -Differentiating\n  -Organizing\n  -Attributing\n'
                            +'Evaluate\n  -Checking\n  -Critiquing\n'
                            +'Create\n  -Generating\n  -Planning\n  -Producing\n'
                            +'What is the main category in Bloom\'s taxonomy of the following question along with it\'s context information?'
                            +'Question: {}\n\nContext: {}\n\nBloom Category:')

QUESTION_STYLE = define_question_style(q_data, q_keys, q_type)

# Load existing partial completions and some logs which are appended
# The script will collect some info in temporary saves. These will remain unused after succesfull completion of this code like everything in data/temp.

temp_base_path = pathlib.Path(__file__).parents[0] / '..' / 'data' / 'temp' 
temp_base_path.mkdir(parents=True, exist_ok=True)
temp_savepath_bloom_taxonomy = temp_base_path / '{}_bloom_taxonomy.pkl'.format(dataset_name)
temp_savepath_difficulty_level = temp_base_path / '{}_difficulty_level.pkl'.format(dataset_name)

if os.path.isfile(temp_savepath_bloom_taxonomy):
    bloom_dict = pickle.load(open(temp_savepath_bloom_taxonomy, 'rb'))
else:
    bloom_dict = {}
if os.path.isfile(temp_savepath_difficulty_level):
    difficult_dict = pickle.load(open(temp_savepath_difficulty_level, 'rb'))
else:
    difficult_dict = {}

temp_savepath_bloom_chat = temp_base_path / '{}_bloom_taxonomy_chat.pkl'.format(dataset_name)
temp_savepath_difficult_chat = temp_base_path / '{}_difficulty_level_chat.pkl'.format(dataset_name)

if os.path.isfile(temp_savepath_bloom_chat):
    bloom_chat_temp = pickle.load(open(temp_savepath_bloom_chat, 'rb'))
else:
    bloom_chat_temp = []
if os.path.isfile(temp_savepath_difficult_chat):
    difficult_chat_temp = pickle.load(open(temp_savepath_difficult_chat, 'rb'))
else:
    difficult_chat_temp = []

def get_bloom(key_list):
    random.shuffle(key_list)
    consecutive_fails = 0
    for key in key_list:
        try:
            if q_type=='MC':
                if QUESTION_STYLE.is_contextful:
                    request = QUESTION_STYLE.bloom_base.format(q_data[key]['question_text'], q_data[key]['context'], get_choice_string(q_data[key]['choices']))
                else:
                    request = QUESTION_STYLE.bloom_base.format(q_data[key]['question_text'], get_choice_string(q_data[key]['choices']))
            else:
                if QUESTION_STYLE.is_contextful:
                    request = QUESTION_STYLE.bloom_base.format(q_data[key]['question_text'], q_data[key]['context'])
                else:
                    request = QUESTION_STYLE.bloom_base.format(q_data[key]['question_text'])
            messages = [{'role': 'system', 'content':QUESTION_STYLE.system_message},
                        {'role': 'user', 'content':request}]
            print('Request: \n')
            print(request)
            completion = openai.ChatCompletion.create(model="gpt-3.5-turbo", messages=messages)
            bloom_chat_temp.append([messages, completion])
            response = completion.choices[0].message.content
            response = response.lower()
            print('Response: \n')
            print(response)

            with open(temp_savepath_bloom_chat, 'wb') as f:
                pickle.dump(bloom_chat_temp, f)

            bloom_classes = ['remember', 'understand', 'apply', 'analyze', 'evaluate', 'create']
            found_class = []
            for bloom_class in bloom_classes:
                if bloom_class in response:
                    found_class.append(bloom_class)
            if len(found_class)==1:
                bloom_dict[key] = found_class[0] 
                with open(temp_savepath_bloom_taxonomy, 'wb') as f:
                    pickle.dump(bloom_dict, f)
            consecutive_fails = 0
        except:
            consecutive_fails += 1
            if consecutive_fails == 10:
                print('got some untracked errors 10 times consecutively. Will terminate know.')
                exit()
            time.sleep(10*consecutive_fails)
            pass 

def get_difficulty(key_list):
    random.shuffle(key_list)
    consecutive_fails = 0
    for key in key_list:
        try:
            if q_type=='MC':
                if QUESTION_STYLE.is_contextful:
                    request = QUESTION_STYLE.difficulty_base.format(q_data[key]['question_text'], q_data[key]['context'], get_choice_string(q_data[key]['choices']))
                else:
                    request = QUESTION_STYLE.difficulty_base.format(q_data[key]['question_text'], get_choice_string(q_data[key]['choices']))
            else:
                if QUESTION_STYLE.is_contextful:
                    request = QUESTION_STYLE.difficulty_base.format(q_data[key]['question_text'], q_data[key]['context'])
                else:
                    request = QUESTION_STYLE.difficulty_base.format(q_data[key]['question_text'])

            messages = [{'role': 'system', 'content':QUESTION_STYLE.system_message},
                        {'role': 'user', 'content':request}]
            
            print('Request: ')
            print(request)
            completion = openai.ChatCompletion.create(model="gpt-3.5-turbo", messages=messages)
            difficult_chat_temp.append([messages, completion])
            response = completion.choices[0].message.content
            response = response.lower()
            print('Response')
            print(response)

            with open(temp_savepath_difficult_chat, 'wb') as f:
                pickle.dump(difficult_chat_temp, f)

            found_class = re.findall('\d', response)
            if len(found_class)==1: 
                found_class = found_class[0]
            elif len(found_class)==3: #Assume that three returned numbers mean that chatgpt repeated the task
                found_class.pop(found_class.index('1'))
                found_class.pop(found_class.index('5'))
                assert len(found_class)==1, 'for three integers in the response, I expected 1 and 5 to be meaningless (on first occurence), due to the model repeating the task.' 
                found_class = found_class[0]
            else:
                print('Expected either one integer of three integers for the respinse on difficulty level, but got {}'.format(found_class))
                found_class = None

            if found_class:
                difficult_dict[key] = int(found_class)
                with open(temp_savepath_difficulty_level, 'wb') as f:
                    pickle.dump(difficult_dict, f)

            consecutive_fails = 0
        except:
            consecutive_fails += 1
            if consecutive_fails == 10:
                print('got some untracked errors 10 times consecutively. Will terminate know.')
                exit()
            time.sleep(10*consecutive_fails)
            pass 

if not args.only_diff:
    remaining_keys = list(set(q_keys)-set(list(bloom_dict.keys())))
    past_iterations = 0
    while len(remaining_keys) > 0 and past_iterations<args.max_iterations:
        print('----------------------------------------------------------------------------------------------------------------')
        print('Getting bloom taxonomy for {} remaining Questions'.format(len(remaining_keys)))
        print('----------------------------------------------------------------------------------------------------------------')
        get_bloom(remaining_keys)
        remaining_keys = list(set(q_keys)-set(list(bloom_dict.keys())))
        past_iterations += 1

if not args.only_bloom:
    remaining_keys = list(set(q_keys)-set(list(difficult_dict.keys())))
    past_iterations = 0
    while len(remaining_keys) > 0 and past_iterations<args.max_iterations:
        print('----------------------------------------------------------------------------------------------------------------')
        print('Getting bloom taxonomy for {} remaining Questions'.format(len(remaining_keys)))
        print('----------------------------------------------------------------------------------------------------------------')
        get_difficulty(remaining_keys)
        remaining_keys = list(set(q_keys)-set(list(difficult_dict.keys())))
        past_iterations += 1

#Now assign the generated values to the .qar file.
for key in q_keys:
    if not args.only_diff:
        q_data[key]['bloom_classification'] = bloom_dict[key]
    if not args.only_bloom:
        if key not in difficult_dict.keys():
            print('Got no difficulty ranking for question {}. Thus I will assign the highest difficulty.\n Note that this happened only once during our testing for a question referring to non-exisiting material.'.format(key))
            q_data[key]['difficulty_level'] = '5'
        else:
            q_data[key]['difficulty_level'] = difficult_dict[key]

save_path = data_path
json.dump(q_data, open(save_path, 'w'))

    
    




        