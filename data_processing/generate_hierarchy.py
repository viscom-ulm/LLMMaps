'''
In general we do not supply the model with the questions context information here, except for US Law. 
'''

import json
import numpy as np
import copy
import pickle
import argparse
import os
import random
import pathlib
import openai
import tiktoken
import time
my_api_key = None
if my_api_key:
    openai.api_key = my_api_key
else:
    raise ValueError('Please add a ChatGPT API KEY to use this script.')

parser = argparse.ArgumentParser()
parser.add_argument('--path', help='path of the preprocessed data file in json format', type=str, default='../data/pubmedqa.qar')
parser.add_argument('--method', help='Method to create hierarchy', type=str, default='from_topic', choices=['from_topic', 'from_topic_list', 'from_topic_list_shallow', 'from_given_structure', 'topic_first'])
parser.add_argument('--given', nargs='+', help='Given list of topics to assort questions into', type=str, default=None)
args = parser.parse_args()

#setting some globally used variables, including datafiles
data_path = pathlib.Path(__file__).parents[0] / args.path
dataset_name = os.path.basename(data_path)[:-4]
method = args.method 

filtered_qa = json.load(open(data_path))
question_keys = list(filtered_qa.keys())

#load some temporary files from last time (or initialize empty)
temp_base_path = pathlib.Path(__file__).parents[0] / '..' / 'data' / 'temp' 
temp_base_path.mkdir(parents=True, exist_ok=True)
temp_savepath_chapter_dict = temp_base_path / '{}_chapter_dict.pkl'.format(dataset_name)
temp_savepath_chapter_assign_chat = temp_base_path / '{}_chapter_assign_chat.pkl'.format(dataset_name)
if os.path.isfile(temp_savepath_chapter_dict):
    chapter_dict = pickle.load(open(temp_savepath_chapter_dict, 'rb'))
else:
    chapter_dict = {}
if os.path.isfile(temp_savepath_chapter_assign_chat):
    chapter_assign_chat_temp = pickle.load(open(temp_savepath_chapter_assign_chat, 'rb'))
else:
    chapter_assign_chat_temp = []
temp_savepath_topic_dict = temp_base_path / '{}_topic_dict.pkl'.format(dataset_name)
temp_save_path_topic_chat = temp_base_path / '{}_topic_chat.pkl'.format(dataset_name)
temp_savepath_topic_assign_chat = temp_base_path / '{}_topic_assign_chat.pkl'.format(dataset_name)
if os.path.isfile(temp_savepath_topic_dict):
    topic_dict = pickle.load(open(temp_savepath_topic_dict, 'rb'))
else:
    topic_dict = {}
if os.path.isfile(temp_save_path_topic_chat):
    topic_assign_chat_temp = pickle.load(open(temp_save_path_topic_chat, 'rb'))
else:
    topic_assign_chat_temp = []

def clean_subtopic_string(outline_string):
    outline_list = []
    
    for proposed_line in outline_string.split('\n'):
        if len(proposed_line)<3: #remove "empty" lines
            continue
        if '.' in proposed_line[:3]:
            proposed_line = proposed_line.split('.')[1]
        elif ':' in proposed_line[:3]:
            proposed_line = proposed_line.split(':')[1]
        elif '-' in proposed_line[:3]:
            proposed_line = proposed_line.split('-')[1]
        if len(proposed_line.split())>5: # Assuming that no topic name should be more than 5 words we can strip possible introductory lines with this. 
            continue
        proposed_line = proposed_line.strip('[0-9]. -\\')
        outline_list.append(proposed_line)
    return '\n'.join(outline_list)

def clean_outline_string(outline_string):
    outline_list = outline_string.split('\n')
    chapter_starts = []
    for i, line in enumerate(outline_list):
        if (line.startswith(('I', 'V', 'X')) and '.' in line[:7]) or ((line.startswith('Chapter'))) or ((line.startswith('Part'))):
            chapter_starts.append(i)
    print(chapter_starts)
    final_list = []

    for i in range(len(chapter_starts)):
        start = chapter_starts[i]
        if i < len(chapter_starts)-1:
            end = chapter_starts[i+1]
        else:
            end = 1000 #dummy value much much  larger than the expected end
        if '. ' in outline_list[start]:     
            chapter = outline_list[start].split('. ', 1)[1]
        elif ': ' in outline_list[start]:     
            chapter = outline_list[start].split(': ', 1)[1]
        else:
            raise ValueError('The line containing chapter name has a structure I am unfamiliar with: ', outline_list[start])
        subchapter_start = None
        try:
            if len(outline_list[start+1]) > 3:
                subchapter_identation = len(outline_list[start+1]) - len(outline_list[start+1].lstrip(' '))
                subchapter_start = outline_list[start+1][0]
            else:
                subchapter_identation = len(outline_list[start+2]) - len(outline_list[start+2].lstrip(' '))
                subchapter_start = outline_list[start+2][0]
        except:
            subchapter_identation = 100
        if subchapter_start == ' ':
            subchapter_start = None
        #stripping generic chapters
        if 'introduction' not in chapter.lower() and 'conclusion' not in chapter.lower() and 'reference' not in chapter.lower() and 'appendi' not in chapter.lower() and 'glossar' not in chapter.lower() and 'bibliograph' not in chapter.lower() and not ('index' in chapter.lower() and len(chapter)<10):
            #remove any dashes not at the start of a line as we use them later for seperating chapters
            for l in outline_list[start:end]:
                current_identation = len(l)-len(l.lstrip(' '))
                if subchapter_start:
                    if l.startswith(subchapter_start):
                        final_list.append(l.replace('-', ' '))

                if current_identation <= subchapter_identation or len(l.lstrip(' '))<=2:
                    final_list.append(l[:4]+l[4:].replace('-', ' '))
    return '\n'.join(final_list)

def clean_topic_string(topic_string):
    topic_list = topic_string.split('\n')
    start = 0
    for i, line in enumerate(topic_list):
        if line.startswith('-'):
            start = i
            break
    topic_list = topic_list[start:]
    for i in range(len(topic_list)):
        topic_list[i] = topic_list[i].replace('- ','', 1).rstrip(' ')
    topic_string = '\n'.join(topic_list)
    return topic_string

def get_response(request):
    messages = [{'role': 'system', 'content':system_message},
                    {'role': 'user', 'content':request}]
    print('Request: ')
    print(messages)
    completion = openai.ChatCompletion.create(model="gpt-3.5-turbo", messages=messages)
    response = completion.choices[0].message.content
    print('Response: ')
    print(response)
    return messages, completion, response
    
def get_chapter(key_list, which_chapter, outlines_assorted, outlines_string, topic_list, q_per_call=1, topic_dict=None): #TODO take care of inputs
    temperature = 1
    chapter_base = which_chapter
    random.shuffle(key_list)
    fail_counter = 0
    for topic in topic_list:
        current_keys = []
        for key in key_list:
            if topic_dict is not None:
                if topic_dict[key] == topic:
                    current_keys.append(key)
            else:
                if filtered_qa[key]['main_topic'] == topic:
                        current_keys.append(key)
    
        for i in range(len(current_keys)//q_per_call+int(len(current_keys)%q_per_call!=0)):
            try:
                used_keys = current_keys[q_per_call*i:q_per_call*(i+1)]
                q_list = ['{}:{}'.format(key, filtered_qa[key]['question_text']) for key in used_keys]
                request = chapter_base.format(topic, outlines_string[topic], '\n'.join(q_list))
                messages, completion, response = get_response(request)
                chapter_assign_chat_temp.append([messages, completion])

                with open(temp_savepath_chapter_assign_chat, 'wb') as f:
                    pickle.dump(chapter_assign_chat_temp, f)

                #NOTE: This currently expects a fixed depth 
                for response_line in response.split('\n'):
                    response_line = response_line.rstrip('. ')
                    try:
                        for key in used_keys:
                            if key in response_line:
                                #print(response_line)
                                chapter_pred, subchapter_pred = response_line.split(':',1)[1].split('-',1)
                                chapter_pred = chapter_pred.strip(' ')
                                chapter_pred = chapter_pred.strip('\\')
                                subchapter_pred = subchapter_pred.strip(' ')
                                subchapter_pred = subchapter_pred.strip('\\')
                                cleaned_response = []
                                if 'chapter' in chapter_pred.lower() and 'subchapter' in subchapter_pred.lower():
                                    chap_num = chapter_pred.split(' ')[1]
                                    for chapter in outlines_assorted[topic].keys():
                                        chap_num_compare = chapter.split('.')[0]
                                        chap_num_compare = chap_num_compare.strip(' ')
                                        if chap_num == chap_num_compare:
                                            cleaned_response.append(chapter)
                                            subchap_num = subchapter_pred.split(' ')[1]
                                            for subchapter in outlines_assorted[topic][chapter]:
                                                subchap_num_compare = subchapter.split('.')[0]
                                                subchap_num_compare = subchap_num_compare.strip(' ')
                                                if subchap_num == subchap_num_compare:
                                                    cleaned_response.append(subchapter)

                                else: 
                                    for chapter in outlines_assorted[topic].keys():
                                        if chapter.lower() in chapter_pred.lower():
                                            cleaned_response.append(chapter)
                                            for subchapter in outlines_assorted[topic][chapter]:
                                                if subchapter.lower() in subchapter_pred.lower():
                                                    cleaned_response.append(subchapter)
                                if len(cleaned_response)==2:
                                    cleaned_response = ' - '.join(cleaned_response)
                                    chapter_dict[key] = cleaned_response
                                if key not in chapter_dict.keys():
                                    found_leaf_paths = []
                                    for chapter in outlines_assorted[topic].keys():
                                        for subchapter in outlines_assorted[topic][chapter]:
                                            if subchapter.lower() in response_line.lower():
                                                found_leaf_paths.append([chapter, subchapter])
                                    if len(found_leaf_paths)==1:
                                        chapter_dict[key] = ' - '.join(found_leaf_paths[0])
                                if key not in chapter_dict.keys():
                                    print('Could not assort the following response line: ', response_line)
                    except:
                        pass
                with open(temp_savepath_chapter_dict, 'wb') as f:
                    pickle.dump(chapter_dict, f)
                fail_counter = 0
            except:
                if fail_counter==10:
                    raise ValueError('Got aome errors ten times in a row and will stop now to not burn money.')
                print('Model probably overloaded, I wait a minute and try again.')
                fail_counter += 1
                time.sleep(60)
            
    return chapter_dict

def assort_outlines(outlines_string, topic_list):
    #make a new outline string getting rid of the chapter numbers and an assorted dictionary for later grouping
    outlines_assorted = {}
    for topic in topic_list:
        new_outline_string = []
        outlines_assorted[topic] = {}
        outline_string = outlines_string[topic]
        last_major = None
        for line in outline_string.split('\n'):
            if '. ' not in line and not line.lstrip(' ').startswith('-') and ': ' not in line:
                continue
            if line.startswith(('I', 'V', 'X')):
                line = line.split('. ', 1)
                outlines_assorted[topic][line[1].strip(' :.-,').strip('\\')] = []
                last_major = line[1].strip(' :.-,').strip('\\')
                new_outline_string.append(line[1].strip(' :.-,').strip('\\'))
            elif line.startswith('Chapter'):
                line = line.split(': ', 1)
                outlines_assorted[topic][line[1].strip(' :.-,').strip('\\')] = []
                last_major = line[1].strip(' ').strip('\\')
                new_outline_string.append(line[1].strip(' :.-,').strip('\\'))
            elif line.startswith('Part'):
                line = line.split(': ', 1)
                outlines_assorted[topic][line[1].strip(' :.-,').strip('\\')] = []
                last_major = line[1].strip(' :.-,').strip('\\')
                new_outline_string.append(line[1].strip(' :.-,').strip('\\'))
            else:
                if '. ' in line[:10]:
                    line = line.split('. ', 1)
                    outlines_assorted[topic][last_major].append(line[1].rstrip(' :.-,').strip('\\'))
                    new_outline_string.append('    '+line[1].rstrip(' :.-,').strip('\\'))
                else:
                    # strip lines at the end not part of the outline. This is only a hotfix, 
                    # assuming that no line we want is longer 150 characters (average around 50-70)
                    # and that each line we dont want is longer.
                    if len(line)<150:
                        line = line.lstrip('[0-9]. -').rstrip(' :.-,')
                        outlines_assorted[topic][last_major].append(line)
                        new_outline_string.append('    '+line)
        outlines_string[topic] = '\n'.join(new_outline_string)
    return outlines_string, outlines_assorted

def get_outlines(generate_outlines, topic_list):
    temp_savepath_outlines_string = temp_base_path / '{}_outlines_string.pkl'.format(dataset_name)
    temp_savepath_outlines_chat = temp_base_path / '{}_outlines_chat.pkl'.format(dataset_name)
    outline_chat_temp = []
    if not os.path.isfile(temp_savepath_outlines_string):
        outlines_string = {}
        for topic in topic_list:
            request = generate_outlines.format(topic)
            messages, completion, response = get_response(request)
            outline_chat_temp.append([messages, completion])
            outlines_string[topic] = clean_outline_string(response)
        pickle.dump(outlines_string, open(temp_savepath_outlines_string, 'wb'))
        pickle.dump(outline_chat_temp, open(temp_savepath_outlines_chat, 'wb'))
    else:
        outlines_string = pickle.load(open(temp_savepath_outlines_string, 'rb'))
    return outlines_string

def top_down2JSON(top_down_dict):
    json_dict = {"name":dataset_name, "children":[]}
    topic_leaf_sizes = {} #used to sort the assorted topics by size
    for topic in top_down_dict.keys():
        topic_leaf_sizes[topic] = 0
        topic_child_dict = {"name":topic, "children":[]}
        for chapter in top_down_dict[topic].keys():
            chapter_child_dict = {"name":chapter, "children":[]}
            for subchapter in top_down_dict[topic][chapter].keys():
                current_keys = top_down_dict[topic][chapter][subchapter]
                leaf_dict = {"name":subchapter, "question_keys":current_keys}
                chapter_child_dict['children'].append(copy.copy(leaf_dict))
                topic_leaf_sizes[topic] += 1
            topic_child_dict['children'].append(copy.copy(chapter_child_dict))
        json_dict['children'].append(copy.copy(topic_child_dict))

    # resort 1st children:
    leaf_sizes = []
    sorted_topics = []
    for k, v in topic_leaf_sizes.items():
        leaf_sizes.append(v)
        sorted_topics.append(k)
    sorting = np.argsort(leaf_sizes)[::-1]
    json_dict["children"] = [json_dict["children"][i] for i in sorting]

    return json_dict

def iterate_over_questions(task, data_dict, max_iterations=20):
    '''
    This function does the looping over all samples and collects the model answers. This is done in the provided task, 
    which has to be a function accepting only a list of sample keys (e.g. a Lambda around get_chapter).

    
    :param task:            function which accepts only the list of sample keys
    :param data_dict:       dictionary which is filled.
    :param max_iterations:  int, maximum number of oterations over the reamining questions after an iteration.
    :raises ValueError:     Only a small fraction of samples could be processed succesfully
    :raises ValueError:     After max_iterations, still too many (>0.5%) questions do not have an assortment.
    '''
    remaining_questions = list(set(question_keys) - set(list(data_dict.keys())))
    past_iterations = 0
    while len(remaining_questions) > 0 and past_iterations<max_iterations:
        questions_before = len(remaining_questions)
        print('\n Processing {}/{} remaining_questions\n'.format(len(remaining_questions), len(question_keys)))
        task(remaining_questions)
        remaining_questions = list(set(question_keys) - set(list(data_dict.keys())))
        if len(remaining_questions) > 100 and len(remaining_questions) > 0.95*questions_before:
            raise ValueError('Something is going awfully wrong, so I throw this error to not let your credit card explode.')
        past_iterations += 1
    if len(remaining_questions) <= 0.005*len(question_keys) and len(remaining_questions) > 0:
        print(f'After {max_iterations} iterations over the qa, I could not find an existing topic (subtopic and so on) for {len(remaining_questions)} samples. '
              +'Since this makes up less then 0.5% of the dataset, I will just skip them. '
              +'Note that at this point this most probably comes probably from ChatGPT being convinced the question can not be assorted to an existing node in the hierarchy.'
              +'So solving this will require manual interference.')
        question_keys = list(data_dict.keys())
    elif len(remaining_questions) > 0:
        print(f'To many questions were not assigned correctly and I do not know why. If it would have happened while building this thing I would have caught the error. '
              +'Next to problems requiring manual interference, this could also be connected to a change in the ChatGPT model '
              +'resulting in yet another representation of the answer which is not accounted before because it would have been an unknown issue at the time of writing this.')
        raise ValueError('More than 0.5% of the dataset could not be assorted in the hierarchy')
    
def num_tokens_from_string(string: str, encoding_name: str) -> int:
    """Returns the number of tokens in a text string (approximate as gpt2 tokenizer is used here)."""
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens

def get_allowed_number_input_questions(additional_strings=[], additional_lists_to_mean=[]):
    '''
    Get the amount of questions to prompt the model with at one time.
    This assumes 4096 tokens to be the maximum of input+output (as of April 2023 for gpt-3.5-turbo this was the case).
    Then get the approximate token number of any additional input and the average question, let the input maximally occupy 0.4 times the allowed tokens. 
    This results in a lower bound, which we take as result due to the two approximations made.
    Input:
        additional_strings: List of str. Accounts for the token sum of all list elements.
        additional_lists_to_mean: List of list of str. Lists of any other strings. Accounts for the token mean of each list.  
    '''
    token_numbers = []
    for key in filtered_qa.keys():
        q = filtered_qa[key]['question_text']
        s = f'{key}:{q}'
        token_numbers.append(num_tokens_from_string(s, 'gpt2'))
    mean_token_number = np.mean(token_numbers)
    max_tokens = 4096
    allowed_input_tokens = max_tokens*0.4
    for s in additional_strings:
        allowed_input_tokens -= num_tokens_from_string(s, 'gpt2')
    for ls in additional_lists_to_mean:
        token_list = []
        for s in ls:
            token_list.append(num_tokens_from_string(s, 'gpt2'))
        allowed_input_tokens -= np.mean(token_list)

    num_questions = allowed_input_tokens//mean_token_number
    return int(num_questions)
  
def get_topic(key_list, which_topic, topic_string, topic_list, q_per_call=1):
    random.shuffle(key_list)
    for i in range(len(key_list)//q_per_call+int(len(key_list)%q_per_call!=0)):
        current_keys = key_list[q_per_call*i:q_per_call*(i+1)]
        q_list = ['{}:{}'.format(key, filtered_qa[key]['question_text']) for key in current_keys]
        request = '\n'.join([which_topic.format(topic_string)]+q_list)
        messages, completion, response = get_response(request)
        topic_assign_chat_temp.append([messages, completion])

        #Check if the answer contained the topic name
        for line in response.split('\n'):
            if any([key in line for key in current_keys]):
                key, my_topic_string = line.split(':',1)
                found_topics = []
                for topic in topic_list:
                    if topic in my_topic_string:
                        found_topics.append(topic)
                
                if len(found_topics) == 1:
                    topic_dict[key] = found_topics[0]

        with open(temp_savepath_topic_dict, 'wb') as f:
            pickle.dump(topic_dict, f)
        with open(temp_savepath_topic_assign_chat, 'wb') as f:
            pickle.dump(topic_assign_chat_temp, f)

system_message = 'You are a helpful assistant fullfiliing the tasks given to him according to how he is instructed.'
generate_outlines = 'Provide me with an outline for a textbook about {}. The highest order of hierarchy schould be numbered with roman numerals. There should only be two levels of hierarchy in the resulting outline. Write your complete response in titlecase.'

which_chapter = ('Consider the following outline of a textbook about {}:\n{}\n '
                    +'I will give you a list of key - question pairs in the format <key>:<question>. We already know that each of these questions can be assorted to the topic of the textbook. '
                    +'Now tell me where in the book, with respect ot the given outline, you would most likely expect to find an answer to each of the questions.' 
                    +'Reply with one line per question in the format <key>:<chapter title>-<subchapter title>, assuming the outline consists only of chapters and subchapters, if there are more levels of hierarchy in the outline, reply respectively.' 
                    +'\n\n{}')
which_topic = ('I will give you a list of key - question pairs in the format <key>:<question>. For each of these questions, consider the following list of topics and respond with the topic under which you would organize the question.\n{}\n' 
                + 'Reply with only one line per question in the format <key>:<topic>, using the respective key from the original question and the topic you assigned from the given list. If no fitting topic is  given, pick the one which is most related. \n')

def gen_hierarchy_given_main_topic(main_topic):
    '''
    Takes a single main_topic, stratifies this into fields and generates a knowledge stratification in form of an outline (total three levels topic-chapter-subchapter).
    The main_topic is usually a single term in the main_topic field of the .qar data
    '''
    generate_topics = 'Provide me with a list of the 5-10 main topics of {}. Your response should only contain the topic list, no introductory lines or similiar. Reply with one topic per line, starting with a dash. Write your complete response in titlecase.'.format(main_topic)
       
    temp_savepath_topic_string = temp_base_path / '{}_topic_string.pkl'.format(dataset_name)
    topic_chat_temp = []
    if not os.path.isfile(temp_savepath_topic_string):
        request = generate_topics
        messages, completion, response = get_response(request)
        topic_chat_temp.append([messages, completion])
        topic_string = clean_topic_string(response)
        pickle.dump(topic_string, open(temp_savepath_topic_string, 'wb'))
        pickle.dump(topic_chat_temp, open(temp_save_path_topic_chat, 'wb'))
    else:
        topic_string = pickle.load(open(temp_savepath_topic_string, 'rb'))
    topic_list = topic_string.split('\n')

    outlines_string = get_outlines(generate_outlines, topic_list)
    outlines_string, outlines_assorted = assort_outlines(outlines_string, topic_list)
    
    q_per_call = get_allowed_number_input_questions(additional_strings=[which_topic, topic_string])
    print(f'~~~~~~~~~~~~~~~~~Going to ask {q_per_call} questions per topic call~~~~~~~~~~~~~~~~~~')

    task = lambda x: get_topic(x, which_topic, topic_string, topic_list, q_per_call=q_per_call)
    iterate_over_questions(task, topic_dict)

    q_per_call_chapter = get_allowed_number_input_questions(additional_strings=[which_chapter], additional_lists_to_mean=[[outlines_string[topic] for topic in outlines_string.keys()]])
    print(f'~~~~~~~~~~~~~~~~~Going to ask {q_per_call_chapter} questions per chapter call~~~~~~~~~~~~~~~~~~')
        
    task = lambda x: get_chapter(x, which_chapter, outlines_assorted, outlines_string, topic_list, q_per_call=q_per_call_chapter, topic_dict=topic_dict)
    iterate_over_questions(task, chapter_dict)
  
    # Now we have the location for every question, and need to transform everything into a hierarchical structuer:
    top_down_dict = {}
    for topic in outlines_assorted.keys():
        top_down_dict[topic] = {}
        for chapter in outlines_assorted[topic].keys():
            top_down_dict[topic][chapter] = {}
            for subchapter in outlines_assorted[topic][chapter]:
                top_down_dict[topic][chapter][subchapter] = []

    for key in question_keys:
        topic = topic_dict[key]
        chapters = chapter_dict[key]
        if ' - ' in chapters:
            chapter, subchapter = chapters.split(' - ')
        elif '-' in chapters:
            chapter, subchapter = chapters.split('-')
        else:
            raise ValueError('Expected a different chapter string, got {}'.format(chapters))
        if chapter not in top_down_dict[topic].keys():
            top_down_dict[topic][chapter] = {}
        if subchapter not in top_down_dict[topic][chapter].keys():
            top_down_dict[topic][chapter][subchapter] = []
        
        top_down_dict[topic][chapter][subchapter].append(key)

    return top_down2JSON(top_down_dict)

def gen_hierarchy_given_topic_list():
    '''
    This function assumes existing stratified topics in the main_topic field of the .qar file. These are then taken to get the "outlines" and finally assort all questions 
    '''
    topic_list = list(set([filtered_qa[key]['main_topic'] for key in question_keys]))
    print('Found and will use the following topic list', topic_list)

    outlines_string = get_outlines(generate_outlines, topic_list)
    outlines_string, outlines_assorted = assort_outlines(outlines_string, topic_list)

    q_per_call_chapter = get_allowed_number_input_questions(additional_strings=[which_chapter], additional_lists_to_mean=[[outlines_string[topic] for topic in outlines_string.keys()]])
    print(f'~~~~~~~~~~~~~~~~~Going to ask {q_per_call_chapter} questions per chapter call~~~~~~~~~~~~~~~~~~')
 
    task = lambda x: get_chapter(x, which_chapter, outlines_assorted, outlines_string, topic_list, q_per_call=q_per_call_chapter, topic_dict=None)
    iterate_over_questions(task, chapter_dict)
       
    # Now we have the location for every question, and need to transform everything into a hierarchical structuer:
    top_down_dict = {}
    for topic in outlines_assorted.keys():
        top_down_dict[topic] = {}
        for chapter in outlines_assorted[topic].keys():
            top_down_dict[topic][chapter] = {}
            for subchapter in outlines_assorted[topic][chapter]:
                top_down_dict[topic][chapter][subchapter] = []

    for key in question_keys:
        topic = filtered_qa[key]['main_topic']
        chapters = chapter_dict[key]
        if ' - ' in chapters:
            chapter, subchapter = chapters.split(' - ')
        elif '-' in chapters:
            chapter, subchapter = chapters.split('-')
        else:
            raise ValueError('Expected a different chapter string, got {}'.format(chapters))
        if chapter not in top_down_dict[topic].keys():
            top_down_dict[topic][chapter] = {}
        if subchapter not in top_down_dict[topic][chapter].keys():
            top_down_dict[topic][chapter][subchapter] = []
        
        top_down_dict[topic][chapter][subchapter].append(key)

    return top_down2JSON(top_down_dict)

def gen_hierarchy_given_topic_list_shallow():
    '''
    This Method is hardcoded for US Bar Exam, as it is the only place were we ever used it. Just replace "US Law: {}" with your main_topic to use the method differently. 
    In this case the context information is vital to understand the question, so it is provided.
    '''
    generate_outlines = 'Consider the following part of US law: {}. Provide me with the list of the subtopics of this area. In your response write only one subtopic per line and no other lines.  Write your complete response in titlecase.'

    topic_list = list(set([filtered_qa[key]['main_topic'] for key in question_keys]))
    print('Found and will use the following topic list', topic_list)

    outlines_string = get_outlines(generate_outlines, topic_list)
    #make a new outline string getting rid of the chapter numbers and an assorted dictionary for later grouping
    outlines_assorted = {}
    for topic in topic_list:
        outlines_assorted[topic] = []
        outline_string = outlines_string[topic]
        for line in outline_string.split('\n'):
            outlines_assorted[topic].append(line)

    #set the number of simultaneously asked questions such, that the number of input tokens is smaller 0.4 max_tokens
    
    which_chapter = ('Consider the following list of subtopics for {} as part of the US law:\n{}\n '
                    +'Now tell me under which of the given subtopics you would organize the following question considering it\'s context information. ' 
                    +'We already know that each of these questions can be assorted to {}. '
                    +'Reply with only one line containing the subtopic and nothing more.\n\n' 
                    +'Context:{}\n\n'
                    +'Question:{}')
    
    temp_savepath_chapter_dict = temp_base_path / '{}_chapter_dict.pkl'.format(dataset_name)
    temp_savepath_chapter_assign_chat = temp_base_path / '{}_chapter_assign_chat.pkl'.format(dataset_name)
    if os.path.isfile(temp_savepath_chapter_dict):
        chapter_dict = pickle.load(open(temp_savepath_chapter_dict, 'rb'))
    else:
        chapter_dict = {}
    if os.path.isfile(temp_savepath_chapter_assign_chat):
        chapter_assign_chat_temp = pickle.load(open(temp_savepath_chapter_assign_chat, 'rb'))
    else:
        chapter_assign_chat_temp = []

    def get_chapter_shallow(key_list, which_chapter, outlines_string, topic_list):
        temperature = 1
        random.shuffle(key_list)
        for topic in topic_list:
            current_keys = []
            for key in key_list:
                if filtered_qa[key]['main_topic'] == topic:
                    current_keys.append(key)
        
            for key in current_keys:
                request = which_chapter.format(topic, outlines_string[topic], topic, filtered_qa[key]['context'], filtered_qa[key]['question_text'])
                messages, completion, response = get_response(request)
                chapter_assign_chat_temp.append([messages, completion])
                response = response.rstrip('. ')

                with open(temp_savepath_chapter_assign_chat, 'wb') as f:
                    pickle.dump(chapter_assign_chat_temp, f)

                predicted_subtopics = []
                for response_line in response.split('\n'):
                    for subtopic in outlines_assorted[topic]:
                        if subtopic in response_line:
                            predicted_subtopics.append(subtopic)
                if len(predicted_subtopics)==1:
                    chapter_dict[key] = predicted_subtopics[0]
                elif len(predicted_subtopics)>len(outlines_assorted[topic]): #filter every subtopic once, in case he repeated the list
                    for subtopic in outlines_assorted[topic]:
                        try:
                            predicted_subtopics.pop(predicted_subtopics.index(subtopic))
                        except:pass
                    if len(predicted_subtopics)==1:
                        chapter_dict[key] = predicted_subtopics[0]
                elif len(predicted_subtopics)>1: #filter entries out which are part of the correct prediction (if 'ab' is in the list an'a' as well, 'ab'is correct)
                    try:
                        # pick the longest and assert that every other is part of it
                        picked = ' '
                        for subtopic in predicted_subtopics:
                            if len(subtopic) > len(picked):
                                picked = subtopic
                        for subtopic in predicted_subtopics:
                            assert subtopic in picked
                        chapter_dict[key] = picked
                    except:pass
                if key in chapter_dict.keys():
                    with open(temp_savepath_chapter_dict, 'wb') as f:
                        pickle.dump(chapter_dict, f)

    task = lambda x: get_chapter_shallow(x, which_chapter, outlines_string, topic_list)
    iterate_over_questions(task, chapter_dict)

    # Now we have the location for every question, and need to transform everything into a hierarchical structuer:
    top_down_dict = {}
    for topic in topic_list:
        top_down_dict[topic] = {}
        for subtopic in outlines_assorted[topic]:
            top_down_dict[topic][subtopic] = []

    for key in question_keys:
        topic = filtered_qa[key]['main_topic']
        subtopic = chapter_dict[key]
        top_down_dict[topic][subtopic].append(key)
    
    json_dict = {"name":dataset_name, "children":[]}
    topic_leaf_sizes = {} #used to sort the assorted topics by size

    for topic in top_down_dict.keys():
        topic_leaf_sizes[topic] = 0
        topic_child_dict = {"name":topic, "children":[]}
        for subtopic in top_down_dict[topic].keys():
            leaf_dict = {"name":subtopic, "question_keys":top_down_dict[topic][subtopic]}
            topic_child_dict['children'].append(copy.copy(leaf_dict))
            topic_leaf_sizes[topic] += 1
        json_dict['children'].append(copy.copy(topic_child_dict))

    # resort 1st children:
    leaf_sizes = []
    sorted_topics = []
    for k, v in topic_leaf_sizes.items():
        leaf_sizes.append(v)
        sorted_topics.append(k)
    sorting = np.argsort(leaf_sizes)
    sorting = np.argsort(leaf_sizes)[::-1]
    json_dict["children"] = [json_dict["children"][i] for i in sorting]

    return json_dict

def gen_hierarchy_given_structure():
    '''
    This method will actually not use CHatGPT at all, as it expects all structural information to be given in the main_topic field of the .qar data as semicolon delimited string, startingwith the global main_topic of the complete dataset.
    Additionally this function currently only supports the structure as given with usbar.qar, but can be easily adapted to larger depths.
    For that just make the loop deeper and assign [] only to the leaf level of top_down_dict
    '''
    
    top_down_dict = {}
    for key in question_keys:
        structure = filtered_qa[key]['main_topic'].split(';')
        topic1 = structure[1]
        topic2 = structure[2]
        if topic1 not in top_down_dict.keys():
            top_down_dict[topic1]={}
        if topic2 not in top_down_dict[topic1].keys():
            top_down_dict[topic1][topic2]=[]
        top_down_dict[topic1][topic2].append(key)

    json_dict = {"name":dataset_name, "children":[]}
    topic_leaf_sizes = {} #used to sort the assorted topics by size

    for topic in top_down_dict.keys():
        topic_leaf_sizes[topic] = 0
        topic_child_dict = {"name":topic, "children":[]}
        for subtopic in top_down_dict[topic].keys():
            leaf_dict = {"name":subtopic, "question_keys":top_down_dict[topic][subtopic]}
            topic_child_dict['children'].append(copy.copy(leaf_dict))
            topic_leaf_sizes[topic] += 1
        json_dict['children'].append(copy.copy(topic_child_dict))

    # resort 1st children:
    leaf_sizes = []
    sorted_topics = []
    for k, v in topic_leaf_sizes.items():
        leaf_sizes.append(v)
        sorted_topics.append(k)
    sorting = np.argsort(leaf_sizes)
    sorting = np.argsort(leaf_sizes)[::-1]
    json_dict["children"] = [json_dict["children"][i] for i in sorting]

    return json_dict

def gen_hierarchy_topic_first():
    '''
    First gets topics per question in a bottom up approach, uses these topics to get the "outline" stratification and assorts questions to those.
    '''
    from collections import Counter
    generate_topics = ('I will give you a list of key - question pairs in the format <key>:<question>. '
                      +'For each of these questions, give me it\'s general topic. The generality of the topic should be like of highschool courses. ' 
                + 'Reply with only one line per question in the format <key>:<topic>, using the respective key from the original question and the topic you assigned.\n')
    
    q_per_call = get_allowed_number_input_questions(additional_strings=[generate_topics])
    q_per_call = min(q_per_call, 25)
    print(f'~~~~~~~~~~~~~~~~~Going to ask {q_per_call} questions per topic call~~~~~~~~~~~~~~~~~~')

    def get_topic(key_list, generate_topics, q_per_call=1):
        random.shuffle(key_list)
        for i in range(len(key_list)//q_per_call+int(len(key_list)%q_per_call!=0)):
            current_keys = key_list[q_per_call*i:q_per_call*(i+1)]
            q_list = ['{}:{}'.format(key, filtered_qa[key]['question_text']) for key in current_keys]
            request = '\n'.join([generate_topics]+q_list)
            messages, completion, response = get_response(request)
            topic_assign_chat_temp.append([messages, completion])

            for line in response.split('\n'):
                if any([key in line for key in current_keys]):
                    key, my_topic_string = line.split(':',1)
                    topic_dict[key] = my_topic_string.split('/')[0].strip(' ')

            with open(temp_savepath_topic_dict, 'wb') as f:
                pickle.dump(topic_dict, f)
            with open(temp_save_path_topic_chat, 'wb') as f:
                pickle.dump(topic_assign_chat_temp, f)

    task = lambda x: get_topic(x, generate_topics, q_per_call=q_per_call_chapter)
    iterate_over_questions(task, topic_dict)

    topics_perq = [topic_dict[key] for key in topic_dict.keys()]

    while len(set(topics_perq))>10:
        #as chatgpt is sometimes prone to overspecification, lets see how many topics he assigned and pick the 10 most assigned ones.
        #we do this gradually to not get false results.
        topics = []
        sizes = []
        for topic, size in Counter(topics_perq).items():
            topics.append(topic)
            sizes.append(size)
        print(len(Counter(topics_perq).keys()))
        if len(topics) > 20:
            th = min(int(len(topics)/2), 50)
        else:
            th = 10
        sorting = np.argsort(sizes)
        topic_list = [topics[i] for i in sorting][-th:]
        topic_string = '\n'.join(topic_list)

        #now reassign the questions who are left without topic now:
        check_keys = list(topic_dict.keys())
        for key in check_keys:
            if topic_dict[key] not in topic_list:
                del topic_dict[key]

        q_per_call = get_allowed_number_input_questions(additional_strings=[which_topic, topic_string])
        print(f'~~~~~~~~~~~~~~~~~Going to ask {q_per_call} questions per topic call~~~~~~~~~~~~~~~~~~')

        task = lambda x: get_topic(x, which_topic, topic_string, topic_list, q_per_call=q_per_call)
        iterate_over_questions(task, topic_dict)

        topics_perq = [topic_dict[key] for key in topic_dict.keys()]
        
    topic_list = list(set(topics_perq))

    outlines_string = get_outlines(generate_outlines, topic_list)
    outlines_string, outlines_assorted = assort_outlines(outlines_string, topic_list)

    q_per_call_chapter = get_allowed_number_input_questions(additional_strings=[which_chapter], additional_lists_to_mean=[[outlines_string[topic] for topic in outlines_string.keys()]])
    print(f'~~~~~~~~~~~~~~~~~Going to ask {q_per_call_chapter} questions per chapter call~~~~~~~~~~~~~~~~~~')
            
    task = lambda x: get_chapter(x, which_chapter, outlines_assorted, outlines_string, topic_list, q_per_call=q_per_call_chapter, topic_dict=topic_dict)
    iterate_over_questions(task, chapter_dict)
        
    # Now we have the location for every question, and need to transform everything into a hierarchical structuer:
    top_down_dict = {}
    for topic in outlines_assorted.keys():
        top_down_dict[topic] = {}
        for chapter in outlines_assorted[topic].keys():
            top_down_dict[topic][chapter] = {}
            for subchapter in outlines_assorted[topic][chapter]:
                top_down_dict[topic][chapter][subchapter] = []

    for key in question_keys:
        topic = topic_dict[key]
        chapters = chapter_dict[key]
        if ' - ' in chapters:
            chapter, subchapter = chapters.split(' - ')
        elif '-' in chapters:
            chapter, subchapter = chapters.split('-')
        else:
            raise ValueError('Expected a different chapter string, got {}'.format(chapters))
        if chapter not in top_down_dict[topic].keys():
            top_down_dict[topic][chapter] = {}
        if subchapter not in top_down_dict[topic][chapter].keys():
            top_down_dict[topic][chapter][subchapter] = []
        
        top_down_dict[topic][chapter][subchapter].append(key)
    
    return top_down2JSON(top_down_dict)

def gen_hierarchy_into_given(given):
    '''
    Assorts all sample questions to a given topic.

    :param given: List of str, contains topics to asort the questions into.
    '''
    topic_list = given
    topic_string = '\n'.join(given)

    q_per_call = get_allowed_number_input_questions(additional_strings=[which_topic, topic_string])
    print(f'~~~~~~~~~~~~~~~~~Going to ask {q_per_call} questions per topic call~~~~~~~~~~~~~~~~~~')

    task = lambda x: get_topic(x, which_topic, topic_string, topic_list, q_per_call=q_per_call)
    iterate_over_questions(task, topic_dict)

    json_dict = {"name":dataset_name, "children":[]}
    for topic in topic_list:
        topic_child_dict = {"name":topic, "question_keys":[]}
        for key in topic_dict.keys():
            if topic_dict[key]==topic:
                topic_child_dict['question_keys'].append(key)
        json_dict['children'].append(copy.copy(topic_child_dict))

    return json_dict

if args.given:
    hierarchy_dict = gen_hierarchy_into_given(args.given)
    h_save_path = pathlib.Path('../data/{}.hir'.format(dataset_name))
    json.dump(hierarchy_dict, open(h_save_path, 'w'))
else:
    if method=='from_topic':
        main_topic = filtered_qa[question_keys[0]]['main_topic']   
        hierarchy_dict = gen_hierarchy_given_main_topic(main_topic)
        h_save_path = pathlib.Path('../data/{}.hir'.format(dataset_name))
        json.dump(hierarchy_dict, open(h_save_path, 'w'))

    elif method=='from_topic_list':
        hierarchy_dict = gen_hierarchy_given_topic_list()
        h_save_path = pathlib.Path('../data/{}.hir'.format(dataset_name))
        json.dump(hierarchy_dict, open(h_save_path, 'w'))

    elif method=='from_topic_list_shallow':
        hierarchy_dict = gen_hierarchy_given_topic_list_shallow()
        h_save_path = pathlib.Path('../data/{}.hir'.format(dataset_name))
        json.dump(hierarchy_dict, open(h_save_path, 'w'))

    elif method=='from_given_structure':
        hierarchy_dict = gen_hierarchy_given_structure()
        h_save_path = pathlib.Path('../data/{}.hir'.format(dataset_name))
        json.dump(hierarchy_dict, open(h_save_path, 'w'))

    elif method=='topic_first':
        hierarchy_dict = gen_hierarchy_topic_first()
        h_save_path = pathlib.Path('../data/{}.hir'.format(dataset_name))
        json.dump(hierarchy_dict, open(h_save_path, 'w'))