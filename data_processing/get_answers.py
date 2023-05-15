import json
import pickle
import argparse
import time
import os
import random
import pathlib
import re 
import numpy as np
from collections import Counter
my_openai_api_key = ""
my_bloom_api_token = ''


parser = argparse.ArgumentParser()
parser.add_argument('--path', help='path of the preprocessed QA data file in json format', type=str, default='../data/pubmedqa.qar')
parser.add_argument('--model', help='LLM model to use, supports chatgpt [ucapitalized], davinci is the gpt3 prime model.', type=str, default='chatgpt', choices=['chatgpt', 'davinci', 'bloom', 'gpt2', 'llama13b'])
parser.add_argument('--fewshot', help='Remove most instructional info from the prompt and do few_shot request.', action='store_true')

args = parser.parse_args()

data_path = args.path 
dataset_name = os.path.basename(data_path)[:-4]
llm_model = args.model
fewshot = args.fewshot

q_data = json.load(open(data_path))
q_keys = list(q_data.keys())
q_type = q_data[q_keys[0]]['question_type']


def get_choice_string(choices):
    return '\n'.join(f'{i+1}. '+choice for i,choice in enumerate(choices))

#setting up the general prompt
if q_type == 'yes/no/maybe':
    request_base = ('Answer the following question correctly. You can only answer this question with \"yes\", \"no\" or \"maybe\". '
                    + 'If you would encounter a case where you do not know the answer, just guess.\n\n'
                    + 'Question: {}\n\nAnswer:')

    #Check is context information is available across the dataset
    contextful = True
    for key in q_keys:
        if q_data[key]['context'] is None or q_data[key]['context']=='':
            contextful = False
    #take the initial q_keys (maybe structured) and shuffle it to remove any bias in structure. Take a fixed seed, to allow same fewshot examples in different models.
    random.Random(69).shuffle(q_keys)

    if fewshot and contextful:
        single_shot_keys = []
        for answer in ['yes', 'no', 'maybe']:
            for key in q_keys:
                if q_data[key]['answer_text_short'] == answer:
                    single_shot_keys.append(key)
                    q_keys.pop(q_keys.index(key))
                    break
        
        assert len(single_shot_keys)==3
        assert q_data[single_shot_keys[0]]['context'] is not None and q_data[single_shot_keys[0]]['context']!='', 'For contextful evaluation of the QA, we do need some context though.'
        assert q_data[single_shot_keys[1]]['context'] is not None and q_data[single_shot_keys[1]]['context']!='', 'For contextful evaluation of the QA, we do need some context though.'
        assert q_data[single_shot_keys[2]]['context'] is not None and q_data[single_shot_keys[2]]['context']!='', 'For contextful evaluation of the QA, we do need some context though.'

        if llm_model == 'bloom3b':
            request_base = ('Question: {}\nContext: {}\nChoices: [yes, no, maybe]\nAnswer: {}\n\n')
        
            request_base = request_base.format(q_data[single_shot_keys[2]]['question_text'], q_data[single_shot_keys[2]]['context'], q_data[single_shot_keys[2]]['answer_text_short'])
        elif llm_model == 'gpt2' or llm_model == 'bloom':
            request_base = ('Question: {}\nContext: {}\nChoices: [yes, no, maybe]\nAnswer: {}\n\n'
                        +'Question: {}\nContext: {}\nChoices: [yes, no, maybe]\nAnswer: {}\n\n'
                        +'Question: {}\nContext: {}\nChoices: [yes, no, maybe]\nAnswer: {}\n\n')
            
            request_base = request_base.format(q_data[single_shot_keys[0]]['question_text'], ' ', q_data[single_shot_keys[0]]['answer_text_short'],
                            q_data[single_shot_keys[1]]['question_text'], ' ', q_data[single_shot_keys[1]]['answer_text_short'],
                            q_data[single_shot_keys[2]]['question_text'], ' ', q_data[single_shot_keys[2]]['answer_text_short'])

        else:
            request_base = ('Question: {}\nContext: {}\nChoices: [yes, no, maybe]\nAnswer: {}\n\n'
                        +'Question: {}\nContext: {}\nChoices: [yes, no, maybe]\nAnswer: {}\n\n'
                        +'Question: {}\nContext: {}\nChoices: [yes, no, maybe]\nAnswer: {}\n\n')
            
            request_base = request_base.format(q_data[single_shot_keys[0]]['question_text'], q_data[single_shot_keys[0]]['context'], q_data[single_shot_keys[0]]['answer_text_short'],
                            q_data[single_shot_keys[1]]['question_text'], q_data[single_shot_keys[1]]['context'], q_data[single_shot_keys[1]]['answer_text_short'],
                            q_data[single_shot_keys[2]]['question_text'], q_data[single_shot_keys[2]]['context'], q_data[single_shot_keys[2]]['answer_text_short'])
        request_base = request_base + 'Question: {}\nContext: {}\nChoices: [yes, no, maybe]\nAnswer: '

    elif fewshot:
        single_shot_keys = []
        for answer in ['yes', 'no', 'maybe']:
            for key in q_keys:
                if q_data[key]['answer_text_short'] == answer:
                    single_shot_keys.append(key)
                    q_keys.pop(q_keys.index(key))
                    break
        
        assert len(single_shot_keys)==3

        request_base = ('Question: {}\nChoices: [yes, no, maybe]\nAnswer: {}\n\n'
                       +'Question: {}\nChoices: [yes, no, maybe]\nAnswer: {}\n\n'
                       +'Question: {}\nChoices: [yes, no, maybe]\nAnswer: {}\n\n')
        
        request_base = request_base.format(q_data[single_shot_keys[0]]['question_text'], q_data[single_shot_keys[0]]['answer_text_short'],
                        q_data[single_shot_keys[1]]['question_text'], q_data[single_shot_keys[1]]['answer_text_short'],
                        q_data[single_shot_keys[2]]['question_text'], q_data[single_shot_keys[2]]['answer_text_short'])
        request_base = request_base + 'Question: {}\nChoices: [yes, no, maybe]\nAnswer: '
    
    elif contextful:
        request_base = ('Answer the following question correctly. You can only answer this question with \"yes\", \"no\" or \"maybe\". '
                    + 'If you would encounter a case where you do not know the answer, just guess.\n\n'
                    + 'Question: {}\n\nContext: {}\n\nAnswer:')

if q_type == 'MC':
    request_base = ('Answer the following question. You can only answer this question with one of the given Choices. '
                    + 'Answer only with the number of the correct choice.\n\n'
                    + 'Question:\n {}\n\nChoices:\n {}\n\nAnswer:\n')

    #Check is context information is available across the dataset
    contextful = True
    for key in q_keys:
        if q_data[key]['context'] is None or q_data[key]['context']=='':
            contextful = False

    #take the initial q_keys (maybe structured) and shuffle it to remove any bias in structure. Take a fixed seed, to allow same fewshot examples in different models.
    random.Random(69).shuffle(q_keys)

    if fewshot and contextful:
        single_shot_keys = []
        for i in range(3):
            key = q_keys[i]
            single_shot_keys.append(key)
        for key in single_shot_keys:
            q_keys.pop(q_keys.index(key))
        
        assert q_data[single_shot_keys[0]]['context'] is not None and q_data[single_shot_keys[0]]['context']!='', 'For contextful evaluation of the QA, we do need some context though.'
        assert q_data[single_shot_keys[1]]['context'] is not None and q_data[single_shot_keys[1]]['context']!='', 'For contextful evaluation of the QA, we do need some context though.'
        assert q_data[single_shot_keys[2]]['context'] is not None and q_data[single_shot_keys[2]]['context']!='', 'For contextful evaluation of the QA, we do need some context though.'

        #Since Bloom can not take that much input:
        if llm_model == 'bloom' or llm_model == 'gpt2':
            request_base = ('Question:\n{}\nContext:\n{}\nChoices:\n{}\nAnswer:\n{}\n\n')
        
            request_base = request_base.format(q_data[single_shot_keys[2]]['question_text'], 
                                               q_data[single_shot_keys[2]]['context'], 
                                               get_choice_string(q_data[single_shot_keys[2]]['choices']), 
                                               q_data[single_shot_keys[2]]['answer_text_short'])
        else:
            request_base = ('Question:\n{}\nContext:\n{}\nChoices:\n{}\nAnswer:\n{}\n\n'
                        +'Question:\n{}\nContext:\n{}\nChoices:\n{}\nAnswer:\n{}\n\n'
                        +'Question:\n{}\nContext:\n{}\nChoices:\n{}\nAnswer:\n{}\n\n')
            
            request_base = request_base.format(q_data[single_shot_keys[0]]['question_text'], q_data[single_shot_keys[0]]['context'], get_choice_string(q_data[single_shot_keys[0]]['choices']), q_data[single_shot_keys[0]]['answer_text_short'],
                            q_data[single_shot_keys[1]]['question_text'], q_data[single_shot_keys[1]]['context'], get_choice_string(q_data[single_shot_keys[1]]['choices']), q_data[single_shot_keys[1]]['answer_text_short'],
                            q_data[single_shot_keys[2]]['question_text'], q_data[single_shot_keys[2]]['context'], get_choice_string(q_data[single_shot_keys[2]]['choices']), q_data[single_shot_keys[2]]['answer_text_short'])
        request_base = request_base + 'Question:\n {}\nContext:\n {}\nChoices:\n {}\nAnswer:\n '

    elif fewshot:
        single_shot_keys = []
        for i in range(3):
            key = q_keys[i]
            single_shot_keys.append(key)
        for key in single_shot_keys:
            q_keys.pop(q_keys.index(key))
        
        request_base = ('Question:\n{}\nChoices:\n{}\nAnswer:\n{}\n\n'
                       +'Question:\n{}\nChoices:\n{}\nAnswer:\n{}\n\n'
                       +'Question:\n{}\nChoices:\n{}\nAnswer:\n{}\n\n')
        
        request_base = request_base.format(q_data[single_shot_keys[0]]['question_text'], get_choice_string(q_data[single_shot_keys[0]]['choices']), q_data[single_shot_keys[0]]['answer_text_short'],
                        q_data[single_shot_keys[1]]['question_text'], get_choice_string(q_data[single_shot_keys[1]]['choices']), q_data[single_shot_keys[1]]['answer_text_short'],
                        q_data[single_shot_keys[2]]['question_text'], get_choice_string(q_data[single_shot_keys[2]]['choices']), q_data[single_shot_keys[2]]['answer_text_short'])
        request_base = request_base + 'Question:\n{}\nChoices:\n{}\nAnswer:\n '
    
    elif contextful:
        request_base = ('Answer the following question. The given context information might help in that. '
                    + 'Answer only with the number of the correct choice.\n\n'
                    + 'Question:\n{}\nContext:\n{}\nChoices:\n{}\nAnswer:\n')
         
temp_base_path = pathlib.Path(__file__).parents[0] / '..' / 'data' / 'temp' 
temp_base_path.mkdir(parents=True, exist_ok=True)


class AnswerMachine():        
    def __init__(self, modelname, dataname, q_type, contextful, q_data, request_base):
        self.modelname = modelname
        self.dataname = dataname
        self.q_type = q_type
        self.contextful = contextful
        self.q_data = q_data
        self.request_base = request_base
        self.q_keys = list(q_data.keys())
        self.import_()
        self.temp_savepath_answer_chat = temp_base_path / '{}_{}_get_answers_chat.pkl'.format(dataname, modelname)
        self.temp_savepath_answers = temp_base_path / '{}_{}_answers.pkl'.format(dataname, modelname)
        if os.path.isfile(self.temp_savepath_answer_chat):
            self.get_answer_chat_temp = pickle.load(open(self.temp_savepath_answer_chat, 'rb'))
        else:
            self.get_answer_chat_temp = []
        if os.path.isfile(self.temp_savepath_answers):
            self.answer_dict = pickle.load(open(self.temp_savepath_answers, 'rb'))
            #print(f'loaded {self.temp_savepath_answers} with {len(self.answer_dict.keys())} keys')
        else:
            self.answer_dict = {}

    def import_(self):
        if self.modelname in ['chatgpt', 'davinci']:
            import openai
            if my_openai_api_key:
                openai.api_key = my_openai_api_key
            else:
                raise ValueError('You selected ChatGPT as model, please also add an API_KEY to use it.')
        elif self.modelname == 'gpt2':
            from transformers import AutoTokenizer, AutoModelForCausalLM
            self.tokenizer = AutoTokenizer.from_pretrained("gpt2")
            self.model = AutoModelForCausalLM.from_pretrained("gpt2", return_dict=True, pad_token_id=self.tokenizer.eos_token_id, device_map="auto")
        elif self.modelname == 'llama13b':
            from transformers import AutoModelForCausalLM
            from transformers import LlamaTokenizer as LLaMATokenizer
            self.tokenizer = LLaMATokenizer.from_pretrained("decapoda-research/llama-13b-hf", device_map="auto")
            self.model = AutoModelForCausalLM.from_pretrained("decapoda-research/llama-13b-hf", return_dict=True, pad_token_id=self.tokenizer.eos_token_id, device_map="auto")
        elif self.modelname == 'bloom':
            import requests
            self.api_url = "https://api-inference.huggingface.co/models/bigscience/bloom"
            if my_bloom_api_token:
                self.headers = {"Authorization": f"Bearer {my_bloom_api_token}"}
            else:
                raise ValueError('You selected the full bloom model. This is run online. If you want to use it, please add an API_TOKEN.')

    def get_response(self, request):
        print('Request:\n')
        print(request)
        if self.modelname == 'chatgpt':
            #Additional system message necessary for the CHatGPT API
            system_message = 'You are a helpful assistant fullfiliing the tasks given to him according to how he is instructed.'
            messages = [{'role': 'system', 'content':system_message},
                            {'role': 'user', 'content':request}]
            completion = openai.ChatCompletion.create(model="gpt-3.5-turbo", messages=messages)
            self.get_answer_chat_temp.append([messages, completion])
            response = completion.choices[0].message.content
            response = response.lower().lstrip('\n \t')
        elif self.modelname == 'davinci':
            completion = openai.Completion.create(model="text-davinci-003", prompt=request)
            self.get_answer_chat_temp.append([request, completion])
            response = completion.choices[0].text
            response = response.lower().lstrip('\n \t')
        elif self.modelname == 'gpt2':
            inputs = self.tokenizer(request, return_tensors="pt")
            response = self.tokenizer.decode(self.model.generate(inputs["input_ids"], max_new_tokens=50)[0])
            response = response[len(request):].lower().lstrip('\n \t')
            response = response.split('\n')[0]
        elif self.modelname == 'llama13b':
            inputs = self.tokenizer(request, return_tensors="pt")
            response = self.tokenizer.decode(self.model.generate(inputs["input_ids"], max_new_tokens=50)[0])
            response = response[len(request):].lower().lstrip('\n \t')
            response = response.split('\n')[0] 
        elif self.modelname == 'bloom':
            response = requests.post(self.api_url, headers=self.headers, json={"inputs": request})
            response = response.json()
            try:
                if 'error' in response.keys():
                    if response['error'].startswith('Rate limit reached'):
                        print('Hourly limit reached, I am sleeping for 65 minutes.')
                        time.sleep(3900)# sleep for little more than one hour if rate limit reached
            except:
                pass
            response = response[0]['generated_text']
            response = response[len(request):].lower().lstrip('\n \t')

        print('Response:\n')
        print(response)
        return response

    def get_answers(self, key_list):
        random.shuffle(key_list)
        consecutive_fails = 0
        fail_th = 10
        for key in key_list:
            try:
                start = time.time()
                if self.q_type=='MC':
                    if self.contextful:
                        request = self.request_base.format(self.q_data[key]['question_text'], self.q_data[key]['context'], get_choice_string(self.q_data[key]['choices']))
                    else:
                        request = self.request_base.format(self.q_data[key]['question_text'], get_choice_string(self.q_data[key]['choices']))
                else:
                    if self.contextful:
                        request = self.request_base.format(self.q_data[key]['question_text'], self.q_data[key]['context'])
                    else:
                        request = self.request_base.format(self.q_data[key]['question_text'])
                
                response = self.get_response(request)

                answer_time = time.time()-start

                with open(self.temp_savepath_answer_chat, 'wb') as f:
                    pickle.dump(self.get_answer_chat_temp, f)

                if self.q_type=='MC':
                    assert len(self.q_data[key]['choices'])<10, 'We did not expects questions with more than 9 choice options, the rest of the coude would break for these cases.'
                    numbers_in_response = re.findall('\d', response)
                    correct_choice = self.q_data[key]['answer_text_short']
                    
                    #check if the first number in the answer is an existing key. Desirable behaviour as it would happen when model only responds with one number.
                    if len(numbers_in_response)>0:
                        if int(numbers_in_response[0]) in list(range(1, len(self.q_data[key]['choices'])+1)):
                            pred_choice = int(numbers_in_response[0])
                        #check if otherwise only one number was predicted, In case the model just rephrased it's answer
                        else:
                            matches = 0
                            last_found = None
                            for i in numbers_in_response:
                                if int(i) in list(range(1, len(self.q_data[key]['choices'])+1)):
                                    matches+=1
                                    last_found = i
                            if matches==1:
                                pred_choice = last_found
                            else:
                                pred_choice = None
                    else:
                        found_labels = [i+1 for i, choice in enumerate(self.q_data[key]['choices']) if choice.lower() in response.lower()]
                        if len(found_labels)==1:
                            pred_choice = found_labels[0]

                    if pred_choice:                            
                        self.answer_dict[key] = {}
                        self.answer_dict[key]['answer_text_short'] = int(pred_choice)
                        self.answer_dict[key]['answer_time'] = answer_time
                        self.answer_dict[key]['answer_text_long'] = self.q_data[key]['choices'][int(pred_choice)-1]
                        self.answer_dict[key]['score_default'] = 'score_exact'
                        self.answer_dict[key]['score_f1'] = None
                        self.answer_dict[key]['score_exact'] = int(int(pred_choice)==correct_choice)
                        with open(self.temp_savepath_answers, 'wb') as f:
                            pickle.dump(self.answer_dict, f)

                elif q_type=='yes/no/maybe':
                    response_words = re.sub("[^\w]", " ",  response).split()
                    if 'yes' in response_words and not 'no' in response_words and not 'maybe' in response_words:
                        self.answer_dict[key] = {}
                        self.answer_dict[key]['answer_text_short'] = 'yes'
                    elif 'no' in response_words and not 'yes' in response_words and not 'maybe' in response_words:
                        self.answer_dict[key] = {}
                        self.answer_dict[key]['answer_text_short'] = 'no'
                    elif 'maybe' in response_words and not 'yes' in response_words and not 'no' in response_words:
                        self.answer_dict[key] = {}
                        self.answer_dict[key]['answer_text_short'] = 'maybe'
                    elif 'yes' in response_words and 'no' in response_words and 'maybe' in response_words:
                        #The model might repeat yes no maybe first (experienced with bloom).
                        c = Counter(response_words)
                        n_yes = c['yes']
                        n_no = c['no']
                        n_maybe = c['maybe']
                        all_same = min(n_yes, n_no, n_maybe)
                        n_yes -= all_same
                        n_no -= all_same
                        n_maybe -= all_same
                        if n_yes>0 and n_no==0 and n_maybe==0:
                            self.answer_dict[key] = {}
                            self.answer_dict[key]['answer_text_short'] = 'yes'
                        elif n_no>0 and n_yes==0 and n_maybe==0:
                            self.answer_dict[key] = {}
                            self.answer_dict[key]['answer_text_short'] = 'no'
                        elif n_maybe>0 and n_yes==0 and n_no==0:
                            self.answer_dict[key] = {}
                            self.answer_dict[key]['answer_text_short'] = 'maybe'
                        #check for a specific response pattern of bloom
                        if key not in self.answer_dict.keys():
                            abc_to_answer = {'a': 'yes', 'b':'no', 'c':'maybe'}
                            if 'a: yes' in response and 'b: no' in response and 'c: maybe' in response:
                                found_words = 0
                                for word in response_words:
                                    print(word, found_words)
                                    if word == 'the' and found_words==0:
                                        found_words += 1
                                    elif word == 'answer' and found_words==1:
                                        found_words += 1
                                    elif word == 'is' and found_words==2:
                                        found_words += 1
                                    elif found_words==3 and any([q==word for q in abc_to_answer.keys()]):
                                        found_answer = abc_to_answer[word]
                                        print(found_answer)
                                        self.answer_dict[key] = {}
                                        self.answer_dict[key]['answer_text_short'] = found_answer

                    if key in self.answer_dict.keys():
                        self.answer_dict[key]['answer_time'] = answer_time
                        self.answer_dict[key]['answer_text_long'] = None
                        self.answer_dict[key]['score_default'] = 'score_exact'
                        self.answer_dict[key]['score_f1'] = None
                        self.answer_dict[key]['score_exact'] = self.answer_dict[key]['answer_text_short']==self.q_data[key]['answer_text_short']
                        with open(self.temp_savepath_answers, 'wb') as f:
                            pickle.dump(self.answer_dict, f)
                else:
                    raise ValueError('the given question_type is not yet implemented, you have ', self.q_type)
                consecutive_fails = 0
            except:
                consecutive_fails += 1
                if consecutive_fails == fail_th:
                    print('got some untracked errors 10 times consequtively. Will terminate know.')
                    exit()
                time.sleep(10*consecutive_fails)
                pass 
 
    def run(self):
        remaining_keys = list(set(self.q_keys)-set(list(self.answer_dict.keys())))
        past_iterations = 0
        while len(remaining_keys) > 0 and past_iterations<10:
            print('----------------------------------------------------------------------------------------------------------------')
            print('Getting answers for {}/{} remaining Questions'.format(len(remaining_keys), len(self.q_keys)))
            print('----------------------------------------------------------------------------------------------------------------')
            self.get_answers(remaining_keys)
            remaining_keys = list(set(self.q_keys)-set(list(self.answer_dict.keys())))
            past_iterations += 1

        if len(remaining_keys) > 0:
            print('{} could not give an answer to {} of {} questions. For these we assign no answers and a negative score'.format(self.modelname, len(remaining_keys), len(self.q_keys)))
            self.fill_negative_answers(remaining_keys)

        save_path = pathlib.Path('../data/{}_{}.qaa'.format(self.dataname, self.modelname))
        json.dump(self.answer_dict, open(save_path, 'w'))
    
    def fill_negative_answers(self, key_list):
        '''
        Fills the entries for all unanswered questions. For these the time is set to the mean of all other times to make the later visualization easier. 
        The score is set to 0 (100% false) and all other fields are empty.
        '''
        mean_time = np.mean([self.answer_dict[key]['answer_time'] for key in self.answer_dict.keys()])
        for key in key_list:
            assert key not in self.answer_dict.keys()
            self.answer_dict[key] = {}
            self.answer_dict[key]['answer_text_short'] = None
            self.answer_dict[key]['answer_time'] = mean_time
            self.answer_dict[key]['answer_text_long'] = None
            self.answer_dict[key]['score_default'] = 'score_exact'
            self.answer_dict[key]['score_f1'] = None
            self.answer_dict[key]['score_exact'] = 0


my_machine = AnswerMachine(llm_model, dataset_name, q_type, contextful, q_data, request_base)
my_machine.run()
        