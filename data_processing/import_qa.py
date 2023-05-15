'''
This code imports the given datasets into the .qar structure for the raw datasets.
In these every sample question is a dictionary with the following fields:
    'question_text' :           Text of the question, without answer options.
    'answer_text_short' :       Short Version of the answer. For MC this is just a number from 1 to the number of choices
    'answer_text_long' :        Long answer version, currently not utilized in the rest of the code. Might be the same as short answer.
    'context' :                 Context string, optional. Should not render the task into text understanding
    'question_type' :           Type of the question, currently we incorporate 'yes/no'maybe' and 'MC' 
    'bloom_classification' :    will be filled later, LLM assigned class of Bloom's taxonomy
    'difficulty_level' :        will be filled later, LLM assigned difficulty level
    'main_topic' :              Main topic of the qa, different behavior in generate_hierarchy for None/not-None, lists or ';' seperated strings
    'choices' :                 Choice options for MC datasets 

If you want to import additional datasets, you just need to map the respective Information from the respective raw dataset to our used format. 
In many cases you can build on existing cases, e.g. when incorporating more HELM datasets you can start with the existing ones, 
medmcqa and sqiq are examples of huggingface datasets and pubmedqa and usbar are examples of other datasets which are imported from local files.  
'''
import argparse
import pathlib
import json
import os
import random
from nltk.metrics.scores import f_measure
import string
import re

parser = argparse.ArgumentParser()
parser.add_argument('--path', help='Path to the original qa dataset', type=str,
                    default='/home/patrik/Documents/llm_vis/pubmedqa/data/ori_pqal.json')
parser.add_argument('--target', help='target name of the preprocessed dataset. Will be appended by .qar and saved in data. If none provided will use the filename of path.', type=str, required=True,
                    choices=['pubmedqa', 'medmcqa', 'sciq', 'usbar', 'mmlu', 'openbookqa', 'naturalqa', 'truthfulqa', 'wikifact'])
parser.add_argument('--from_helm', help='set if the imported dataset comes from the helm result database. This will also create the .qaa files. As there are often several data modalities for a given dataset, refer to the readme or the code for additional instructions.', action='store_true')
args = parser.parse_args()

data_path = args.path
target = args.target
from_helm = args.from_helm

# The following two functions are taken from the HELM github repository 'https://github.com/stanford-crfm/helm/blob/main/src/helm/benchmark/metrics/basic_metrics.py'


def normalize_text(text: str) -> str:
    """Lower text and remove punctuation, articles and extra whitespace.
    Copied from the [QuAC](http://quac.ai/) evaluation script found at
    https://s3.amazonaws.com/my89public/quac/scorer.py"""

    def remove_articles(text: str) -> str:
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text: str) -> str:
        return " ".join(text.split())

    def remove_punc(text: str) -> str:
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text: str) -> str:
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(text))))


def f1_score(gold: str, pred: str) -> float:
    ret = f_measure(set(normalize_text(gold).split()),
                    set(normalize_text(pred).split()))
    if ret is None:  # answer is the empty string after normalizing
        return 0.0

    return ret


letter_list = ['', 'A', 'B', 'C', 'D', 'E', 'F',
               'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O']
if from_helm:
    import glob
    import copy
    # given that helm contains over 6000 datafiles with various data modalities, we can not cover all cases here, however with this code your target case should be easy to adapt.
    # We only load the datasets concerned with knowledge assessment.
    # in this case we expect the path to be the directory containing the helm files.
    # Since HELM provides result files for various (>6000) LLM evaluations, this code will also produce the respective .qaa files.
    # Furthermore for datasets where it is applicable, a shallow hierarchy will be returned. (e.g. mmlu or wikifact where a subject separation is available)
    # We imprt the following modalities:
    #   mmlu: multiple_choice_joint data_augmentation=canonical only unperturbed
    #   openbookqa: multiple_choice_joint groups=ablation_multiple_choice
    # And the following models (if you want to use different models, just change picked models and everything should work, given the evaluations are provided by helm):
    #   anthropic_stanford-online-all-v4-s3
    #   together_bloom
    #   together_gpt-j-6b
    #   together_gpt-neox-20b
    #   together_opt-66b
    #   together_opt-175b
    picked_models = ['anthropic_stanford-online-all-v4-s3', 'together_bloom',
                     'together_gpt-j-6b', 'together_gpt-neox-20b', 'together_opt-66b', 'together_opt-175b']
    if target == 'mmlu':
        unperturbed = True
        # get the files we are interested in
        mmlu_files = []
        subject_lens = {}
        for file_ in glob.glob(os.path.join(data_path, '*')):
            filename = os.path.basename(file_)
            if 'mmlu' in filename and 'data_augmentation' in filename and 'multiple_choice_joint' in filename and any([model in filename for model in picked_models]):
                dataname = filename.split('_model')[0]
                subject = dataname[13:].split('_method')[0]
                model = filename.split('_model=')[1]
                model = model.split('_data_augmentation')[0]
                mmlu_files.append((file_, model, subject))
                data = json.load(open(file_))
                if subject not in subject_lens.keys():
                    subject_lens[subject] = []
                subject_lens[subject].append(len(data['request_states']))

        # assert that the data is the same for all instances
        for subject in subject_lens.keys():
            assert len(set(subject_lens[subject])) == 1

        # get a list of subjects and models
        all_models = []
        all_subjects = []
        data_dict = {}
        for sample in mmlu_files:
            file_, model, subject = sample
            all_models.append(model)
            all_subjects.append(subject)
            if model not in data_dict.keys():
                data_dict[model] = {}
            assert subject not in data_dict[model].keys()
            data_dict[model][subject] = json.load(open(file_))
        all_models = list(set(all_models))
        all_subjects = list(set(all_subjects))
        print('models total: ', len(all_models))
        for model in sorted(all_models):
            print(model)

        incomplete_models = []
        for model in all_models:
            for subject in all_subjects:
                if subject not in data_dict[model].keys():
                    incomplete_models.append(model)
        incomplete_models = list(set(incomplete_models))
        print('incomplete models: ', incomplete_models)

        # transform the data into our format
        raw_data = data_dict[picked_models[0]]
        filtered_qa = {}
        for subject in all_subjects:
            for sample in raw_data[subject]['request_states']:
                sample_id = sample['instance']['id']
                if 'perturbation' in sample['instance'].keys():
                    continue
                key = '_'.join([target, subject, sample_id])
                key = key.replace('=', '_')
                options = []
                correct_option = None
                for idx, option in enumerate(sample['instance']['references']):
                    options.append(option['output']['text'])
                    if len(option['tags']) > 0:
                        if option['tags'][0] == 'correct':
                            correct_option = idx+1
                    else:
                        assert len(option['tags']) <= 1
                        assert option['tags'] == []
                if key in filtered_qa.keys():
                    assert filtered_qa[key]['question_text'] == sample['instance']['input']['text']
                    assert filtered_qa[key]['answer_text_short'] == correct_option
                    assert filtered_qa[key]['answer_text_long'] == options[correct_option-1]
                    assert filtered_qa[key]['main_topic'] == subject
                else:
                    filtered_qa[key] = {}
                    filtered_qa[key]['question_text'] = sample['instance']['input']['text']
                    filtered_qa[key]['answer_text_short'] = correct_option
                    filtered_qa[key]['answer_text_long'] = options[correct_option-1]
                    filtered_qa[key]['context'] = None
                    filtered_qa[key]['question_type'] = 'MC'
                    filtered_qa[key]['bloom_classification'] = None
                    filtered_qa[key]['difficulty_level'] = None
                    filtered_qa[key]['main_topic'] = subject
                    filtered_qa[key]['choices'] = options
        savepath = pathlib.Path(
            __file__).parents[0] / '..' / 'data' / (target+'.qar')
        savepath.parents[0].mkdir(parents=True, exist_ok=True)
        json.dump(filtered_qa, open(savepath, 'w'))

        # now get the .qaa
        for model in all_models:
            answer_dict = {}
            for subject in all_subjects:
                for sample in data_dict[model][subject]['request_states']:
                    if 'perturbation' in sample['instance'].keys():
                        continue
                    sample_id = sample['instance']['id']
                    key = '_'.join([target, subject, sample_id])
                    key = key.replace('=', '_')
                    if key not in answer_dict.keys():
                        if sample['result']['request_time'] != 0:
                            answer_time = sample['result']['request_time']
                        else:
                            answer_time = sample['result']['batch_request_time'] / \
                                sample['result']['batch_size']
                        answer_dict[key] = {}
                        answer_dict[key]['answer_text_short'] = letter_list.index(
                            sample['result']['completions'][0]['text'].strip())
                        answer_dict[key]['answer_text_long'] = None
                        answer_dict[key]['answer_time'] = answer_time
                        answer_dict[key]['score_f1'] = None
                        answer_dict[key]['score_exact'] = answer_dict[key]['answer_text_short'] == filtered_qa[key]['answer_text_short']
                        answer_dict[key]['score_default'] = answer_dict[key]['score_exact']
                    else:
                        if answer_dict[key]['score_exact']:
                            continue
                        else:
                            if letter_list.index(sample['result']['completions'][0]['text'].strip()) == filtered_qa[key]['answer_text_short']:
                                if sample['result']['request_time'] != 0:
                                    answer_time = sample['result']['request_time']
                                else:
                                    answer_time = sample['result']['batch_request_time'] / \
                                        sample['result']['batch_size']
                                answer_dict[key]['answer_text_short'] = letter_list.index(
                                    sample['result']['completions'][0]['text'].strip())
                                answer_dict[key]['answer_text_long'] = None
                                answer_dict[key]['answer_time'] = answer_time
                                answer_dict[key]['score_f1'] = None
                                answer_dict[key]['score_exact'] = answer_dict[key]['answer_text_short'] == filtered_qa[key]['answer_text_short']
                                answer_dict[key]['score_default'] = answer_dict[key]['score_exact']
            savepath = pathlib.Path(
                __file__).parents[0] / '..' / 'data' / ('_'.join([target, model])+'.qaa')
            json.dump(answer_dict, open(savepath, 'w'))

        # and now get the shallow .hir file which only branches the dataset into the given subjects:
        hir = {'name': target, 'children': []}
        for subject in all_subjects:
            sub_dict = {'name': subject, 'question_keys': [
                key for key in filtered_qa.keys() if filtered_qa[key]['main_topic'] == subject]}
            hir['children'].append(copy.copy(sub_dict))
        savepath = pathlib.Path(
            __file__).parents[0] / '..' / 'data' / (target+'.hir')
        json.dump(hir, open(savepath, 'w'))

    elif target == 'openbookqa':
        # get the files we are interested in
        chosen_files = []
        subject_lens = {}
        for file_ in glob.glob(os.path.join(data_path, '*')):
            filename = os.path.basename(file_)
            if 'commonsense_dataset=openbookqa' in filename and 'groups=ablation_multi' in filename and 'multiple_choice_joint' in filename and any([model in filename for model in picked_models]):
                subject = filename.split('_dataset')[0]
                model = filename.split('_model=')[1]
                model = model.split('_groups=')[0]
                chosen_files.append((file_, model, subject))
                data = json.load(open(file_))
                if subject not in subject_lens.keys():
                    subject_lens[subject] = []
                subject_lens[subject].append(len(data['request_states']))

        # assert that the data is the same for all instances
        for subject in subject_lens.keys():
            assert len(set(subject_lens[subject])) == 1

        # get a list of subjects and models
        all_models = []
        all_subjects = []
        data_dict = {}
        for sample in chosen_files:
            file_, model, subject = sample
            all_models.append(model)
            all_subjects.append(subject)
            if model not in data_dict.keys():
                data_dict[model] = {}
            assert subject not in data_dict[model].keys()
            data_dict[model][subject] = json.load(open(file_))
        all_models = list(set(all_models))
        all_subjects = list(set(all_subjects))
        print('models total: ', len(all_models))
        for model in sorted(all_models):
            print(model)

        incomplete_models = []
        for model in all_models:
            for subject in all_subjects:
                if subject not in data_dict[model].keys():
                    incomplete_models.append(model)
        incomplete_models = list(set(incomplete_models))
        print('incomplete models: ', incomplete_models)

        # transform the data into our format
        raw_data = data_dict[picked_models[0]]
        filtered_qa = {}
        for subject in all_subjects:
            for sample in raw_data[subject]['request_states']:
                sample_id = sample['instance']['id']
                if 'perturbation' in sample['instance'].keys():
                    continue
                key = '_'.join([target, subject, sample_id])
                key = key.replace('=', '_')
                options = []
                correct_option = None
                for idx, option in enumerate(sample['instance']['references']):
                    options.append(option['output']['text'])
                    if len(option['tags']) > 0:
                        if option['tags'][0] == 'correct':
                            correct_option = idx+1
                    else:
                        assert len(option['tags']) <= 1
                        assert option['tags'] == []
                if key in filtered_qa.keys():
                    assert filtered_qa[key]['question_text'] == sample['instance']['input']['text']
                    assert filtered_qa[key]['answer_text_short'] == correct_option
                    assert filtered_qa[key]['answer_text_long'] == options[correct_option-1]
                    assert filtered_qa[key]['main_topic'] == subject
                else:
                    filtered_qa[key] = {}
                    filtered_qa[key]['question_text'] = sample['instance']['input']['text']
                    filtered_qa[key]['answer_text_short'] = correct_option
                    filtered_qa[key]['answer_text_long'] = options[correct_option-1]
                    filtered_qa[key]['context'] = None
                    filtered_qa[key]['question_type'] = 'MC'
                    filtered_qa[key]['bloom_classification'] = None
                    filtered_qa[key]['difficulty_level'] = None
                    filtered_qa[key]['main_topic'] = subject
                    filtered_qa[key]['choices'] = options
        savepath = pathlib.Path(
            __file__).parents[0] / '..' / 'data' / (target+'.qar')
        savepath.parents[0].mkdir(parents=True, exist_ok=True)
        json.dump(filtered_qa, open(savepath, 'w'))

        # now get the .qaa
        for model in all_models:
            answer_dict = {}
            for subject in all_subjects:
                for sample in data_dict[model][subject]['request_states']:
                    if 'perturbation' in sample['instance'].keys():
                        continue
                    sample_id = sample['instance']['id']
                    key = '_'.join([target, subject, sample_id])
                    key = key.replace('=', '_')
                    if key not in answer_dict.keys():
                        if sample['result']['request_time'] != 0:
                            answer_time = sample['result']['request_time']
                        else:
                            answer_time = sample['result']['batch_request_time'] / \
                                sample['result']['batch_size']
                        answer_dict[key] = {}
                        answer_dict[key]['answer_text_short'] = letter_list.index(
                            sample['result']['completions'][0]['text'].strip())
                        answer_dict[key]['answer_text_long'] = None
                        answer_dict[key]['answer_time'] = answer_time
                        answer_dict[key]['score_f1'] = None
                        answer_dict[key]['score_exact'] = answer_dict[key]['answer_text_short'] == filtered_qa[key]['answer_text_short']
                        answer_dict[key]['score_default'] = answer_dict[key]['score_exact']
                    else:
                        if answer_dict[key]['score_exact']:
                            continue
                        else:
                            if letter_list.index(sample['result']['completions'][0]['text'].strip()) == filtered_qa[key]['answer_text_short']:
                                if sample['result']['request_time'] != 0:
                                    answer_time = sample['result']['request_time']
                                else:
                                    answer_time = sample['result']['batch_request_time'] / \
                                        sample['result']['batch_size']
                                answer_dict[key]['answer_text_short'] = letter_list.index(
                                    sample['result']['completions'][0]['text'].strip())
                                answer_dict[key]['answer_text_long'] = None
                                answer_dict[key]['answer_time'] = answer_time
                                answer_dict[key]['score_f1'] = None
                                answer_dict[key]['score_exact'] = answer_dict[key]['answer_text_short'] == filtered_qa[key]['answer_text_short']
                                answer_dict[key]['score_default'] = answer_dict[key]['score_exact']
            savepath = pathlib.Path(
                __file__).parents[0] / '..' / 'data' / ('_'.join([target, model])+'.qaa')
            json.dump(answer_dict, open(savepath, 'w'))

        # and now get the shallow .hir file which only branches the dataset into the given subjects:
        hir = {'name': target, 'children': []}
        for subject in all_subjects:
            sub_dict = {'name': subject, 'question_keys': [
                key for key in filtered_qa.keys() if filtered_qa[key]['main_topic'] == subject]}
            hir['children'].append(copy.copy(sub_dict))
        savepath = pathlib.Path(
            __file__).parents[0] / '..' / 'data' / (target+'.hir')
        json.dump(hir, open(savepath, 'w'))

    elif target == 'naturalqa':
        # get the files we are interested in
        chosen_files = []
        subject_lens = {}
        for file_ in glob.glob(os.path.join(data_path, '*')):
            filename = os.path.basename(file_)
            if 'natural_qa_mode=closedbook' in filename and 'data_augmentation=canonical' in filename and any([model in filename for model in picked_models]):
                subject = filename.split('_mode=')[1].split('_model')[0]
                model = filename.split('_model=')[1]
                model = model.split('_data_augmentation=')[0]
                print(file_, model, subject)
                chosen_files.append((file_, model, subject))
                data = json.load(open(file_))
                if subject not in subject_lens.keys():
                    subject_lens[subject] = []
                subject_lens[subject].append(len(data['request_states']))

        # assert that the data is the same for all instances
        for subject in subject_lens.keys():
            assert len(set(subject_lens[subject])) == 1

        # get a list of subjects and models
        all_models = []
        all_subjects = []
        data_dict = {}
        for sample in chosen_files:
            file_, model, subject = sample
            all_models.append(model)
            all_subjects.append(subject)
            if model not in data_dict.keys():
                data_dict[model] = {}
            assert subject not in data_dict[model].keys()
            data_dict[model][subject] = json.load(open(file_))
        all_models = list(set(all_models))
        all_subjects = list(set(all_subjects))
        print('models total: ', len(all_models))
        for model in sorted(all_models):
            print(model)

        incomplete_models = []
        for model in all_models:
            for subject in all_subjects:
                if subject not in data_dict[model].keys():
                    incomplete_models.append(model)
        incomplete_models = list(set(incomplete_models))
        print('incomplete models: ', incomplete_models)

        # transform the data into our format
        raw_data = data_dict[picked_models[0]]
        filtered_qa = {}
        for subject in all_subjects:
            for sample in raw_data[subject]['request_states']:
                sample_id = sample['instance']['id']
                if 'perturbation' in sample['instance'].keys():
                    continue
                key = '_'.join([target, subject, sample_id])
                key = key.replace('=', '_')
                options = []
                correct_option = None
                for idx, option in enumerate(sample['instance']['references']):
                    options.append(option['output']['text'])
                if key in filtered_qa.keys():
                    assert filtered_qa[key]['question_text'] == sample['instance']['input']['text']
                    assert filtered_qa[key]['main_topic'] == subject
                else:
                    filtered_qa[key] = {}
                    filtered_qa[key]['question_text'] = sample['instance']['input']['text']
                    filtered_qa[key]['answer_text_short'] = None
                    filtered_qa[key]['answer_text_long'] = options
                    filtered_qa[key]['context'] = None
                    filtered_qa[key]['question_type'] = 'open'
                    filtered_qa[key]['bloom_classification'] = None
                    filtered_qa[key]['difficulty_level'] = None
                    filtered_qa[key]['main_topic'] = subject
                    filtered_qa[key]['choices'] = options
        savepath = pathlib.Path(
            __file__).parents[0] / '..' / 'data' / (target+'.qar')
        savepath.parents[0].mkdir(parents=True, exist_ok=True)
        json.dump(filtered_qa, open(savepath, 'w'))

        # now get the .qaa
        for model in all_models:
            answer_dict = {}
            for subject in all_subjects:
                for sample in data_dict[model][subject]['request_states']:
                    if 'perturbation' in sample['instance'].keys():
                        continue
                    sample_id = sample['instance']['id']
                    key = '_'.join([target, subject, sample_id])
                    key = key.replace('=', '_')
                    if key not in answer_dict.keys():
                        if sample['result']['request_time'] != 0:
                            answer_time = sample['result']['request_time']
                        else:
                            answer_time = sample['result']['batch_request_time'] / \
                                sample['result']['batch_size']
                        answer_dict[key] = {}
                        answer_dict[key]['answer_text_short'] = None
                        answer_dict[key]['answer_text_long'] = sample['result']['completions'][0]['text'].strip(
                        )
                        answer_dict[key]['answer_time'] = answer_time
                        answer_dict[key]['score_f1'] = max([f1_score(
                            gold, answer_dict[key]['answer_text_long']) for gold in filtered_qa[key]['answer_text_long']])
                        answer_dict[key]['score_exact'] = None
                        answer_dict[key]['score_default'] = answer_dict[key]['score_f1']
                    else:
                        if answer_dict[key]['score_exact']:
                            continue
                        else:
                            new_score = max([f1_score(gold, sample['result']['completions'][0]['text'].strip(
                            )) for gold in filtered_qa[key]['answer_text_long']])
                            if new_score > answer_dict[key]['score_f1']:
                                if sample['result']['request_time'] != 0:
                                    answer_time = sample['result']['request_time']
                                else:
                                    answer_time = sample['result']['batch_request_time'] / \
                                        sample['result']['batch_size']
                                answer_dict[key]['answer_text_short'] = None
                                answer_dict[key]['answer_text_long'] = sample['result']['completions'][0]['text'].strip(
                                )
                                answer_dict[key]['answer_time'] = answer_time
                                answer_dict[key]['score_f1'] = new_score
                                answer_dict[key]['score_exact'] = None
                                answer_dict[key]['score_default'] = new_score
            savepath = pathlib.Path(
                __file__).parents[0] / '..' / 'data' / ('_'.join([target, model])+'.qaa')
            json.dump(answer_dict, open(savepath, 'w'))

        # and now get the shallow .hir file which only branches the dataset into the given subjects:
        hir = {'name': target, 'children': []}
        for subject in all_subjects:
            sub_dict = {'name': subject, 'question_keys': [
                key for key in filtered_qa.keys() if filtered_qa[key]['main_topic'] == subject]}
            hir['children'].append(copy.copy(sub_dict))
        savepath = pathlib.Path(
            __file__).parents[0] / '..' / 'data' / (target+'.hir')
        json.dump(hir, open(savepath, 'w'))

    elif target == 'truthfulqa':
        # get the files we are interested in
        chosen_files = []
        subject_lens = {}
        for file_ in glob.glob(os.path.join(data_path, '*')):
            filename = os.path.basename(file_)
            if 'task=mc_single' in filename and 'data_augmentation=canonical' in filename and 'multiple_choice_joint' in filename and any([model in filename for model in picked_models]):
                subject = filename.split('_task=')[1].split('_method=')[0]
                model = filename.split('_model=')[1]
                model = model.split('_data_augmentation=')[0]
                chosen_files.append((file_, model, subject))
                data = json.load(open(file_))
                if subject not in subject_lens.keys():
                    subject_lens[subject] = []
                subject_lens[subject].append(len(data['request_states']))

        # assert that the data is the same for all instances
        for subject in subject_lens.keys():
            assert len(set(subject_lens[subject])) == 1

        # get a list of subjects and models
        all_models = []
        all_subjects = []
        data_dict = {}
        for sample in chosen_files:
            file_, model, subject = sample
            all_models.append(model)
            all_subjects.append(subject)
            if model not in data_dict.keys():
                data_dict[model] = {}
            assert subject not in data_dict[model].keys()
            data_dict[model][subject] = json.load(open(file_))
        all_models = list(set(all_models))
        all_subjects = list(set(all_subjects))
        print('models total: ', len(all_models))
        for model in sorted(all_models):
            print(model)

        incomplete_models = []
        for model in all_models:
            for subject in all_subjects:
                if subject not in data_dict[model].keys():
                    incomplete_models.append(model)
        incomplete_models = list(set(incomplete_models))
        print('incomplete models: ', incomplete_models)

        # transform the data into our format
        raw_data = data_dict[picked_models[0]]
        filtered_qa = {}
        for subject in all_subjects:
            for sample in raw_data[subject]['request_states']:
                sample_id = sample['instance']['id']
                if 'perturbation' in sample['instance'].keys():
                    continue
                key = '_'.join([target, subject, sample_id])
                key = key.replace('=', '_')
                options = []
                correct_option = None
                for idx, option in enumerate(sample['instance']['references']):
                    options.append(option['output']['text'])
                    if len(option['tags']) > 0:
                        if option['tags'][0] == 'correct':
                            correct_option = idx+1
                    else:
                        assert len(option['tags']) <= 1
                        assert option['tags'] == []
                if key in filtered_qa.keys():
                    assert filtered_qa[key]['question_text'] == sample['instance']['input']['text']
                    assert filtered_qa[key]['answer_text_short'] == correct_option
                    assert filtered_qa[key]['answer_text_long'] == options[correct_option-1]
                    assert filtered_qa[key]['main_topic'] == subject
                else:
                    filtered_qa[key] = {}
                    filtered_qa[key]['question_text'] = sample['instance']['input']['text']
                    filtered_qa[key]['answer_text_short'] = correct_option
                    filtered_qa[key]['answer_text_long'] = options[correct_option-1]
                    filtered_qa[key]['context'] = None
                    filtered_qa[key]['question_type'] = 'MC'
                    filtered_qa[key]['bloom_classification'] = None
                    filtered_qa[key]['difficulty_level'] = None
                    filtered_qa[key]['main_topic'] = subject
                    filtered_qa[key]['choices'] = options
        savepath = pathlib.Path(
            __file__).parents[0] / '..' / 'data' / (target+'.qar')
        savepath.parents[0].mkdir(parents=True, exist_ok=True)
        json.dump(filtered_qa, open(savepath, 'w'))

        # now get the .qaa
        for model in all_models:
            answer_dict = {}
            for subject in all_subjects:
                for sample in data_dict[model][subject]['request_states']:
                    if 'perturbation' in sample['instance'].keys():
                        continue
                    sample_id = sample['instance']['id']
                    key = '_'.join([target, subject, sample_id])
                    key = key.replace('=', '_')
                    if key not in answer_dict.keys():
                        if sample['result']['request_time'] != 0:
                            answer_time = sample['result']['request_time']
                        else:
                            answer_time = sample['result']['batch_request_time'] / \
                                sample['result']['batch_size']
                        answer_dict[key] = {}
                        answer_dict[key]['answer_text_short'] = letter_list.index(
                            sample['result']['completions'][0]['text'].strip())
                        answer_dict[key]['answer_text_long'] = None
                        answer_dict[key]['answer_time'] = answer_time
                        answer_dict[key]['score_f1'] = None
                        answer_dict[key]['score_exact'] = answer_dict[key]['answer_text_short'] == filtered_qa[key]['answer_text_short']
                        answer_dict[key]['score_default'] = answer_dict[key]['score_exact']
                    else:
                        if answer_dict[key]['score_exact']:
                            continue
                        else:
                            if letter_list.index(sample['result']['completions'][0]['text'].strip()) == filtered_qa[key]['answer_text_short']:
                                if sample['result']['request_time'] != 0:
                                    answer_time = sample['result']['request_time']
                                else:
                                    answer_time = sample['result']['batch_request_time'] / \
                                        sample['result']['batch_size']
                                answer_dict[key]['answer_text_short'] = letter_list.index(
                                    sample['result']['completions'][0]['text'].strip())
                                answer_dict[key]['answer_text_long'] = None
                                answer_dict[key]['answer_time'] = answer_time
                                answer_dict[key]['score_f1'] = None
                                answer_dict[key]['score_exact'] = answer_dict[key]['answer_text_short'] == filtered_qa[key]['answer_text_short']
                                answer_dict[key]['score_default'] = answer_dict[key]['score_exact']
            savepath = pathlib.Path(
                __file__).parents[0] / '..' / 'data' / ('_'.join([target, model])+'.qaa')
            json.dump(answer_dict, open(savepath, 'w'))

        # and now get the shallow .hir file which only branches the dataset into the given subjects:
        hir = {'name': target, 'children': []}
        for subject in all_subjects:
            sub_dict = {'name': subject, 'question_keys': [
                key for key in filtered_qa.keys() if filtered_qa[key]['main_topic'] == subject]}
            hir['children'].append(copy.copy(sub_dict))
        savepath = pathlib.Path(
            __file__).parents[0] / '..' / 'data' / (target+'.hir')
        json.dump(hir, open(savepath, 'w'))

    elif target == 'wikifact':
        unperturbed = True
        # get the files we are interested in
        chosen_files = []
        subject_lens = {}
        for file_ in glob.glob(os.path.join(data_path, '*')):
            filename = os.path.basename(file_)
            if 'wikifact_k=5' in filename and any([model in filename for model in picked_models]):
                subject = filename.split('_subject=')[1].split('_model')[0]
                model = filename.split('_model=')[1]
                model = model.replace('.json', '')
                chosen_files.append((file_, model, subject))
                data = json.load(open(file_))
                if subject not in subject_lens.keys():
                    subject_lens[subject] = []
                subject_lens[subject].append(len(data['request_states']))

        # assert that the data is the same for all instances
        for subject in subject_lens.keys():
            assert len(set(subject_lens[subject])) == 1

        # get a list of subjects and models
        all_models = []
        all_subjects = []
        data_dict = {}
        for sample in chosen_files:
            file_, model, subject = sample
            all_models.append(model)
            all_subjects.append(subject)
            if model not in data_dict.keys():
                data_dict[model] = {}
            assert subject not in data_dict[model].keys()
            data_dict[model][subject] = json.load(open(file_))
        all_models = list(set(all_models))
        all_subjects = list(set(all_subjects))
        print('models total: ', len(all_models))
        for model in sorted(all_models):
            print(model)

        incomplete_models = []
        for model in all_models:
            for subject in all_subjects:
                if subject not in data_dict[model].keys():
                    incomplete_models.append(model)
        incomplete_models = list(set(incomplete_models))
        print('incomplete models: ', incomplete_models)

        # transform the data into our format
        raw_data = data_dict[picked_models[0]]
        filtered_qa = {}
        for subject in all_subjects:
            for sample in raw_data[subject]['request_states']:
                sample_id = sample['instance']['id']
                if 'perturbation' in sample['instance'].keys():
                    continue
                key = '_'.join([target, subject, sample_id])
                key = key.replace('=', '_')
                options = []
                for idx, option in enumerate(sample['instance']['references']):
                    options.append(option['output']['text'])
                if key in filtered_qa.keys():
                    assert filtered_qa[key]['question_text'] == sample['instance']['input']['text']
                    assert filtered_qa[key]['main_topic'] == subject
                else:
                    filtered_qa[key] = {}
                    filtered_qa[key]['question_text'] = sample['instance']['input']['text']
                    filtered_qa[key]['answer_text_short'] = None
                    filtered_qa[key]['answer_text_long'] = options
                    filtered_qa[key]['context'] = None
                    filtered_qa[key]['question_type'] = 'open'
                    filtered_qa[key]['bloom_classification'] = None
                    filtered_qa[key]['difficulty_level'] = None
                    filtered_qa[key]['main_topic'] = subject
                    filtered_qa[key]['choices'] = options
        savepath = pathlib.Path(
            __file__).parents[0] / '..' / 'data' / (target+'.qar')
        savepath.parents[0].mkdir(parents=True, exist_ok=True)
        json.dump(filtered_qa, open(savepath, 'w'))

        # now get the .qaa
        for model in all_models:
            answer_dict = {}
            for subject in all_subjects:
                for sample in data_dict[model][subject]['request_states']:
                    if 'perturbation' in sample['instance'].keys():
                        continue
                    sample_id = sample['instance']['id']
                    key = '_'.join([target, subject, sample_id])
                    key = key.replace('=', '_')
                    if key not in answer_dict.keys():
                        if sample['result']['request_time'] != 0:
                            answer_time = sample['result']['request_time']
                        else:
                            answer_time = sample['result']['batch_request_time'] / \
                                sample['result']['batch_size']
                        answer_dict[key] = {}
                        answer_dict[key]['answer_text_short'] = sample['result']['completions'][0]['text'].strip(
                        )
                        answer_dict[key]['answer_text_long'] = sample['result']['completions'][0]['text'].strip(
                        )
                        answer_dict[key]['answer_time'] = answer_time
                        answer_dict[key]['score_f1'] = None
                        answer_dict[key]['score_exact'] = any([normalize_text(answer_dict[key]['answer_text_short']) == normalize_text(
                            option) for option in filtered_qa[key]['answer_text_long']])
                        answer_dict[key]['score_default'] = answer_dict[key]['score_exact']
                    else:
                        if answer_dict[key]['score_exact']:
                            continue
                        else:
                            new_score = any([normalize_text(sample['result']['completions'][0]['text'].strip(
                            )) == normalize_text(option) for option in filtered_qa[key]['answer_text_long']])
                            if new_score:
                                if sample['result']['request_time'] != 0:
                                    answer_time = sample['result']['request_time']
                                else:
                                    answer_time = sample['result']['batch_request_time'] / \
                                        sample['result']['batch_size']
                                answer_dict[key]['answer_text_short'] = sample['result']['completions'][0]['text'].strip(
                                )
                                answer_dict[key]['answer_text_long'] = sample['result']['completions'][0]['text'].strip(
                                )
                                answer_dict[key]['answer_time'] = answer_time
                                answer_dict[key]['score_default'] = new_score
                                answer_dict[key]['score_f1'] = None
                                answer_dict[key]['score_exact'] = new_score
            savepath = pathlib.Path(
                __file__).parents[0] / '..' / 'data' / ('_'.join([target, model])+'.qaa')
            json.dump(answer_dict, open(savepath, 'w'))

        # and now get the shallow .hir file which only branches the dataset into the given subjects:
        hir = {'name': target, 'children': []}
        for subject in all_subjects:
            sub_dict = {'name': subject, 'question_keys': [
                key for key in filtered_qa.keys() if filtered_qa[key]['main_topic'] == subject]}
            hir['children'].append(copy.copy(sub_dict))
        savepath = pathlib.Path(
            __file__).parents[0] / '..' / 'data' / (target+'.hir')
        json.dump(hir, open(savepath, 'w'))

else:
    if target == 'pubmedqa':
        qa = json.load(open(data_path))
        filtered_qa = {}
        for key in qa.keys():
            filtered_qa[key] = {}
            # Text of the question, without answer options.
            filtered_qa[key]['question_text'] = qa[key]['QUESTION']
            # Short Version of the anser. For MC this is just a number from 1 to the number of choices
            filtered_qa[key]['answer_text_short'] = qa[key]['final_decision']
            # Long answer version, currently not utilized in the rest of the code. Might be the same as short answer.
            filtered_qa[key]['answer_text_long'] = qa[key]['LONG_ANSWER']
            # Context string, optional. Should not render the task into text understanding
            filtered_qa[key]['context'] = qa[key]['CONTEXTS']
            # Type of the question, currently we incorporate 'yes/no'maybe' and 'MC'
            filtered_qa[key]['question_type'] = 'yes/no/maybe'
            # will be filled later
            filtered_qa[key]['bloom_classification'] = None
            filtered_qa[key]['difficulty_level'] = None  # will be filled later
            # Main topic of the qa, different behavior in generate_hierarchy for None/not-None, lists or ';' seperated strings
            filtered_qa[key]['main_topic'] = 'Biomedical Research'
            # Choice options for MC datasets
            filtered_qa[key]['choices'] = None

        savepath = pathlib.Path(
            __file__).parents[0] / '..' / 'data' / (target+'.qar')
        savepath.parents[0].mkdir(parents=True, exist_ok=True)
        json.dump(filtered_qa, open(savepath, 'w'))

    elif target == 'medmcqa':
        from datasets import load_dataset
        dataset = load_dataset("medmcqa")
        filtered_qa = {}
        for sample in dataset['validation']:  # test does not contain labels
            if sample['subject_name'] != 'Unknown':
                key = sample['id']
                filtered_qa[key] = {}
                filtered_qa[key]['question_text'] = sample['question']
                filtered_qa[key]['answer_text_short'] = sample['cop']+1
                filtered_qa[key]['answer_text_long'] = sample['exp']
                filtered_qa[key]['context'] = None
                filtered_qa[key]['question_type'] = 'MC'
                filtered_qa[key]['bloom_classification'] = None
                filtered_qa[key]['difficulty_level'] = None
                # Main topic of the qa, different behavior in generate_hierarchy for None/not-None
                filtered_qa[key]['main_topic'] = sample['subject_name']
                filtered_qa[key]['choices'] = [sample['opa'], sample['opb'],
                                               sample['opc'], sample['opd']]  # Choice options for MC datasets
        savepath = pathlib.Path(
            __file__).parents[0] / '..' / 'data' / (target+'.qar')
        savepath.parents[0].mkdir(parents=True, exist_ok=True)
        json.dump(filtered_qa, open(savepath, 'w'))

    elif target == 'sciq':
        from datasets import load_dataset
        dataset = load_dataset("sciq")
        answers = ['A', 'B', 'C', 'D']
        print(dataset)
        print(dataset['test'])
        filtered_qa = {}
        for idx, sample in enumerate(dataset['test']):
            key = f'sciq_test_id_{idx}'
            options = [sample['correct_answer'], sample['distractor1'],
                       sample['distractor2'], sample['distractor3']]
            random.shuffle(options)
            filtered_qa[key] = {}
            filtered_qa[key]['question_text'] = sample['question']
            filtered_qa[key]['answer_text_short'] = options.index(
                sample['correct_answer'])+1
            filtered_qa[key]['answer_text_long'] = sample['correct_answer']
            filtered_qa[key]['context'] = sample['support']
            filtered_qa[key]['question_type'] = 'MC'
            filtered_qa[key]['bloom_classification'] = None
            filtered_qa[key]['difficulty_level'] = None
            # Main topic of the qa, different behavior in generate_hierarchy for None/not-None or string/list/dict
            filtered_qa[key]['main_topic'] = 'Science'
            # Choice options for MC datasets
            filtered_qa[key]['choices'] = options
        savepath = pathlib.Path(
            __file__).parents[0] / '..' / 'data' / (target+'.qar')
        savepath.parents[0].mkdir(parents=True, exist_ok=True)
        json.dump(filtered_qa, open(savepath, 'w'))

    elif target == 'usbar':
        # NOTE that you my have to change the mapping of answer_id and category_id to new values as we do not now the hidden data structure of NCBE in extent to what we got.
        # Also, the usbar data has an existing difficulty ranking. Currently this is not used/imported as we were interested in the model assigned difficulty.
        # In case you want to use it, the lines are there and can just be (un-)commented

        practice = json.load(
            open(os.path.join(data_path, 'NCBE_MBE_practice.json')))
        sim1 = json.load(
            open(os.path.join(data_path, 'NCBE_MBE_simulated_1.json')))
        sim2 = json.load(
            open(os.path.join(data_path, 'NCBE_MBE_simulated_2.json')))
        sim3 = json.load(
            open(os.path.join(data_path, 'NCBE_MBE_simulated_3.json')))

        # Some dictionaries to map the cryptic entries of the obtained json files to usable values (category names and answer indices)
        map_category_id_practice = {
            '31763': 'Civil Procedure',
            '31764': 'Constitutional Law',
            '31765': 'Contracts',
            '31766': 'Criminal Law and Procedure',
            '31767': 'Evidence',
            '31768': 'Real Property',
            '31769': 'Torts'
        }

        map_category_id_sim1 = {
            '34586': 'Torts',
            '34587': 'Real Property',
            '34588': 'Civil Procedure',
            '34589': 'Contracts',
            '34590': 'Evidence',
            '34591': 'Criminal Law and Procedure',
            '34592': 'Constitutional Law'
        }

        map_category_id_sim2 = {
            '34593': 'Constitutional Law',
            '34594': 'Real Property',
            '34595': 'Civil Procedure',
            '34596': 'Evidence',
            '34597': 'Criminal Law and Procedure',
            '34598': 'Torts',
            '34599': 'Contracts'
        }

        map_category_id_sim3 = {
            '47196': 'Civil Procedure',
            '47197': 'Constitutional Law',
            '47198': 'Contracts',
            '47199': 'Criminal Law and Procedure',
            '47200': 'Evidence',
            '47201': 'Real Property',
            '47202': 'Torts'
        }

        map_answer = {
            'RA==': 4,  # D
            'QQ==': 1,  # A
            'Qg==': 2,  # B
            'Qw==': 3  # C
        }

        map_difficulty = {
            'Easy': 1,
            'Moderate': 2,
            'Difficult': 3,
            'Expert': 4,
        }

        filtered_qa = {}

        def fill_qa_dict(json_data, category_map):
            for item in json_data:
                key = item['id']
                filtered_qa[key] = {}

                text = item['question_content']
                question_and_context = []
                options = []
                # Needed to fix indices [1288770, 1290120, 1290123], since the identation of answer options is missing:
                num_identations = 0
                for line in text.split('\n'):
                    if '\t' in line:
                        num_identations += 1

                for line in text.split('\n'):
                    try:
                        line = line.replace(re.findall(
                            r' paraeid.*?>', repr(line))[0][:-1], '')
                    except:
                        pass
                    if line.startswith('<p>'):
                        line = line.replace('<p>', '')
                        line = line.replace('</p>', '')
                        line = line.replace('\r', '')
                        question_and_context.append(line)
                    if line.startswith('\t') and num_identations > 0:
                        line = line.replace('\t<li>', '')
                        line = line.replace('</li>\r', '')
                        options.append(line)
                    elif line.startswith('<li>') and num_identations == 0:
                        line = line.replace('<li>', '')
                        line = line.replace('</li>\r', '')
                        options.append(line)

                context = '\n'.join(question_and_context[:-1])
                question = question_and_context[-1]
                # print(options)
                if len(options) == 5:  # In the data this is question id 964387
                    context = options[0]
                    options.pop(0)
                assert len(options) == 4

                answer_idx = map_answer[item['correct_answer']]
                category = category_map[str(item['question_category_id'])]
                difficulty = item['difficulty']

                filtered_qa[key]['question_text'] = question
                filtered_qa[key]['answer_text_short'] = answer_idx
                filtered_qa[key]['answer_text_long'] = options[answer_idx-1]
                filtered_qa[key]['context'] = context
                filtered_qa[key]['question_type'] = 'MC'
                filtered_qa[key]['bloom_classification'] = None
                filtered_qa[key]['difficulty_level'] = None
                # filtered_qa[key]['difficulty_level'] = map_difficulty[difficulty] #YOu can use the existing difficulty levels in the data, we did not to stay comparable to all other visualizations in out paper.
                filtered_qa[key]['main_topic'] = category
                # Choice options for MC datasets
                filtered_qa[key]['choices'] = options

        fill_qa_dict(practice, map_category_id_practice)
        fill_qa_dict(sim1, map_category_id_sim1)
        fill_qa_dict(sim2, map_category_id_sim2)
        fill_qa_dict(sim3, map_category_id_sim3)
        assert len(list(filtered_qa.keys())) == 625
        savepath = pathlib.Path(
            __file__).parents[0] / '..' / 'data' / (target+'.qar')
        savepath.parents[0].mkdir(parents=True, exist_ok=True)
        json.dump(filtered_qa, open(savepath, 'w'))
