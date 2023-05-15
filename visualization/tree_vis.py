from anytree import Node, search, PreOrderIter, LevelOrderGroupIter
import svgwrite
import svgpathtools
import random
import json
import argparse
import math
import os
import pathlib
import uuid
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from SecretColors import Palette
import SecretColors as sc
from shapely.geometry import Point, Polygon
import colorsys
import re

parser = argparse.ArgumentParser()
parser.add_argument('--draw_empty_leaf', help='draw empty leaf nodes', action='store_true')
parser.add_argument('--draw_qna_samples', help='draw blue noise plot of num questions in node', action='store_true')
parser.add_argument('--max_num_stacked_leafs', help='maximum number of horizontally stacked leaves before vertical stacking starts.', type=int, default=9)
parser.add_argument('--layer_width', help='Width of each cell.', type=int, default=380)
parser.add_argument('--qna_sample_quality', help='the generation of blue noise samples dominates processing time, thus you can set the quality to normal or high.', type=str, default='high')
parser.add_argument('--llm_models', nargs='+', help='list of llm models as they appear in the .qaa files. can be passed instead of path list.', type=str, default=None, required=True)
parser.add_argument('--datasets', nargs='+', help='list of datasets as they appear in the .qaa,.qar and .hir files. can be passed instead of path list.', type=str, default=None, required=True)
args = parser.parse_args()

# Define on node class with font_size, bounding_box as well as bounding box for direct children
bloom_classification_labels = ['remember', 'understand', 'apply', 'analyze', 'evaluate', 'create'] # List of bloom classifications with the same order as in the nodes.
class MyNode(Node):
    def __init__(self, name, bounding_box=(0, 0, 0, 0), bounding_box_children=(0, 0, 0, 0), font_size=8, parent=None, accuracy_aggregate=[], num_questions=0, 
                 average_time_aggregate=[], accuracy_list=[], average_time_list=[], datasets=[], llm_models=[], 
                 bloom_classification_aggregate=[], bloom_classification_list=[], difficulty_level_aggregate=[], difficulty_level_list=[], hallucination_score=None, **kwargs):
        super().__init__(name, parent=parent, **kwargs)
        self.bounding_box = bounding_box
        self.bounding_box_children = bounding_box_children
        self.font_size = font_size
        self.llm_models = llm_models #list of llm_models order is kept for accuracy and time. e.g. accuracy list has shape [num_models, num_questions] and accuracy has shape [num_models]
        self.datasets = datasets
        self.accuracy_aggregate = accuracy_aggregate #range 0-1
        self.accuracy_list = accuracy_list
        self.num_questions = num_questions
        self.average_time_aggregate = average_time_aggregate
        self.average_time_list = average_time_list
        self.bloom_classification_aggregate = bloom_classification_aggregate # List of tuples for each model. Tuple is (num_questions, accuracy) per bloom class  
        self.bloom_classification_list = bloom_classification_list # List of accuracy_lists for each model, one accuracy_list per bloom class
        self.difficulty_level_aggregate = difficulty_level_aggregate
        self.difficulty_level_list = difficulty_level_list
        self.hallucination_score = hallucination_score

   
# Function to prepare datasets in such a way that they are similar. I.e. removing questions that are not answered by all models and if requested removing empty leaf nodes.
def remove_bad_keys(keys, parent):
    if 'children' in parent.keys():
        for child in parent['children']:
            remove_bad_keys(keys, child)
    else:
        for key in keys:
            if key in parent['question_keys']:
                parent['question_keys'].pop(parent['question_keys'].index(key))


# removes empty leaf nodes
def remove_empty_leaf_nodes(parent):
    if 'children' in parent.keys():
        idx_to_pop = []
        for child in parent['children']:
            if 'children' in child.keys():
                remove_empty_leaf_nodes(child)
            if 'question_keys' in child.keys():
                if len(child['question_keys']) == 0:
                    idx_to_pop.append(parent['children'].index(child))
            elif 'children' in child.keys():
                if len(child['children']) == 0:
                    idx_to_pop.append(parent['children'].index(child))
        for idx in idx_to_pop[::-1]:
            parent['children'].pop(idx)


# Remove the question indices from all files which were used for few_shot prompting:
def clean_data(answer_data, question_data, data_left, data_right):
    all_answer_keys = []
    for a_data in answer_data:
        all_answer_keys.append(list(a_data.keys()))
    used_keys = set(all_answer_keys[0])
    for i in range(1, len(all_answer_keys)):
        used_keys = used_keys.intersection(set(all_answer_keys[i]))
    keys_to_remove = list(set(question_data.keys())-used_keys)
    for key in keys_to_remove: 
        del question_data[key]
      
    remove_bad_keys(keys_to_remove, data_left)
    remove_bad_keys(keys_to_remove, data_right)
            
    if not args.draw_empty_leaf:
        remove_empty_leaf_nodes(data_left)
        remove_empty_leaf_nodes(data_right)

    return answer_data, question_data, data_left, data_right


# get the number of leaves for the parent
def get_num_leaves(parent):
    num_leaves = 0
    if 'children' in parent.keys():
        for child in parent['children']:
            if 'children' in child.keys():
                num_leaves_ = get_num_leaves(child)
                num_leaves += num_leaves_
            else:
                num_leaves += 1
    else:
        num_leaves += 1
    return num_leaves


# get the hallucination score
def get_hallucination_score(xt):
    '''
    Input: sorted list of accuracy values (with rising difficulty)
    '''
    x = []
    for n, a in xt:
        if n > 1:
            x.append(a)
    n = len(x)
    monotonicity = 1/(n-1) * sum([np.sign(x[i+1]-x[i]) for i in range(n-1)])
    monotonicity = (monotonicity + 1)/2
    return monotonicity


#This function assumes what is asserted before, that we have all models for all datasets.
def get_combined_sets(dataset_names=[], model_names=[], data_base_path=''):
    question_data = {}
    answer_data = [{} for i in range(len(model_names))]
    hierarchy_data_list = []
    for dataset_name in dataset_names:
        q_path = data_base_path / '{}.qar'.format(dataset_name)
        q_data = json.load(open(q_path))
        question_data = {**question_data, **q_data}
        for model_idx, model_name in enumerate(model_names):
            a_path = data_base_path / '{}_{}.qaa'.format(dataset_name, model_name)
            a_data = json.load(open(a_path))
            answer_data[model_idx] = {**answer_data[model_idx], **a_data}
        h_path = data_base_path / '{}.hir'.format(dataset_name)
        hierarchy_data_list.append(json.load(open(h_path)))

    # get number of leaf_nodes per dataset hierarchy
    num_leaves = []
    leaves_per_branch = []
    for h_data in hierarchy_data_list:
        per_branch = []
        num_leaves.append(get_num_leaves(h_data))
        for child in h_data['children']:
            per_branch.append(get_num_leaves(child))
        leaves_per_branch.append(per_branch)

    #branches are already sorted from .hir generation, now sort the data list (large to small)
    sorting = np.argsort(num_leaves)[::-1]
    num_leaves = [num_leaves[i] for i in sorting]
    hierarchy_data_list = [hierarchy_data_list[i] for i in sorting]
    leaves_per_branch = [leaves_per_branch[i] for i in sorting]

    #simple sorting to get roughly symmetric trees. 
    if len(hierarchy_data_list)==2:
        root_left = MyNode(hierarchy_data_list[0]['name'])
        root_right = MyNode(hierarchy_data_list[1]['name'])
        data_left = {'name': hierarchy_data_list[0]['name'], 'children':[hierarchy_data_list[0]]}
        data_right = {'name': hierarchy_data_list[1]['name'], 'children':[hierarchy_data_list[1]]}
    else:
        if len(hierarchy_data_list)%2==0:
            root_left = MyNode('multiple_datasets')
            root_right = MyNode('multiple_datasets')
            data_left = {'name': 'multiple_datasets', 'children':hierarchy_data_list[::2]}
            data_right = {'name': 'multiple_datasets', 'children':hierarchy_data_list[1::2]}
        else:
            root_left = MyNode('multiple_datasets')
            root_right = MyNode('multiple_datasets')
            children_left = hierarchy_data_list[:-1:2]
            children_right = hierarchy_data_list[1:-1:2]
            last_split_left = {'name':hierarchy_data_list[-1]['name'], 'children':hierarchy_data_list[-1]['children'][1::2]}
            last_split_right = {'name':hierarchy_data_list[-1]['name'], 'children':hierarchy_data_list[-1]['children'][::2]}
            children_left.append(last_split_left)
            children_right.append(last_split_right)
            data_left = {'name': 'multiple_datasets', 'children':children_left}
            data_right = {'name': 'multiple_datasets', 'children':children_right}
    return answer_data, question_data, root_left, root_right, data_left, data_right


#load the data
def load_tree_from_json():
    '''
    Adapted for multiple models, means also that node attributes concerned of this will be wrapped in a list, even if only one model is used.
    We currently expect that all datasets in input are processed by all models in input.
    Iteration over different datasets needs to be adapted still.
    '''
    # Load the data 
    if args.llm_models is None or args.datasets is None:
        raise ValueError('Please provide at least one dataset and one llm model in the arguments.')
    else:
        qa_dataset_names = args.datasets
        llm_model_names = args.llm_models
        if not isinstance(qa_dataset_names, list):
            qa_dataset_names = [qa_dataset_names]
        if not isinstance(llm_model_names, list):
            llm_model_names = [llm_model_names]
    qa_dataset_names = sorted(list(set(qa_dataset_names)))
    llm_model_names = sorted(list(set(llm_model_names)))
    custom_sorted_llm_models = []
    for model in ['bloom', 'gpt2', 'davinci', 'chatgpt', 'llama13b']:
        if model in llm_model_names:
            custom_sorted_llm_models.append(model)
    for model in llm_model_names:
        if model not in custom_sorted_llm_models:
            custom_sorted_llm_models.append(model)
    llm_model_names = custom_sorted_llm_models
    #Checking if every possible pairing of the Input is given:
    
    data_path = pathlib.Path(__file__).parents[0] / '..' / 'data' 
    num_models = len(llm_model_names)

    # answer_data is a list with len(num_models)
    # question_data is just one dictionary
    # hierarchy_data is just one dictionary 
    if len(qa_dataset_names)>1:
        # Combine the datasets to a single tree. The hierarchy is extended to involve all_datasets and the q/a datasets are just combined, as no keys should duplicate across datasets
        # In case more datasets are added, their keys have to be made sure to not overlap.
        answer_data, question_data, root_left, root_right, data_left, data_right = get_combined_sets(qa_dataset_names, llm_model_names, data_path)
    else:
        h_path = data_path / '{}.hir'.format(qa_dataset_names[0])
        hierarchy_data = json.load(open(h_path))

        answer_data = [{} for i in range(len(llm_model_names))]
        for model_idx, model_name in enumerate(llm_model_names):
            a_path = data_path / '{}_{}.qaa'.format(qa_dataset_names[0], model_name)
            a_data = json.load(open(a_path))
            answer_data[model_idx] = {**a_data}

        #load question_data to get bloom taxonomy and difficulty_level
        q_path = data_path / '{}.qar'.format(qa_dataset_names[0])
        question_data = json.load(open(q_path))


        root_left = MyNode(hierarchy_data['name'])
        root_right = MyNode(hierarchy_data['name'])
        left_split = []
        right_split = []
        left_leaves = 0
        right_leaves = 0
        for child in hierarchy_data['children']:
            if left_leaves <= right_leaves:
                left_split.append(child)
                left_leaves += get_num_leaves(child)
            else:
                right_split.append(child)
                right_leaves += get_num_leaves(child)
                
        data_left = {'name': hierarchy_data['name'], 'children':left_split}
        data_right = {'name': hierarchy_data['name'], 'children':right_split}

    answer_data, question_data, data_left, data_right = clean_data(answer_data, question_data, data_left, data_right)

    #NOTE: This score selection assumes that the question type across a single dataset is constant. 
    #If we want to adapt for different cases, we need to do some (more complex) stuff in import.
    take_score = 'default'
    if take_score == 'default':
        my_score = []
        for i in range(num_models):
            if answer_data[i][list(answer_data[i].keys())[0]]['score_default'] in answer_data[i][list(answer_data[i].keys())[0]]:
                my_score.append(answer_data[i][list(answer_data[i].keys())[0]]['score_default'])
            else:
                my_score.append('score_default')
    elif take_score == 'exact':
        my_score = ['score_exact']*num_models
    elif take_score == 'f1':
        my_score = ['score_f1']*num_models
    
    def collect_child_data(node, node_dict):
        for child_data in node_dict['children']:
            if 'children' in child_data.keys():
                child = MyNode(child_data['name'], parent=node)
                collect_child_data(child, child_data)
            else:
                question_keys = child_data['question_keys']
                num_questions = len(question_keys)
                accuracies = [[] for _ in range(num_models)]
                times = [[] for _ in range(num_models)]
                bloom_classification_list = [[[] for _ in range(len(bloom_classification_labels))] for __ in range(num_models)]
                difficulty_level_list = [[[] for _ in range(5)] for __ in range(num_models)]
                for key in question_keys:
                    for i in range(num_models):
                        current_accuracy = answer_data[i][key][my_score[i]]
                        accuracies[i].append(current_accuracy)
                        times[i].append(answer_data[i][key]['answer_time'])
                        if question_data[key]['bloom_classification'] is not None:
                            bloom_classification_list[i][bloom_classification_labels.index(question_data[key]['bloom_classification'])].append(current_accuracy)
                        if question_data[key]['difficulty_level'] is not None:
                            difficulty_level_list[i][int(question_data[key]['difficulty_level'])-1].append(current_accuracy)
                accuracy_aggregate = [0 for _ in range(num_models)]
                average_response_time = [0 for _ in range(num_models)]
                bloom_classification_aggregate = [[(0,0) for _ in range(len(bloom_classification_labels))] for __ in range(num_models)]
                difficulty_level_aggregate = [[(0,0) for _ in range(5)] for __ in range(num_models)]
                for i in range(num_models):
                    if (len(accuracies[i]) > 0):
                        accuracy_aggregate[i] = sum(accuracies[i])/len(accuracies[i])
                    if (len(times[i]) > 0):
                        average_response_time[i] = sum(times[i])/len(times[i])
                    for j in range(len(bloom_classification_labels)):
                        class_accuracies = bloom_classification_list[i][j]
                        if len(class_accuracies) > 0:
                            bloom_classification_aggregate[i][j] = (len(class_accuracies), sum(class_accuracies)/len(class_accuracies))
                    for j in range(5):
                        class_accuracies = difficulty_level_list[i][j]
                        if len(class_accuracies) > 0:
                            difficulty_level_aggregate[i][j] = (len(class_accuracies), sum(class_accuracies)/len(class_accuracies))

                child = MyNode(child_data['name'], parent=node, accuracy_aggregate=accuracy_aggregate, 
                               num_questions=num_questions, average_time_aggregate=average_response_time,
                               accuracy_list=accuracies.copy(), average_time_list=times.copy(),
                               bloom_classification_list=bloom_classification_list, bloom_classification_aggregate=bloom_classification_aggregate,
                               difficulty_level_list=difficulty_level_list, difficulty_level_aggregate=difficulty_level_aggregate,
                               llm_models=llm_model_names, datasets=qa_dataset_names)

    collect_child_data(root_left, data_left)
    collect_child_data(root_right, data_right)

    def propagate_leaf_data(node):
        accuracies = [[] for i in range(num_models)]
        num_questions = 0
        average_times = [[] for i in range(num_models)]
        bloom_classification_list = [[[] for _ in range(len(bloom_classification_labels))] for __ in range(num_models)]
        difficulty_level_list = [[[] for _ in range(5)] for __ in range(num_models)]
        for child in node.children:
            if child.children:
                accuracies_, num_questions_, average_times_, bloom_classification_list_, difficulty_level_list_ = propagate_leaf_data(child)
            else: 
                accuracies_ = child.accuracy_list
                num_questions_ = child.num_questions
                average_times_ = child.average_time_list
                bloom_classification_list_ = child.bloom_classification_list
                difficulty_level_list_ = child.difficulty_level_list
            num_questions += num_questions_
            for i in range(num_models):
                accuracies[i] += accuracies_[i].copy()
                average_times[i] += average_times_[i]
                for j in range(len(bloom_classification_labels)):
                    bloom_classification_list[i][j] += bloom_classification_list_[i][j]
                for j in range(5):
                    difficulty_level_list[i][j] += difficulty_level_list_[i][j]

        accuracy_aggregate = [0 for _ in range(num_models)]
        average_response_time = [0 for _ in range(num_models)]
        bloom_classification_aggregate = [[(0,0) for _ in range(len(bloom_classification_labels))] for __ in range(num_models)]
        difficulty_level_aggregate = [[(0,0) for _ in range(5)] for __ in range(num_models)]
        for i in range(num_models):
            if (len(accuracies[i]) > 0):
                accuracy_aggregate[i] = sum(accuracies[i])/len(accuracies[i])
            if (len(average_times[i]) > 0):
                average_response_time[i] = sum(average_times[i])/len(average_times[i])
            for j in range(len(bloom_classification_labels)):
                class_accuracies = bloom_classification_list[i][j]
                if len(class_accuracies) > 0:
                    bloom_classification_aggregate[i][j] = (len(class_accuracies), sum(class_accuracies)/len(class_accuracies))
            for j in range(5):
                class_accuracies = difficulty_level_list[i][j]
                if len(class_accuracies) > 0:
                    difficulty_level_aggregate[i][j] = (len(class_accuracies), sum(class_accuracies)/len(class_accuracies))
        node.accuracy_aggregate = accuracy_aggregate
        node.accuracy_list = accuracies.copy()
        node.num_questions = num_questions
        node.average_time_aggregate = average_response_time
        node.average_time_list = average_times.copy()
        node.bloom_classification_list = bloom_classification_list.copy()
        node.bloom_classification_aggregate = bloom_classification_aggregate.copy()
        node.difficulty_level_list = difficulty_level_list.copy()
        node.difficulty_level_aggregate = difficulty_level_aggregate.copy()
        node.llm_models = llm_model_names
        node.datasets = qa_dataset_names

        return node.accuracy_list, node.num_questions, node.average_time_list, node.bloom_classification_list, node.difficulty_level_list

    propagate_leaf_data(root_left)
    propagate_leaf_data(root_right)

    #get a single root_node with the combined data of both branches.
    root_node = MyNode('Combined stats')

    accuracies = [[] for i in range(num_models)]
    num_questions = 0
    average_times = [[] for i in range(num_models)]
    bloom_classification_list = [[[] for _ in range(len(bloom_classification_labels))] for __ in range(num_models)]
    difficulty_level_list = [[[] for _ in range(5)] for __ in range(num_models)]
    for node in [root_left, root_right]:
        accuracies_ = node.accuracy_list
        average_times_ = node.average_time_list
        bloom_classification_list_ = node.bloom_classification_list
        difficulty_level_list_ = node.difficulty_level_list
        num_questions += node.num_questions
        for i in range(num_models):
            accuracies[i] += accuracies_[i].copy()
            average_times[i] += average_times_[i]
            for j in range(len(bloom_classification_labels)):
                bloom_classification_list[i][j] += bloom_classification_list_[i][j]
            for j in range(5):
                difficulty_level_list[i][j] += difficulty_level_list_[i][j]

    accuracy_aggregate = [0 for _ in range(num_models)]
    average_response_time = [0 for _ in range(num_models)]
    bloom_classification_aggregate = [[(0,0) for _ in range(len(bloom_classification_labels))] for __ in range(num_models)]
    difficulty_level_aggregate = [[(0,0) for _ in range(5)] for __ in range(num_models)]
    for i in range(num_models):
        if (len(accuracies[i]) > 0):
            accuracy_aggregate[i] = sum(accuracies[i])/len(accuracies[i])
        if (len(average_times[i]) > 0):
            average_response_time[i] = sum(average_times[i])/len(average_times[i])
        for j in range(len(bloom_classification_labels)):
            class_accuracies = bloom_classification_list[i][j]
            if len(class_accuracies) > 0:
                bloom_classification_aggregate[i][j] = (len(class_accuracies), sum(class_accuracies)/len(class_accuracies))
        for j in range(5):
            class_accuracies = difficulty_level_list[i][j]
            if len(class_accuracies) > 0:
                difficulty_level_aggregate[i][j] = (len(class_accuracies), sum(class_accuracies)/len(class_accuracies))
                
    root_node.accuracy_aggregate = accuracy_aggregate
    root_node.accuracy_list = accuracies.copy()
    root_node.num_questions = num_questions
    root_node.average_time_aggregate = average_response_time
    root_node.average_time_list = average_times.copy()
    root_node.bloom_classification_list = bloom_classification_list.copy()
    root_node.bloom_classification_aggregate = bloom_classification_aggregate.copy()
    root_node.difficulty_level_list = difficulty_level_list.copy()
    root_node.difficulty_level_aggregate = difficulty_level_aggregate.copy()
    root_node.llm_models = llm_model_names
    root_node.datasets = qa_dataset_names
    root_node.child_names = [child.name for child in root_left.children]+[child.name for child in root_right.children]
    
    return root_left, root_right, root_node


# Calculate font size for given layer by interpolating between min and max font size
def calc_font_size(layer, max_layer):
    #fontsize = [0, 24, 16, 13][layer]
    #return fontsize
    return int(font_size_min + (font_size_max-font_size_min)*(1.0-(layer-1)/(max_layer-1)))


# Calculate vertical spacing between labels for given layer by interpolating between min and max spacing
def calc_spacing(layer, max_layer):
    return spacing_min + (spacing_max-spacing_min)*(1.0-layer/max_layer)


# Calculate height of an element for a given layer as sum of its font size and spacing
def calc_element_height(layer, max_layer):
    return calc_font_size(layer, max_layer) + calc_spacing(layer, max_layer)


# Calculate height of entire llmmap
def calc_height(root, max_num_stacked_leafs):
    height = 0
    for node in PreOrderIter(root):
        if node.depth == max_depth-1:
            if (len(node.children) < max_num_stacked_leafs):
                num_stacked = len(node.children)
            else:
                num_stacked = max_num_stacked_leafs
            height = height + num_stacked*calc_element_height(max_depth, max_depth) + leaf_spacing
    return height + leaf_spacing # add one more leaf_spacing for last leaf group


# Return the maximal number of leaf nodes that are side-by-side based on max_num_stacked_leafs
def get_max_leaf_stack_depth(root, max_num_stacked_leafs):
    leaf_nodes = search.findall(root, lambda node: not node.children)
    max_leaf_stack_depth = max(math.ceil((len(node.siblings)+1)/max_num_stacked_leafs) for node in leaf_nodes)
    return max_leaf_stack_depth


# Position the text elements representing the nodes
def calc_label_positions(root, direction, layer_width, max_num_stacked_leafs, leaf_spacing):
    leaf_nodes = search.findall(root, lambda node: not node.children)
    max_depth = max(node.depth for node in leaf_nodes)
    cur_depth = max_depth

    prev_layer_min_y = 0
    prev_layer_max_y = total_height

    # Get a list of node groups with one group for each tree level and reverse it
    node_groups = list(LevelOrderGroupIter(root))
    node_groups.reverse()
    for node_group in node_groups:
        layer_pos_x = (cur_depth-1) * layer_width
        layer_font_size = calc_font_size(cur_depth, max_depth)
        layer_spacing = calc_spacing(cur_depth, max_depth)
        layer_height = len(node_group) * calc_element_height(cur_depth, max_depth)
        #TODO: allow for different offsets at different layers
        layer_offset_y = prev_layer_min_y + ((prev_layer_max_y-prev_layer_min_y)-layer_height) / 2
        prev_layer_min_y = layer_offset_y

        cur_node = 0
        cur_parent = 0
        cur_sibling = 1
        cur_leaf_node = 0
        base_offset = leaf_spacing
        leaf_grid_offset_y = 0

        num_siblings = 0
        for node in node_group:
            if (len(node.children) == 0):
                # Draw leaf nodes
                if (node.parent != cur_parent):
                    # If we encounter a node with a new parent, we have to update our base_offset and reset cur_sibling
                    cur_parent = node.parent
                    num_siblings = len(node.siblings)+1
                    if (cur_node > 0):
                        if (cur_sibling < max_num_stacked_leafs):
                            base_offset = base_offset + cur_sibling*(layer_font_size+layer_spacing) + leaf_spacing
                        else:
                            base_offset = base_offset + (max_num_stacked_leafs)*(layer_font_size+layer_spacing) + leaf_spacing
                    cur_sibling = 1
                else:
                    cur_sibling+=1

                num_siblings_first_column = num_siblings % max_num_stacked_leafs
                # Draw siblings in all other columens
                leaf_grid_offset_x = (cur_sibling-1) // (max_num_stacked_leafs)
                leaf_grid_offset_y = (cur_sibling-1) % (max_num_stacked_leafs)
                pos_x = layer_pos_x + leaf_grid_offset_x*layer_width + leaf_grid_offset_x*leaf_spacing/2
                pos_y = base_offset+(leaf_grid_offset_y*(layer_font_size+layer_spacing))
                if (leaf_grid_offset_x > 0 and cur_sibling > num_siblings-num_siblings_first_column):
                    pos_y += ((max_num_stacked_leafs-num_siblings_first_column)*(layer_font_size+layer_spacing))/2
                prev_layer_max_y = pos_y
                if (direction == "right"):
                    bounding_box = (pos_x, pos_y, layer_width*1, layer_font_size+layer_spacing)
                elif (direction == "left"):
                    bounding_box = (-pos_x, pos_y, -layer_width*1, layer_font_size+layer_spacing)
                else:
                    print("ERROR: invalid direction specification")
                node.bounding_box = bounding_box
                node.font_size = layer_font_size
                if (leaf_grid_offset_x==0 and leaf_grid_offset_y < max_num_stacked_leafs):
                    cur_leaf_node+=1
            else:
                # Draw inner nodes
                pos_y = layer_offset_y + cur_node*(layer_font_size+layer_spacing)
                prev_layer_max_y = pos_y
                if (direction == "right"):
                    bounding_box = (layer_pos_x, pos_y, layer_width, layer_font_size+layer_spacing)
                elif (direction == "left"):
                    bounding_box = (-layer_pos_x, pos_y, -layer_width, layer_font_size+layer_spacing)
                else:
                    print("ERROR: invalid direction specification")
                node.bounding_box = bounding_box
                node.font_size = layer_font_size
            cur_node+=1
        if (cur_depth == max_depth):
            # Vertically center leaf nodes to align left and right tree in the middle
            if (cur_sibling < max_num_stacked_leafs):
                base_offset = base_offset + (cur_sibling)*(layer_font_size+layer_spacing)
            else:
                base_offset = base_offset + (max_num_stacked_leafs)*(layer_font_size+layer_spacing)
            new_offset = (total_height-base_offset)/2
            for node in node_group:
                node.bounding_box = (node.bounding_box[0], node.bounding_box[1]+new_offset, node.bounding_box[2], node.bounding_box[3])
            prev_layer_min_y = new_offset
            prev_layer_max_y = new_offset + base_offset
        cur_depth-=1


# Calculate for each node the bounding box of its direct children
def calc_bounding_boxes(root, direction):
    for node in PreOrderIter(root):
        if (direction == "right"):
            pos_x = total_width
        elif (direction == "left"):
            pos_x = -total_width
        else:
            print("ERROR: invalid direction specification")
        pos_y = total_height
        width = 0
        height = 0
        for child in node.children:
            if (len(child.children) == 0):
                # process leaf nodes
                if (direction == "right"):
                    pos_x = min(pos_x, child.bounding_box[0])
                    pos_y = min(pos_y, child.bounding_box[1])
                    width = max(width, (child.bounding_box[0]+child.bounding_box[2])-pos_x)
                    if (len(child.siblings)+1 <= max_num_stacked_leafs):
                        height = (len(child.siblings)+1)*child.bounding_box[3]
                    else:
                        height = max_num_stacked_leafs*child.bounding_box[3]
                elif (direction == "left"):
                    pos_x = max(pos_x, child.bounding_box[0])
                    pos_y = min(pos_y, child.bounding_box[1])
                    width = min(width, (child.bounding_box[0]+child.bounding_box[2])-pos_x)
                    if (len(child.siblings)+1 <= max_num_stacked_leafs):
                        height = (len(child.siblings)+1)*child.bounding_box[3]
                    else:
                        height = max_num_stacked_leafs*child.bounding_box[3]
                else:
                    print("ERROR: invalid direction specification")
            else:
                # process inner nodes
                if (direction == "right"):
                    pos_x = min(pos_x, child.bounding_box[0])
                    pos_y = min(pos_y, child.bounding_box[1])
                    width = max(width, child.bounding_box[2])
                    height += child.bounding_box[3]
                elif (direction == "left"):
                    pos_x = max(pos_x, child.bounding_box[0])
                    pos_y = min(pos_y, child.bounding_box[1])
                    width = min(width, child.bounding_box[2])
                    height += child.bounding_box[3]
                else:
                    print("ERROR: invalid direction specification")
        node.bounding_box_children = (pos_x, pos_y, width, height)


# Returns path data for a rounded box
def rounded_box_path(pos_x, pos_y, width, height, curve_radius):
    if (width < 0):
        pos_x = pos_x + width
        width = -width
    if (height < 0):
        pos_y = pos_y + height
        height = -height
    path_data = "M "+str(pos_x+curve_radius)+","+str(pos_y)
    path_data +="L "+str(pos_x+width-curve_radius)+","+str(pos_y)+" "
    path_data+= "A "+str(curve_radius)+","+str(curve_radius)+" 0 0 1 "+str(pos_x+width)+","+str(pos_y+curve_radius)+" "
    path_data+= "L "+str(pos_x+width)+","+str(pos_y+height-curve_radius)+" "
    path_data+= "A "+str(curve_radius)+","+str(curve_radius)+" 0 0 1 "+str(pos_x+width-curve_radius)+","+str(pos_y+height)+" "
    path_data+= "L "+str(pos_x+curve_radius)+","+str(pos_y+height)+" "
    path_data+= "A "+str(curve_radius)+","+str(curve_radius)+" 0 0 1 "+str(pos_x)+","+str(pos_y+height-curve_radius)+" "
    path_data+= "L "+str(pos_x)+","+str(pos_y+curve_radius)+" "
    path_data+= "A "+str(curve_radius)+","+str(curve_radius)+" 0 0 1 "+str(pos_x+curve_radius)+","+str(pos_y)+" "
    return path_data


def rounded_box_path_multimodel(pos_x, pos_y, width, height, curve_radius, vertical_pos, curve_radius_outside):
    if (width < 0):
        pos_x = pos_x + width
        width = -width
    if (height < 0):
        pos_y = pos_y + height
        height = -height
    curve_radius_top = curve_radius
    curve_radius_bottom = curve_radius
    if vertical_pos == 'top':
        curve_radius_top = curve_radius_outside
    elif vertical_pos == 'bottom':
        curve_radius_bottom = curve_radius_outside
    #Changes made to fix bugs at leaf node bar borders
    if width < layer_width*0.9:
        if pos_x >0:
            path_data = "M "+str(pos_x+curve_radius_top)+","+str(pos_y)
            path_data +="L "+str(pos_x+width-curve_radius_top)+","+str(pos_y)+" "
            path_data+= "A "+str(0)+","+str(0)+" 0 0 1 "+str(pos_x+width)+","+str(pos_y+0)+" "
            path_data+= "L "+str(pos_x+width)+","+str(pos_y+height-0)+" "
            path_data+= "A "+str(0)+","+str(0)+" 0 0 1 "+str(pos_x+width-0)+","+str(pos_y+height)+" "
            path_data+= "L "+str(pos_x+curve_radius_bottom)+","+str(pos_y+height)+" "
            path_data+= "A "+str(curve_radius_bottom)+","+str(curve_radius_bottom)+" 0 0 1 "+str(pos_x)+","+str(pos_y+height-curve_radius_bottom)+" "
            path_data+= "L "+str(pos_x)+","+str(pos_y+curve_radius_top)+" "
            path_data+= "A "+str(curve_radius_top)+","+str(curve_radius_top)+" 0 0 1 "+str(pos_x+curve_radius_top)+","+str(pos_y)+" "
        else:
            path_data = "M "+str(pos_x+0)+","+str(pos_y)
            path_data +="L "+str(pos_x+width-curve_radius_top)+","+str(pos_y)+" "
            path_data+= "A "+str(curve_radius_top)+","+str(curve_radius_top)+" 0 0 1 "+str(pos_x+width)+","+str(pos_y+curve_radius_top)+" "
            path_data+= "L "+str(pos_x+width)+","+str(pos_y+height-curve_radius_bottom)+" "
            path_data+= "A "+str(curve_radius_bottom)+","+str(curve_radius_bottom)+" 0 0 1 "+str(pos_x+width-curve_radius_bottom)+","+str(pos_y+height)+" "
            path_data+= "L "+str(pos_x+0)+","+str(pos_y+height)+" "
            path_data+= "A "+str(0)+","+str(0)+" 0 0 1 "+str(pos_x)+","+str(pos_y+height-0)+" "
            path_data+= "L "+str(pos_x)+","+str(pos_y+0)+" "
            path_data+= "A "+str(0)+","+str(0)+" 0 0 1 "+str(pos_x+0)+","+str(pos_y)+" "
    else:
        path_data = "M "+str(pos_x+curve_radius_top)+","+str(pos_y)
        path_data +="L "+str(pos_x+width-curve_radius_top)+","+str(pos_y)+" "
        path_data+= "A "+str(curve_radius_top)+","+str(curve_radius_top)+" 0 0 1 "+str(pos_x+width)+","+str(pos_y+curve_radius_top)+" "
        path_data+= "L "+str(pos_x+width)+","+str(pos_y+height-curve_radius_bottom)+" "
        path_data+= "A "+str(curve_radius_bottom)+","+str(curve_radius_bottom)+" 0 0 1 "+str(pos_x+width-curve_radius_bottom)+","+str(pos_y+height)+" "
        path_data+= "L "+str(pos_x+curve_radius_bottom)+","+str(pos_y+height)+" "
        path_data+= "A "+str(curve_radius_bottom)+","+str(curve_radius_bottom)+" 0 0 1 "+str(pos_x)+","+str(pos_y+height-curve_radius_bottom)+" "
        path_data+= "L "+str(pos_x)+","+str(pos_y+curve_radius_top)+" "
        path_data+= "A "+str(curve_radius_top)+","+str(curve_radius_top)+" 0 0 1 "+str(pos_x+curve_radius_top)+","+str(pos_y)+" "
    return path_data


# Returns path data for a subtree
def subtree_path(direction, start_x, start_y1, start_y2, end_x, end_y1, end_y2, curve_radius, smoothness):
    assert 0.0 <= smoothness <= 1.0, "smoothness must be in[0.0,1.0]"
    if (direction == "left"):
        sign = -1
        arc_param = 0
    elif (direction == "right"):
        sign = 1
        arc_param = 1
    else:
        print("ERROR: invalid direction specification")
    inner_width = ((sign*end_x-sign*start_x)-2*curve_radius)
    path_data = "M"+str(start_x)+","+str(start_y1+curve_radius)
    path_data+= "A"+str(curve_radius)+","+str(curve_radius)+" 0 0 "+str(arc_param)+" "+str(start_x+sign*curve_radius)+","+str(start_y1)
    path_data+= "L"+str(start_x+(sign*(1-smoothness)*inner_width))+","+str(start_y1)
    #path_data+= "C"+str(start_x+(sign*(1-smoothness)*inner_width)+(sign*smoothness*inner_width*0.2))+" "+str(start_y1)+", " +str(end_x-(sign*smoothness*inner_width*0.5))+" "+str(end_y1)+", " +str(end_x)+" "+str(end_y1)
    path_data+= "C"+str(start_x+(sign*(1-smoothness)*inner_width)+(sign*smoothness*inner_width*0.2))+","+str(start_y1)+" "+str(end_x-(sign*smoothness*inner_width*0.5))+","+str(end_y1)+" "+str(end_x)+","+str(end_y1)
    path_data+= "L"+str(end_x)+","+str(end_y2)
    #path_data+= "C"+str(end_x-(sign*smoothness*inner_width*0.5))+" "+str(end_y2)+", " +str(start_x+(sign*(1-smoothness)*inner_width)+(sign*smoothness*inner_width*0.2))+" "+str(start_y2)+", " +str(start_x+(sign*(1-smoothness)*inner_width))+" "+str(start_y2)
    path_data+= "C"+str(end_x-(sign*smoothness*inner_width*0.5))+","+str(end_y2)+" "+str(start_x+(sign*(1-smoothness)*inner_width)+(sign*smoothness*inner_width*0.2))+","+str(start_y2)+" "+str(start_x+(sign*(1-smoothness)*inner_width))+","+str(start_y2)
    path_data+= "L"+str(start_x+sign*curve_radius)+","+str(start_y2)
    path_data+= "A"+str(curve_radius)+","+str(curve_radius)+" 0 0 "+str(arc_param)+" "+str(start_x)+","+str(start_y2-curve_radius)
    path_data+= "L"+str(start_x)+","+str(start_y1+curve_radius)
    return path_data


# Returns path data for a subtree
def subtree_path_multimodel(direction, start_x, start_y1, start_y2, end_x, end_y1, end_y2, curve_radius, smoothness, vertical_pos, curve_radius_outside):
    assert 0.0 <= smoothness <= 1.0, "smoothness must be in[0.0,1.0]"
    if (direction == "left"):
        sign = -1
        arc_param = 0
    elif (direction == "right"):
        sign = 1
        arc_param = 1
    else:
        print("ERROR: invalid direction specification")
    curve_radius_top = curve_radius
    curve_radius_bottom = curve_radius
    if vertical_pos == 'top':
        curve_radius_top = curve_radius_outside
    elif vertical_pos == 'bottom':
        curve_radius_bottom = curve_radius_outside
    inner_width = ((sign*end_x-sign*start_x)-2*curve_radius)
    path_data = "M"+str(start_x)+","+str(start_y1+curve_radius_top)
    path_data+= "A"+str(curve_radius_top)+","+str(curve_radius_top)+" 0 0 "+str(arc_param)+" "+str(start_x+sign*curve_radius_top)+","+str(start_y1)
    path_data+= "L"+str(start_x+(sign*(1-smoothness)*inner_width))+","+str(start_y1)
    #path_data+= "C"+str(start_x+(sign*(1-smoothness)*inner_width)+(sign*smoothness*inner_width*0.2))+" "+str(start_y1)+", " +str(end_x-(sign*smoothness*inner_width*0.5))+" "+str(end_y1)+", " +str(end_x)+" "+str(end_y1)
    path_data+= "C"+str(start_x+(sign*(1-smoothness)*inner_width)+(sign*smoothness*inner_width*0.2))+","+str(start_y1)+" "+str(end_x-(sign*smoothness*inner_width*0.5))+","+str(end_y1)+" " +str(end_x)+","+str(end_y1)
    path_data+= "L"+str(end_x)+","+str(end_y2)
    #path_data+= "C"+str(end_x-(sign*smoothness*inner_width*0.5))+" "+str(end_y2)+", " +str(start_x+(sign*(1-smoothness)*inner_width)+(sign*smoothness*inner_width*0.2))+" "+str(start_y2)+", " +str(start_x+(sign*(1-smoothness)*inner_width))+" "+str(start_y2)
    path_data+= "C"+str(end_x-(sign*smoothness*inner_width*0.5))+","+str(end_y2)+" "+str(start_x+(sign*(1-smoothness)*inner_width)+(sign*smoothness*inner_width*0.2))+","+str(start_y2)+" "+str(start_x+(sign*(1-smoothness)*inner_width))+","+str(start_y2)
    path_data+= "L"+str(start_x+sign*curve_radius_bottom)+","+str(start_y2)
    path_data+= "A"+str(curve_radius_bottom)+","+str(curve_radius_bottom)+" 0 0 "+str(arc_param)+" "+str(start_x)+","+str(start_y2-curve_radius_bottom)
    path_data+= "L"+str(start_x)+","+str(start_y1+curve_radius_bottom)
    return path_data


# Returns path data for a subtree at second lowest level, which embedds the leaf nodes
def subtree_path_embedding(direction, start_x, start_y1, start_y2, end_x1, end_x2, end_y1, end_y2, curve_radius, smoothness):
    assert 0.0 <= smoothness <= 1.0, "smoothness must be in[0.0,1.0]"
    if (direction == "left"):
        sign = -1
        arc_param = 0
    elif (direction == "right"):
        sign = 1
        arc_param = 1
    else:
        print("ERROR: invalid direction specification")
    inner_width = ((sign*end_x1-sign*start_x)-2*curve_radius)
    path_data = "M"+str(start_x)+","+str(start_y1+curve_radius)
    path_data+= "A"+str(curve_radius)+","+str(curve_radius)+" 0 0 "+str(arc_param)+" "+str(start_x+sign*curve_radius)+","+str(start_y1)
    path_data+= "L"+str(start_x+(sign*(1-smoothness)*inner_width))+","+str(start_y1)
    #path_data+= "C"+str(start_x+(sign*(1-smoothness)*inner_width)+(sign*smoothness*inner_width*0.2))+" "+str(start_y1)+", " +str(end_x1-(sign*smoothness*inner_width*0.5))+" "+str(end_y1)+", " +str(end_x1)+" "+str(end_y1)
    path_data+= "C"+str(start_x+(sign*(1-smoothness)*inner_width)+(sign*smoothness*inner_width*0.2))+","+str(start_y1)+" "+str(end_x1-(sign*smoothness*inner_width*0.5))+","+str(end_y1)+" " +str(end_x1)+","+str(end_y1)
    path_data+= "L"+str(end_x2-sign*curve_radius)+","+str(end_y1)
    path_data+= "A"+str(curve_radius)+","+str(curve_radius)+" 0 0 "+str(arc_param)+" "+str(end_x2)+","+str(end_y1+curve_radius)
    path_data+= "L"+str(end_x2)+","+str(end_y2-curve_radius)
    path_data+= "A"+str(curve_radius)+","+str(curve_radius)+" 0 0 "+str(arc_param)+" "+str(end_x2-sign*curve_radius)+","+str(end_y2)
    path_data+= "L"+str(end_x1)+","+str(end_y2)
    #path_data+= "C"+str(end_x1-(sign*smoothness*inner_width*0.5))+" "+str(end_y2)+", " +str(start_x+(sign*(1-smoothness)*inner_width)+(sign*smoothness*inner_width*0.2))+" "+str(start_y2)+", " +str(start_x+(sign*(1-smoothness)*inner_width))+" "+str(start_y2)
    path_data+= "C"+str(end_x1-(sign*smoothness*inner_width*0.5))+","+str(end_y2)+" "+str(start_x+(sign*(1-smoothness)*inner_width)+(sign*smoothness*inner_width*0.2))+","+str(start_y2)+" "+str(start_x+(sign*(1-smoothness)*inner_width))+","+str(start_y2)
    path_data+= "L"+str(start_x+sign*curve_radius)+","+str(start_y2)
    path_data+= "A"+str(curve_radius)+","+str(curve_radius)+" 0 0 "+str(arc_param)+" "+str(start_x)+","+str(start_y2-curve_radius)
    path_data+= "L"+str(start_x)+","+str(start_y1+curve_radius)
    return path_data


# Returns path data for a subtree centerline to align text to
def subtree_center_line(direction, start_x, start_y1, start_y2, end_x, end_y1, end_y2, curve_radius, smoothness):
    if (direction == "left"):
        inner_width = (-end_x+start_x)-2*curve_radius
        path_data = "M "+str(end_x)+","+str((end_y1+end_y2)/2)
        #path_data+= " C"+str(end_x+(smoothness*inner_width*0.5))+" "+str((end_y1+end_y2)/2)+" , " +str(start_x-((1-smoothness)*inner_width)-(smoothness*inner_width*0.2))+" "+str((start_y1+start_y2)/2)+" , " +str(start_x-((1-smoothness)*inner_width))+" "+str((start_y1+start_y2)/2)
        path_data+= " C "+str(end_x+(smoothness*inner_width*0.5))+","+str((end_y1+end_y2)/2)+" "+str(start_x-((1-smoothness)*inner_width)-(smoothness*inner_width*0.2))+","+str((start_y1+start_y2)/2)+" "+str(start_x-((1-smoothness)*inner_width))+","+str((start_y1+start_y2)/2)
        path_data+= " L "+str(start_x-curve_radius)+","+str((start_y1+start_y2)/2)
    elif (direction == "right"):
        inner_width = (end_x-start_x)-2*curve_radius
        path_data = "M "+str(start_x+curve_radius)+" , "+str((start_y1+start_y2)/2)
        path_data+= " L "+str(start_x+((1-smoothness)*inner_width))+" , "+str((start_y1+start_y2)/2)
        #path_data+= " C"+str(start_x+((1-smoothness)*inner_width)+(smoothness*inner_width*0.2))+" "+str((start_y1+start_y2)/2)+" , " +str(end_x-(smoothness*inner_width*0.5))+" "+str((end_y1+end_y2)/2)+" , " +str(end_x)+" "+str((end_y1+end_y2)/2)
        path_data+= " C "+str(start_x+((1-smoothness)*inner_width)+(smoothness*inner_width*0.2))+","+str((start_y1+start_y2)/2)+" "+str(end_x-(smoothness*inner_width*0.5))+","+str((end_y1+end_y2)/2)+" "+str(end_x)+","+str((end_y1+end_y2)/2)
    else:
        print("ERROR: invalid direction specification")
    return path_data


def __compute_voronoi_regions(points_per_site, points):
    """ Generates samples for Lloyd relaxation, following the density distribution of the input
    data.

    Args:
        input_points (List[float]): 1D-Array of values.: 1D-Array of values.
        scaling (float): Aspect ratio scaling, computed by using our adaptive height method.
        centralized (bool): Whether the samples should follow the density of the points in height
            as well to generate the samples in a violin plot like fashion.
        num_samples (int): Number of samples for the pdf. Defaults to 8192.
        bandwidth (int): Bandwith for the kernel density estimation. This value is passed
            to `kde.factor`, used by `scipy.stats.gaussian_kde`. Defaults to 0.2.

    Returns:
        List[List[float]] 2D-Array, of samples to be used for Lloyd relaxation.
    """
    diff = tf.math.subtract(points_per_site, points)
    dist = tf.norm(diff, ord='euclidean', axis=2, keepdims=True)
    return tf.math.argmin(dist, axis=1)


def __compute_centroids(voronoi, points_index_tensor, sites_per_point, ones, zeros, num_points,
                        num_sites, num_dims_per_points):
    mask = tf.math.equal(points_index_tensor, voronoi)
    mask_tiles = tf.tile(mask, [1, 1, num_dims_per_points])
    sites_mask = tf.reshape(mask_tiles, [num_points, num_sites, num_dims_per_points])
    masked_sites_sum = tf.squeeze(tf.math.reduce_sum(tf.where(sites_mask, sites_per_point, zeros),
                                                     axis=1, keepdims=True), axis=1)
    counts = tf.split(tf.squeeze(tf.reduce_sum(tf.where(sites_mask, ones, zeros),
                                               axis=1, keepdims=True), axis=1),
                      num_dims_per_points, axis=1)[0]
    return tf.math.divide(masked_sites_sum, counts)


def construct_polygon(path):
    polygon_points = []
    for i in range(len(path)):
        segment = path[i]
        if isinstance(segment, svgpathtools.path.Line):
            # add line between start and end point
            polygon_points.append((segment.start.real, segment.start.imag))
            polygon_points.append((segment.end.real, segment.end.imag))
        elif isinstance(segment, svgpathtools.path.CubicBezier):
            # approximate Bezier curve segment with 10 points
            bezier_path = svgpathtools.Path(segment)
            NUM_SAMPLES = 10
            for i in range(NUM_SAMPLES):
                bezier_point = bezier_path.point(i/(NUM_SAMPLES-1))
                polygon_points.append((bezier_point.real, bezier_point.imag))
        elif isinstance(segment, svgpathtools.path.Arc):
            # approximate by line between start and end point
            polygon_points.append((segment.start.real, segment.start.imag))
            polygon_points.append((segment.end.real, segment.end.imag))
    polygon = Polygon(polygon_points)
    return polygon


def sample_points_in_polygon(polygon, n_points, qnapoint_offset):
    '''
    samples n_points in the polygon.
    '''
    sampled_points = []
    bounds = polygon.bounds
    x1_bound = bounds[0]
    x2_bound = bounds[2]
    if abs(bounds[0]) > abs(bounds[2]):
        x2_bound += qnapoint_offset
    else:
        x1_bound += qnapoint_offset
    while len(sampled_points) < n_points:
        point = (random.uniform(x1_bound, x2_bound), random.uniform(bounds[1], bounds[3]))
        if polygon.contains(Point(point)):
            sampled_points.append(point)
    return sampled_points


def draw_qa_samples(drawing, path_data, point_color, n_points, num_questions, circle_radius=2.3, qnapoint_offset=0):
    '''
    Draws random points and relaxes them in a blue noise pattern. 
    The path_data is transformed into a polygon. 
    Sample and support points are sampled inside the polygon modified by qna_offset (horizontal threshold).
    Points are relaxed and added to the drawing.

    :param drawing: svgwrite Drawing object
    :param path_data: svg path data 
    :param point_color: color of the plotted points 
    :param n_points: int, number of the support samples for lloyd relaxation 
    :param num_questions: int, number of sample points
    :param circle_radius: float, radius of the plotted points
    :param qnapoint_offset: float, horizontal offset of the points from the center  
    '''
    path = svgpathtools.parse_path(path_data)
    polygon = construct_polygon(path)
    sampled_points = sample_points_in_polygon(polygon, n_points, qnapoint_offset)
    x = sample_points_in_polygon(polygon, num_questions, qnapoint_offset)
    max_iterations=2000
    bn = blue_noise_single_class(x, sampled_points, polygon, max_iterations=max_iterations)
    bn = [(x,y) for x,y in bn.tolist()]
    for point in bn:
        drawing.add(drawing.circle(point, r=circle_radius, fill=point_color, stroke='none'))


def norm(p, bounds):
    scale = (bounds[2]-bounds[0])
    p[:,0] = (p[:,0] - bounds[0]) / scale
    p[:,1] = (p[:,1] - bounds[1]) / scale
    return p


def denorm(p, bounds):
    scale = (bounds[2]-bounds[0])
    p[:,0] = p[:,0] * scale + bounds[0]
    p[:,1] = p[:,1] * scale + bounds[1]
    return p


def blue_noise_single_class(input_points, random_points, polygon, max_iterations=200):
    bounds = polygon.bounds
    #normalize the data domain
    points = norm(np.array(input_points), bounds)
    sites = norm(np.array(random_points), bounds)
    aspect_ratio_scaling = (bounds[3]-bounds[1])/(bounds[2]-bounds[0])

    #prepare data for relaxation 
    points = tf.convert_to_tensor(points, dtype=tf.float32)
    sites = tf.convert_to_tensor(sites, dtype=tf.float32)

    num_points = tf.shape(points).numpy()[0]
    num_dims_per_points = tf.shape(points).numpy()[1]
    num_sites = tf.shape(sites).numpy()[0]
    points_per_site = tf.reshape(tf.tile(sites, [1, num_points]),
                                 [num_sites, num_points, num_dims_per_points])
    sites_per_point = tf.reshape(tf.tile(sites, [num_points, 1]),
                                 [num_points, num_sites, num_dims_per_points])

    zeros = tf.zeros(sites_per_point.shape)
    ones = tf.ones(sites_per_point.shape)

    points_index_tensor = [x for x in range(num_points)]
    points_index_tensor = tf.expand_dims(tf.convert_to_tensor(points_index_tensor, dtype=tf.int64),
                                         axis=1)
    points_index_tensor = tf.expand_dims(tf.tile(points_index_tensor, [1, num_sites]),
                                         axis=2)

    # Lloyd iterations
    for i in range(max_iterations):

        voronoi = __compute_voronoi_regions(points_per_site, points)

        centroids = __compute_centroids(voronoi, points_index_tensor, sites_per_point, ones, zeros,
                                        num_points, num_sites, num_dims_per_points)

        x, y = tf.split(centroids, num_or_size_splits=2, axis=1)
        # if a point is running out of the domain, put the point back in the center of the plot
        #check if any centroid is nan (should only happen because a query point was not the closest to any random point in sites)
        isOutsidePlotx = tf.math.is_nan(x)
        isOutsidePloty = tf.math.is_nan(y)
        correctedX = tf.where(isOutsidePlotx,
                              0.5, x)

        correctedY = tf.where(isOutsidePloty,
                              0.5*aspect_ratio_scaling, y)
        # put the relax point, back to it's original data-dimension.
        points = tf.squeeze(tf.stack([correctedX, correctedY], axis=1), axis=2)

    return denorm(points.numpy(), bounds)


def get_class_color(node, model_id, color_offset, shade=-1):
    def get_class_id(node):
        if (node.depth == 1):
            return node.parent.children.index(node)
        else:
            return get_class_id(node.parent)

    if (color_mode == "data_centric"):
        color_name = list(color_palette.get_color_dict)[color_offset+get_class_id(node)]
        if (shade == -1):
            shade = 10+model_id*10
    elif (color_mode == "model_centric"):
        color_name = list(color_palette.get_color_dict)[2*model_id]
        if (shade == -1):
            shade = 20
    else:
        print("ERROR: invalid color_mode specification")
    return color_palette.get(color_name, shade=shade)


def modelbar_to_svg(drawing, direction, path_data, start_x, start_y1, start_y2, end_x, end_y1, end_y2, accuracy, fill_color):
    # Clip the outline path based on the percentage of correct answers to obtain something similar to a bar chart
    unique_clip_path_id = str(uuid.uuid4())
    # Define clipping path as a rectangle
    clip_path = drawing.defs.add(drawing.clipPath(id=str(unique_clip_path_id)))
    if (direction == "left"):
        percentage_width = (-end_x+start_x)*accuracy
        rect = svgwrite.shapes.Rect(insert=(start_x-percentage_width, min(start_y1, end_y1)), size=(percentage_width, max(start_y2,end_y2)-min(start_y1,end_y1)), stroke='red', fill='none')
    elif (direction == "right"):
        # Clip the outline path based on the percentage of correct answers to obtain something similar to a bar chart
        percentage_width = (end_x-start_x)*accuracy
        rect = svgwrite.shapes.Rect(insert=(start_x, min(start_y1, end_y1)), size=(percentage_width, max(start_y2,end_y2)-min(start_y1,end_y1)), stroke='red', fill='none')
    else:
        print("ERROR: invalid direction specification")
    clip_path.add(rect)
    drawing.add(drawing.path(d=path_data, stroke='none', fill=fill_color, clip_path='url(#'+str(unique_clip_path_id)+')'))


def clip_sample_path(drawing, direction, path_data, start_x, start_y1, start_y2, end_x, end_y1, end_y2):
    # Clip the outline path to not overdraw points and icons
    unique_clip_path_id = str(uuid.uuid4())
    # Define clipping path as a rectangle
    clip_path = drawing.defs.add(drawing.clipPath(id=str(unique_clip_path_id)))
    if (direction == "left"):
        rect = svgwrite.shapes.Rect(insert=(start_x, min(start_y1, end_y1)), size=(abs(start_x-end_x), max(start_y2,end_y2)-min(start_y1,end_y1)), stroke='red', fill='none')
    elif (direction == "right"):
        # Clip the outline path based on the percentage of correct answers to obtain something similar to a bar chart
        rect = svgwrite.shapes.Rect(insert=(start_x, min(start_y1, end_y1)), size=(abs(start_x-end_x), max(start_y2,end_y2)-min(start_y1,end_y1)), stroke='red', fill='none')
    else:
        print("ERROR: invalid direction specification")
    clip_path.add(rect)
    return  svgwrite.path.Path(d=path_data, stroke='none', fill='none', clip_path='url(#'+str(unique_clip_path_id)+')')


def display_accuracy(drawing, direction, end_x, end_y1, end_y2, accuracy, text_color):
    font_size = 18
    if (direction == "left"):
        accuracy_label = drawing.g(style="font-size:"+str(font_size)+";font-family:Roboto Condensed;fill:"+str(text_color))
        accuracy_label.add(drawing.text(("%.2f" % (accuracy*100))+"%", text_anchor="middle", alignment_baseline="baseline", transform="translate("+str(end_x+font_size)+","+str(end_y1+(end_y2-end_y1)/2)+"),rotate(90)"))
        drawing.add(accuracy_label)
    elif (direction == "right"):
        accuracy_label = drawing.g(style="font-size:"+str(font_size)+";font-family:Roboto Condensed;fill:"+str(text_color))
        accuracy_label.add(drawing.text(("%.2f" % (accuracy*100))+"%", text_anchor="middle", alignment_baseline="baseline", transform="translate("+str(end_x-font_size)+","+str(end_y1+(end_y2-end_y1)/2)+"),rotate(-90)"))
        drawing.add(accuracy_label)
    else:
        print("ERROR: invalid direction specification")


def draw_tachometer(drawing, pos_x, pos_y, radius, speed, icon_color, scale):

    # Compute Cartesian coordinates for tachometer needle
    def polar_to_cartesian(pos_x, pos_y, radius, angle_in_degrees):
        angle_in_radians = (angle_in_degrees - 90) * math.pi / 180.0
        x = pos_x + (radius * math.cos(angle_in_radians))
        y = pos_y + (radius * math.sin(angle_in_radians))
        return {'x': x, 'y': y}

    # Define angle of tachometer's display area, and compute needle orientation
    start_angle = 240
    end_angle = 120
    needle_angle = start_angle + ((360-start_angle)+end_angle)*speed/scale
    if needle_angle > 360:
        needle_angle = needle_angle - 360
    
    # Convert angles to cartesioan coordinates to be fed into SVG path
    start = polar_to_cartesian(pos_x, pos_y, radius, end_angle)
    end = polar_to_cartesian(pos_x, pos_y, radius, start_angle)
    needle_endpoint = polar_to_cartesian(pos_x, pos_y, radius, needle_angle)

    # Translate vertically to center tachometer along pos_y
    offset_y = max(start['y']-pos_y, end['y']-pos_y)/2
    start['y'] = start['y'] + offset_y
    end['y'] = end['y'] + offset_y
    needle_endpoint['y'] = needle_endpoint['y'] + offset_y

    path_data = "M"+str(start['x'])+","+str(start['y'])
    path_data+= "A"+str(radius)+","+str(radius)+" 0 1 0 "+str(end['x'])+","+str(end['y'])
    drawing.add(drawing.path(d=path_data, stroke=icon_color, stroke_width='1.3', stroke_linecap='round', fill='none'))

    needle_path = "M "+str(pos_x)+" "+str(pos_y+offset_y)
    needle_path += "L "+str(("%.2f" % (needle_endpoint['x'])))+" "+str(("%.2f" % (needle_endpoint['y'])))
    drawing.add(drawing.path(d=needle_path, stroke=icon_color, stroke_width='1.3', stroke_linecap='round', fill='none'))

    drawing.add(drawing.circle(center=(pos_x, pos_y+offset_y), r=radius*0.2, fill=icon_color))


def draw_hallucination_icon(drawing, pos_x, pos_y, radius, hallucination, icon_color):
    # Clip the outline path based on the percentage of correct answers to obtain something similar to a bar chart
    unique_clip_path_id = str(uuid.uuid4())
    clip_path = drawing.defs.add(drawing.clipPath(id=str(unique_clip_path_id)))
    rect = svgwrite.shapes.Rect(insert=(pos_x-radius, pos_y+radius-hallucination*(2*radius)), size=(2*radius, hallucination*(2*radius)), stroke='red', fill='none')
    clip_path.add(rect)
    drawing.add(drawing.circle(center=(pos_x, pos_y), r=radius*1.0, fill=color_palette.get('red', shade=60), stroke='none', clip_path='url(#'+str(unique_clip_path_id)+')'))
    
    drawing.add(drawing.circle(center=(pos_x, pos_y), r=radius*1.0, fill='none', stroke=icon_color))
    drawing.add(drawing.circle(center=(pos_x, pos_y), r=radius*0.8, fill='none', stroke=icon_color))
    drawing.add(drawing.circle(center=(pos_x, pos_y), r=radius*0.6, fill='none', stroke=icon_color))
    drawing.add(drawing.circle(center=(pos_x, pos_y), r=radius*0.4, fill='none', stroke=icon_color))
    drawing.add(drawing.circle(center=(pos_x-0.1*radius, pos_y-0.1*radius), r=radius*0.05, fill=icon_color, stroke='none'))
    drawing.add(drawing.circle(center=(pos_x+0.1*radius, pos_y-0.1*radius), r=radius*0.05, fill=icon_color, stroke='none'))
    drawing.add(drawing.line(start=(pos_x-0.15*radius, pos_y+0.1*radius), end=(pos_x+0.15*radius, pos_y+0.1*radius), stroke=icon_color))


def display_response_time(drawing, direction, pos_x, pos_y, radius, response_time, icon_color, scale):
    if (direction == "left"):
        draw_tachometer(drawing, pos_x, pos_y, radius, response_time, icon_color, scale)
    elif (direction == "right"):
        draw_tachometer(drawing, pos_x, pos_y, radius, response_time, icon_color, scale)
    else:
        print("ERROR: invalid direction specification")


def display_difficulty(drawing, direction, pos_x, pos_y, icon_height, icon_color, difficulty_aggregate):
    difficulty = sum([s[0]*(i+1) for i,s in enumerate(difficulty_aggregate)])/sum([s[0] for s in difficulty_aggregate])
    if (direction == "left"):
        for i in range(0,5):
            #path_data_outline = rounded_box_path(pos_x-(i+1)*(difficulty_bar_width+difficulty_bar_gap), start_y1+0.1*(start_y2-start_y1), difficulty_bar_width, 0.8*(start_y2-start_y1), 1)
            path_data_outline = rounded_box_path(pos_x-(i+1)*(difficulty_bar_width+difficulty_bar_gap), pos_y-0.4*icon_height, difficulty_bar_width, 0.8*icon_height, 1)
            if i+1 <= difficulty:
                drawing.add(drawing.path(d=path_data_outline, stroke_width='0.5', stroke=icon_color, fill=icon_color))
            else:
                drawing.add(drawing.path(d=path_data_outline, stroke_width='0.5', stroke=icon_color, fill='none'))
    elif (direction == "right"):
        for i in range(0,5):
            #path_data_outline = rounded_box_path(pos_x+i*(difficulty_bar_width+difficulty_bar_gap), start_y1+0.1*(start_y2-start_y1), difficulty_bar_width, 0.8*(start_y2-start_y1), 1)
            path_data_outline = rounded_box_path(pos_x+i*(difficulty_bar_width+difficulty_bar_gap), pos_y-0.4*icon_height, difficulty_bar_width, 0.8*icon_height, 1)
            if i+1 <= difficulty:
                drawing.add(drawing.path(d=path_data_outline, stroke_width='0.5', stroke=icon_color, fill=icon_color))
            else:
                drawing.add(drawing.path(d=path_data_outline, stroke_width='0.5', stroke=icon_color, fill='none'))
    else:
        print("ERROR: invalid direction specification")


def display_hallucination(drawing, direction, pos_x, offset_x, pos_y, radius, hallucination_aggregate, icon_color):
    if (direction == "left"):
        draw_hallucination_icon(drawing, pos_x-offset_x, pos_y, radius, hallucination_aggregate, icon_color)
    elif (direction == "right"):
        draw_hallucination_icon(drawing, pos_x+offset_x, pos_y, radius, hallucination_aggregate, icon_color)
    else:
        print("ERROR: invalid direction specification")


# Return a color with maximal contrast with respect to color1 and color2
def get_max_contrast_color(color1, color2):
    # Convert input colors to RGB tuples
    rgb_color1 = sc.utils.hex_to_rgb(color1)
    rgb_color2 = sc.utils.hex_to_rgb(color2)
    
    # Convert RGB tuples to HSL tuples
    hls_color1 = colorsys.rgb_to_hls(*rgb_color1)
    hls_color2 = colorsys.rgb_to_hls(*rgb_color2)
    
    # Calculate the average hue and difference in lightness between the two colors
    hue = (hls_color1[0] + hls_color2[0]) / 2
    lightness_diff = abs(hls_color1[1] - hls_color2[1])
    
    # Determine whether to increase or decrease the lightness of the output color
    if hls_color1[1] < hls_color2[1]:
        new_lightness = min(1.0, hls_color1[1] + (lightness_diff / 2))
    else:
        new_lightness = max(0.0, hls_color1[1] - (lightness_diff / 2))
        
    # Convert the HSL tuple back to an RGB tuple
    new_rgb = colorsys.hls_to_rgb(hue, new_lightness, 1)
    
    # Convert the RGB tuple to hex color string
    return sc.utils.rgb_to_hex(new_rgb[0], new_rgb[1], new_rgb[2])


def cubic_bezier_y(x, x0, y0, x1, y1, x2, y2, x3, y3, t_init):
    t = t_init
    while True:
        xt = (1 - t)**3*x0 + 3*(1-t)**2*t*x1 + 3*(1-t)*t**2*x2 + t**3*x3
        if x - 1 <= xt <= x + 1:
            y = (1 - t)**3*y0 + 3*(1-t)**2*t*y1 + 3*(1-t)*t**2*y2 + t**3*y3
            return y
        elif xt < x:
            t = min(t + 0.001, 1)  # limit t to a maximum of 1
        else:
            t = max(t - 0.001, 0)  # limit t to a minimum of 0


def cubic_bezier_y_left(x, x0, y0, x1, y1, x2, y2, x3, y3, t_init):
    t = t_init
    while True:
        xt = (1 - t)**3*x0 + 3*(1-t)**2*t*x1 + 3*(1-t)*t**2*x2 + t**3*x3
        if x - 1 <= xt <= x + 1:
            y = (1 - t)**3*y0 + 3*(1-t)**2*t*y1 + 3*(1-t)*t**2*y2 + t**3*y3
            return y
        elif xt < x:
            t = min(t + 0.001, 1)  # limit t to a maximum of 1
        else:
            t = max(t - 0.001, 0)  # limit t to a minimum of 0


def get_branch(node):
    branch = []
    branch.append(node.name)
    done = False
    while not done:
        try:
            branch.append(node.parent.name)
            node = node.parent
        except:
            done = True
    return branch

def get_coordinates(path_data, direction, t):
    if direction == "left":
        t = 1.0 - t

    # Split the path data string into a list of path commands and remove withespace as well as commas
    commands = re.split(',| ', path_data)
    commands = [x for x in commands if x != ',']
    commands = [x for x in commands if x != '']

    start_x = float(commands[1])
    end_x = float(commands[len(commands)-2])
    if (direction == "right"):
        extend_x = end_x - start_x
    elif (direction == "left"):
        extend_x = abs(start_x) - abs(end_x)

    pos_x = start_x + (t * extend_x)
    pos_y = None

    i = 0
    while i < len(commands):
        command = commands[i]
        if command == "M":
            x0, y0 = float(commands[i+1]), float(commands[i+2])
            if (x0 == pos_x):
                pos_y = y0
            i = i + 3

        elif command == "L":
            x0, y0 = float(commands[i-2]), float(commands[i-1])
            x1, y1 = float(commands[i+1]), float(commands[i+2])
            if (min(x0, x1) <= pos_x <= max(x0, x1)):
                t_local = (pos_x - min(x0, x1)) / (max(x0, x1) - min(x0, x1))
                pos_y = (1 - t_local) * y0 + t_local * y1
            i = i + 3

        elif command == "C":
            x0, y0 = float(commands[i-2]), float(commands[i-1])
            x1, y1 = float(commands[i+1]), float(commands[i+2])
            x2, y2 = float(commands[i+3]), float(commands[i+4])
            x3, y3 = float(commands[i+5]), float(commands[i+6])
            if (min(x0, x3) <= pos_x <= max(x0, x3)):
                if (direction == "right"):
                    pos_y = cubic_bezier_y(pos_x, x0, y0, x1, y1, x2, y2, x3, y3, 0.5)#t_init)
                elif (direction == "left"):
                    pos_y = cubic_bezier_y_left(-pos_x, -x3, y3, -x2, y2, -x1, y1, -x0, y0, 0.5)
            i = i + 7

    # Return the final x and y coordinates
    return (pos_x, pos_y)


# Generate SVG code for tree node oriented into direction
# Does not consider root node!
def tree_to_svg(drawing, node, direction, layer_width, smoothness, color_offset, draw_qna_samples):
    if (direction == "left"):
        sign = -1
    elif (direction == "right"):
        sign = 1
    else:
        print("ERROR: invalid direction specification")
    num_models = len(node.llm_models)
    #base_color_id = (num_models+1 if num_models==5 else num_models) if color_mode == "model_centric" else -1
    base_color_id = int((num_models-1)/2) if color_mode == "model_centric" else -1
    if color_mode == "model_centric" and hasattr(node, 'parent') and node.parent is not None:
        branch = get_branch(node)[-2]
        dark_grey = root_node.child_names.index(branch)%2==0
        if dark_grey:
            empty_color_gray = color_palette.gray_neutral(shade=30)
        else:
            empty_color_gray = color_palette.gray_neutral(shade=10)
        border_color_gray = color_palette.gray_neutral(shade=40)

    if (node.depth > 0):
        center_line = ""
        curve_radius = node.font_size/2
        if num_models > 1:
            if curve_radius > node.bounding_box[3]/num_models*1.1:     
                curve_radius = node.bounding_box[3]/num_models*1.1
        border_color = get_class_color(node, base_color_id, color_offset, shade=8)
        if (node.depth < max_depth):
            # Draw inner nodes
            empty_color = get_class_color(node, base_color_id, color_offset, shade=3)
            start_x = node.bounding_box[0]
            start_y1 = node.bounding_box[1]
            start_y2 = node.bounding_box[1]+node.bounding_box[3]
            end_x = node.bounding_box_children[0]
            # widen by curve_radius for all but last two layers
            if (node.depth < max_depth-1):
                end_x = end_x+sign*curve_radius
            end_y1 = node.bounding_box_children[1]
            end_y2 = node.bounding_box_children[1]+node.bounding_box_children[3]
            # Compute outline path for subtree
            if (node.depth == max_depth-1):
                # Path for second last layer should incorporate leaf nodes
                path_data = subtree_path_embedding(direction, start_x, start_y1, start_y2, end_x, end_x+node.bounding_box_children[2]+sign*leaf_spacing/2, end_y1-leaf_spacing/2, end_y2+leaf_spacing/2, curve_radius, smoothness)
            else:
                path_data = subtree_path(direction, start_x, start_y1, start_y2, end_x, end_y1, end_y2, curve_radius, smoothness)
            center_line = subtree_center_line(direction, start_x, start_y1, start_y2, end_x, end_y1, end_y2, curve_radius, smoothness)
            drawing.add(drawing.path(d=center_line, stroke='red', fill='none'))
            
            if color_mode == "model_centric":
                drawing.add(drawing.path(d=path_data, stroke='none', fill=empty_color_gray))
            else:
                drawing.add(drawing.path(d=path_data, stroke='none', fill=empty_color))
            for i in range(num_models):
                m_start_y1 = start_y1 + i * ((start_y2-start_y1)/num_models)
                m_start_y2 = start_y1 + (i+1) * ((start_y2-start_y1)/num_models)
                m_end_y1 = end_y1 + i * ((end_y2-end_y1)/num_models)
                m_end_y2 = end_y1 + (i+1) * ((end_y2-end_y1)/num_models)
                if i == 0:
                    vertical_pos = 'top'
                elif i == num_models-1:
                    vertical_pos = 'bottom'
                else:
                    vertical_pos = 'middle'
                model_path_data = subtree_path_multimodel(direction, start_x, m_start_y1, m_start_y2, end_x, m_end_y1, m_end_y2, curve_radius/num_models, smoothness, vertical_pos, curve_radius)
                modelbar_to_svg(drawing, direction, model_path_data, start_x, m_start_y1, m_start_y2, end_x, m_end_y1, m_end_y2, node.accuracy_aggregate[i], get_class_color(node, i, color_offset))

            row_height = (start_y2-start_y1)
            if num_models > 1:
                contrast_color_index = int((num_models-1)/2)
                #contrast_color_index = num_models+1 if num_models==5 else num_models
                contrast_empty_color = empty_color
            else:
                contrast_color_index = 0
                contrast_empty_color = empty_color
            icon_color = get_max_contrast_color(contrast_empty_color, get_class_color(node, contrast_color_index, color_offset))
            icon_offset_t = 0.0
            if node.num_questions > 0:
                difficulty_icon_width = 5*(difficulty_bar_width+difficulty_bar_gap)
                coords = get_coordinates(center_line, direction, icon_offset_t)
                display_difficulty(drawing, direction, coords[0], coords[1], start_y2-start_y1, icon_color, node.difficulty_level_aggregate[0])
            if num_models == 1:
                if node.num_questions > 0:
                    icon_offset_t = 0.095
                    coords = get_coordinates(center_line, direction, icon_offset_t)
                    display_response_time(drawing, direction, coords[0], coords[1], 0.75*(row_height/2), node.average_time_aggregate[0], icon_color, 2*root_node.average_time_aggregate[0])
                if node.depth == 1:
                    icon_offset_t = 0.165
                    coords = get_coordinates(center_line, direction, icon_offset_t)
                    display_hallucination(drawing, direction, coords[0], 0, coords[1], 0.75*(row_height/2), get_hallucination_score(node.difficulty_level_aggregate[0]), icon_color)
                    display_accuracy(drawing, direction, end_x, end_y1, end_y2, node.accuracy_aggregate[0], icon_color)
            if color_mode == "model_centric":
                drawing.add(drawing.path(d=path_data, stroke=border_color_gray, fill='none'))
            else:
                drawing.add(drawing.path(d=path_data, stroke=border_color, fill='none'))
            if draw_qna_samples and node.num_questions > 0:
                qnapoint_offset = sign * (difficulty_icon_width+row_height/1.5 + 0.75*(row_height/2))
                point_canvas_path_data = subtree_path(direction, start_x, start_y1, start_y2, end_x, end_y1, end_y2, curve_radius, smoothness)
                qna_sample_color = get_max_contrast_color(contrast_empty_color, get_class_color(node, contrast_color_index, color_offset))
                if args.qna_sample_quality == 'high':
                    n_points = max(40000, 20*node.num_questions)
                else: 
                    n_points = max(min(max(2000, 10*node.num_questions), 40000), 2*node.num_questions)
                draw_qa_samples(drawing, point_canvas_path_data, qna_sample_color, n_points, node.num_questions, qnapoint_offset=qnapoint_offset)
        else:
            # Draw leaf nodes
            empty_color = get_class_color(node, base_color_id, color_offset, shade=2)
            path_data_outline = rounded_box_path(node.bounding_box[0], node.bounding_box[1], node.bounding_box[2], node.bounding_box[3], curve_radius)
            drawing.add(drawing.path(d=path_data_outline, stroke='none', fill=empty_color))
            if num_models > 1:
                contrast_color_index = int((num_models-1)/2)
                #contrast_color_index = num_models+1 if num_models==5 else num_models
                contrast_empty_color = empty_color
            else:
                contrast_color_index = 0
                contrast_empty_color = empty_color
            for i in range(num_models):
                if node.accuracy_aggregate[i] > 0:
                    m_start_y = node.bounding_box[1] + i * (node.bounding_box[3]/num_models)
                    if i == 0:
                        vertical_pos = 'top'
                    elif i == num_models-1:
                        vertical_pos = 'bottom'
                    else:
                        vertical_pos = 'middle'
                    path_data_fill = rounded_box_path_multimodel(node.bounding_box[0], m_start_y, node.accuracy_aggregate[i]*node.bounding_box[2], node.bounding_box[3]/num_models, curve_radius/num_models, vertical_pos, curve_radius)
                    drawing.add(drawing.path(d=path_data_fill, stroke='none', fill=get_class_color(node, i, color_offset)))
            icon_color = get_max_contrast_color(contrast_empty_color, get_class_color(node, contrast_color_index, color_offset))
            icon_offset = 5
            if node.num_questions > 0:
                display_difficulty(drawing, direction, node.bounding_box[0]+sign*icon_offset, node.bounding_box[1]+node.bounding_box[3]/2, node.bounding_box[3], icon_color, node.difficulty_level_aggregate[0])
                icon_offset = icon_offset + 1.4*(5*(difficulty_bar_width+difficulty_bar_gap))
            if num_models == 1:
                display_response_time(drawing, direction, node.bounding_box[0]+sign*icon_offset, node.bounding_box[1]+node.bounding_box[3]/2, 0.75*(node.bounding_box[3]/2), node.average_time_aggregate[0], icon_color, 2*root_node.average_time_aggregate[0])
            if draw_qna_samples and node.num_questions > 0:
                qnapoint_offset = sign * (icon_offset+node.bounding_box[3]/1.5 + 0.75*(node.bounding_box[3]/2))
                qna_sample_color = get_max_contrast_color(contrast_empty_color, get_class_color(node, contrast_color_index, color_offset))
                if args.qna_sample_quality == 'high':
                    n_points = max(20000, 20*node.num_questions)
                else: 
                    n_points = max(min(max(2000, 10*node.num_questions), 40000), 2*node.num_questions)
                draw_qa_samples(drawing, path_data_outline, qna_sample_color, n_points, node.num_questions, qnapoint_offset=qnapoint_offset)
            drawing.add(drawing.path(d=path_data_outline, stroke=border_color, stroke_width='0.5', fill='none'))
        # Draw text labels
        g = drawing.g(style="font-size:" + str(node.font_size) + ";font-family:Roboto Condensed")
        if (node.depth < max_depth):
            text_path = drawing.path(d=center_line, fill="none", stroke="none")
            text = svgwrite.text.Text("")
            if (direction == "left"):
                if num_models == 1:
                    text.add(svgwrite.text.TextPath(text_path, text=node.name, method="align", startOffset="81%", baseline_shift=str(-node.font_size/3), text_anchor="end"))
                else:
                    text.add(svgwrite.text.TextPath(text_path, text=node.name, method="align", startOffset="90%", baseline_shift=str(-node.font_size/3), text_anchor="end"))
            elif (direction == "right"):
                if num_models == 1:
                    text.add(svgwrite.text.TextPath(text_path, text=node.name, method="align", startOffset="19%", baseline_shift=str(-node.font_size/3), text_anchor="start"))
                else:
                    text.add(svgwrite.text.TextPath(text_path, text=node.name, method="align", startOffset="10%", baseline_shift=str(-node.font_size/3), text_anchor="start"))
            else:
                print("ERROR: invalid direction specification")

            g.add(text_path)
            g.add(text)
        else:
            leafnode_label_offset = icon_offset+node.bounding_box[3]/1.5 + 0.75*(node.bounding_box[3]/2) + 5
            if (direction == "left"):             
                g.add(drawing.text(node.name, insert=(node.bounding_box[0] - leafnode_label_offset, node.bounding_box[1] + node.bounding_box[3]/2), text_anchor="end", alignment_baseline="central"))
            elif (direction == "right"):
                g.add(drawing.text(node.name, insert=(node.bounding_box[0] + leafnode_label_offset, node.bounding_box[1] + node.bounding_box[3]/2), text_anchor="start", alignment_baseline="central"))
            else:
                print("ERROR: invalid direction specification")
        drawing.add(g)
    for i, child in enumerate(node.children):
        tree_to_svg(drawing, child, direction, layer_width, smoothness, color_offset, draw_qna_samples)


# Generate SVG code for Bloom taxonomy
def bloom_to_svg(drawing, center_x, center_y, bloom_data, total_num_questions, width=200, level_height=20, draw_qna_samples=True):
    # Define the labels for each level
    labels = ['Remember', 'Understand', 'Apply', 'Analyze', 'Evaluate', 'Create']
    
    # Draw each level as a rectangle with text in the center
    for i, label in enumerate(labels):
        num_questions, accuracy = bloom_data[i]
        text_label = label
        empty_color = color_palette.gray_neutral(shade=10)
        fill_color = color_palette.gray_neutral(shade=40)
        border_color = color_palette.gray_neutral(shade=70)

        level_width = width - i*0.15*width
        x = center_x - level_width/2
        y = center_y - (i+1)*level_height
        drawing.add(drawing.rect((x, y), (level_width, level_height), fill=empty_color, stroke='none'))
        drawing.add(drawing.rect((x, y), (level_width * accuracy, level_height), fill=fill_color, stroke='none'))
        drawing.add(drawing.rect((x, y), (level_width, level_height), fill='none', stroke=border_color))
        g_label = drawing.g(style="font-size:15;font-family:Roboto Condensed")
        g_label.add(drawing.text(text_label, insert=(center_x, y + level_height/2), alignment_baseline='middle', text_anchor='middle', fill=border_color))
        drawing.add(g_label)
        num_questions = int(100*num_questions/total_num_questions)
        x_max = x+level_width
        y_max = y+level_height
        rect_path = f"M{x},{y} {x_max},{y} {x_max},{y_max} {x},{y_max}"
        if num_questions >0 and draw_qna_samples:
            
            if args.qna_sample_quality == 'high':
                n_points = max(50000, 10*num_questions)
            else:
                n_points = max(2000, 5*num_questions)
            draw_qa_samples(drawing, rect_path, border_color, n_points, num_questions, 1.4)
        

# Generate SVG code for info box showing model and data set parameters
def legend_to_svg(root_node, drawing, pos_x, pos_y, width):
    entry_height = 20
    num_models = len(root_node.llm_models)
    g_label = drawing.g(style="font-size:"+str(0.8*entry_height)+";font-family:Roboto Condensed")
    for i in range(num_models):
        model_name = root_node.llm_models[i]
        accuracy = root_node.accuracy_aggregate[i]
        empty_color = get_class_color(root_node, i, 0, shade=3)
        model_color = get_class_color(root_node, i, 0, shade=-1)
        border_color = get_class_color(root_node, i, 0, shade=8)
        x = pos_x-width/2
        y = pos_y+i*1.1*entry_height
        drawing.add(drawing.rect((x, y), (width, entry_height), fill=empty_color, stroke='none'))
        drawing.add(drawing.rect((x, y), (width * accuracy, entry_height), fill=model_color, stroke='none'))
        drawing.add(drawing.rect((x, y), (width, entry_height), fill='none', stroke=border_color))
        g_label.add(drawing.text(keywords_to_camel_case(model_name)+f' | {accuracy*100:.2f}%', insert=(pos_x, pos_y+i*1.1*entry_height+entry_height/2), text_anchor="middle", alignment_baseline="central"))
    drawing.add(g_label)


# Convert internal lower case strings to camel case for better UI output
def keywords_to_camel_case(input_string):
    map_names = {
        'chatgpt':'ChatGPT',
        'davinci':'GPT-3',
        'bloom':'BLOOM',
        'gpt2':'GPT-2',
        'llama13b':'LLaMa-13B',
        'pubmedqa':'PubMedQA',
        'sciq':'SciQ',
        'together_bloom':'BLOOM',
        'together_gpt-j-6b':'GPT-J-6B',
        'together_gpt-neox-20b':'GPT-NeoX-20B',
        'together_opt-66b':'OPT-66B',
        'together_opt-175b':'OPT-175B',
        'openbookqa':'OpenbookQA',
        'naturalqa':'NaturalQuestions',
        'truthfulqa':'TruthfulQA',
        'wikifact':'WikiFact',
        'mmlu':'MMLU',
        'usbar':'US Bar'
    }

    input_words = input_string.split()
    for i in range(len(input_words)):
        if input_words[i].strip(',') in map_names.keys():
            input_words[i] = input_words[i].replace(input_words[i].strip(','), map_names[input_words[i].strip(',')])
    output_string = ' '.join(input_words)
    return output_string


# Generate SVG code for info box showing model and data set parameters
def info_to_svg(drawing, root_node, total_height, draw_qna_samples):
    titel_font_size = 32
    info_posy = 350
    #TODO: font positioning is hard coded for illustration purposes, text need linebreaks or fontsize adaption
    #drawing.add(drawing.rect((-total_width/2, 0), (total_width, total_height), fill='none', stroke='red')) # draw debug rectangle around canvas
    if len(root_node.llm_models) == 1:
        # Single model use case
        g_title = drawing.g(style="font-size:"+str(titel_font_size)+";font-family:Roboto Condensed")
        if len(root_node.datasets) == 2:
            model_text = str(root_node.llm_models[0])+" Performance on {} and {}".format(*root_node.datasets)
        elif len(root_node.datasets) > 2:
            model_text = str(root_node.llm_models[0])+" Performance on " + "{}, "*(len(root_node.datasets)-2) + "{} and {}"
            model_text = model_text.format(*root_node.datasets)
        else:
            model_text = str(root_node.llm_models[0])+" Performance on {}".format(root_node.datasets[0])
        g_title.add(drawing.text(keywords_to_camel_case(model_text), insert=(0, titel_font_size*0.0), text_anchor="middle", alignment_baseline="central"))
        drawing.add(g_title)

        bloom_to_svg(drawing, 0, total_height/2+info_posy, bloom_data=root_node.bloom_classification_aggregate[0], total_num_questions=root_node.num_questions, draw_qna_samples=draw_qna_samples)

        g_stats = drawing.g(style="font-size:18;font-family:Roboto Condensed")
        g_stats.add(drawing.text("Overall accuracy: "+("%.2f" % (root_node.accuracy_aggregate[0]*100))+"%", insert=(0, total_height/2+info_posy+30), text_anchor="middle", alignment_baseline="central"))
        g_stats.add(drawing.text("Hallucination Score: "+f'{get_hallucination_score(root_node.difficulty_level_aggregate[0]):.2f}', insert=(0, total_height/2+info_posy+50), text_anchor="middle", alignment_baseline="central"))
        g_stats.add(drawing.text("Number of questions: "+str(root_node.num_questions), insert=(0, total_height/2+info_posy+70), text_anchor="middle", alignment_baseline="central"))
        drawing.add(g_stats)
    else:
        # Multiple model use case
        g_title = drawing.g(style="font-size:"+str(titel_font_size)+";font-family:Roboto Condensed")
        if len(root_node.llm_models) == 2:
            model_text = "{} and {}".format(*root_node.llm_models)
        else:
            model_text = "{}, "*(len(root_node.llm_models)-2) + "{} and {}"
            model_text = model_text.format(*root_node.llm_models)

        if len(root_node.datasets) == 2:
            data_text = "Performance Comparison on {} and {}".format(*root_node.datasets)
        elif len(root_node.datasets) > 2:
            data_text = "Performance Comparison on " + "{}, "*(len(root_node.datasets)-2) + "{} and {}"
            data_text = data_text.format(*root_node.datasets)
        else:
            data_text = "Performance Comparison on {}".format(root_node.datasets[0])

        g_title.add(drawing.text(keywords_to_camel_case(model_text), insert=(0, titel_font_size*-1.5), text_anchor="middle", alignment_baseline="central"))
        g_title.add(drawing.text(keywords_to_camel_case(data_text), insert=(0, titel_font_size*-0.5), text_anchor="middle", alignment_baseline="central"))
        drawing.add(g_title)

        legend_to_svg(root_node, drawing, 0, total_height/2+info_posy, 200)

        g_stats = drawing.g(style="font-size:20;font-family:Roboto Condensed")
        g_stats.add(drawing.text("Number of questions: "+str(root_node.num_questions), insert=(0, total_height/2+info_posy+len(root_node.llm_models)*25), text_anchor="middle", alignment_baseline="central"))
        drawing.add(g_stats)


###############################################################
#                                                             #
#    Generate trees, compute the layout and write it to SVG   # 
#                                                             #
###############################################################

#########################################
# Configurable visualization parameters #
#########################################
font_size_min = 13                                  # Minimal font size (for the leaf nodes)
font_size_max = 24                                  # Maximal font size (for the root nodes)
spacing_min = 0                                     # Minimal vertical text label margins (for the leaf nodes)
spacing_max = 10                                    # Maximal vertical text label margins (for the root nodes)
leaf_spacing = 3                                    # Spacing between child groups at leaf level
max_num_stacked_leafs = args.max_num_stacked_leafs  # Number of sibblings to be stacked at leaf node level before starting new column
layer_width = args.layer_width                                   # Width of path representing one layer
smoothness = 0.9                                    # Smoothness of the subtree paths
color_palette = Palette("material")                 # Use Google's material color palette
difficulty_bar_width = 3                            # Width of the bars depicitng the difficulty level
difficulty_bar_gap = 1.3                            # Gap between the bars depicitng the difficulty level
draw_qna_samples = args.draw_qna_samples            # Bluenoise plot number of questions per subdiscipline

# Load knowledge hierarchy
root_left, root_right, root_node = load_tree_from_json()
# If data for multiple models is loaded use hue to encode model (model_centric), otherwise subfield (data_centric)
if len(root_node.llm_models) == 1:
    color_mode = "data_centric"
else:
    color_mode = "model_centric"

# Calculate the maximum depth of the tree
leaf_nodes_left = search.findall(root_left, lambda node: not node.children)
leaf_nodes_right = search.findall(root_right, lambda node: not node.children)
max_depth = max(max(node.depth for node in leaf_nodes_left),max(node.depth for node in leaf_nodes_right))

# Calculate total width and height of SVG drawing
max_leaf_stack_depth=max(get_max_leaf_stack_depth(root_left,max_num_stacked_leafs),get_max_leaf_stack_depth(root_right,max_num_stacked_leafs))
total_width = 2.01 * (((max_depth-1) * layer_width)+(max_leaf_stack_depth * layer_width))+200
total_height = max(calc_height(root_left, max_num_stacked_leafs), calc_height(root_right, max_num_stacked_leafs))

# Calculate the positinioning of all labels and derive the bounding boxes
calc_label_positions(root_left, "left", layer_width, max_num_stacked_leafs, leaf_spacing)
calc_bounding_boxes(root_left, "left")
calc_label_positions(root_right, "right", layer_width, max_num_stacked_leafs, leaf_spacing)
calc_bounding_boxes(root_right, "right")

# Save the visualization as SVG
# TODO: Currently the layout only works for balanced trees with the same depth in all branches
drawing = svgwrite.Drawing(filename="{}_{}.svg".format('_'.join(root_node.llm_models), '_'.join(root_node.datasets)))
drawing.viewbox(-total_width/2, -64, total_width, total_height+64) #the 64 comes from the heading partially lying in the negative to not overdraw the map (2*titel_font_size)
tree_to_svg(drawing, root_left, "left", layer_width, smoothness, color_offset=0, draw_qna_samples=draw_qna_samples)
tree_to_svg(drawing, root_right, "right", layer_width, smoothness, color_offset=5, draw_qna_samples=draw_qna_samples)

info_to_svg(drawing, root_node, total_height, draw_qna_samples)

drawing.save()
print('LLMMap saved as', "{}_{}.svg".format('_'.join(root_node.llm_models), '_'.join(root_node.datasets)))