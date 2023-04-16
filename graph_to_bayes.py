import random
import numpy as np
from pgmpy.models import BayesianNetwork
from pgmpy.factors.discrete import TabularCPD

def random_graph_generator(n=25, dim=5, seed=4):

    dict_list = []
    random.seed(seed)

    for i in range(n):
        row_dict = {}
        inward_sum = 0
        outward_sum = 0

        row = i // dim
        col = i % dim

        # Add edge to the left
        if col > 0:
            left_index = i - 1
            prob_left = random.uniform(0, 1)
            row_dict[left_index] = prob_left
            outward_sum += prob_left

        # Add edge to the right
        if col < dim - 1:
            right_index = i + 1
            prob_right = random.uniform(0, 1 - outward_sum)
            row_dict[right_index] = prob_right
            outward_sum += prob_right

        # Add edge above
        if row > 0:
            up_index = i - dim
            prob_up = random.uniform(0, 1 - inward_sum - outward_sum)
            row_dict[up_index] = prob_up
            inward_sum += prob_up

        # Add edge below
        if row < dim - 1:
            down_index = i + dim
            prob_down = random.uniform(0, 1 - inward_sum - outward_sum)
            row_dict[down_index] = prob_down
            inward_sum += prob_down
            
        total_prob = inward_sum + outward_sum
        for key in row_dict.keys():
            row_dict[key] /= total_prob

        dict_list.append(row_dict)
    return dict_list


def random_graph_generator_self(n=25, dim=5, seed=4):

    dict_list = []
    random.seed(seed)

    for i in range(n):
        row_dict = {}
        inward_sum = 0
        outward_sum = 0

        row = i // dim
        col = i % dim

        # Add edge to the left
        if col > 0:
            left_index = i - 1
            prob_left = random.uniform(0, 1)
            row_dict[left_index] = prob_left
            outward_sum += prob_left

        # Add edge to the right
        if col < dim - 1:
            right_index = i + 1
            prob_right = random.uniform(0, 1 - outward_sum)
            row_dict[right_index] = prob_right
            outward_sum += prob_right

        # Add edge above
        if row > 0:
            up_index = i - dim
            prob_up = random.uniform(0, 1 - inward_sum - outward_sum)
            row_dict[up_index] = prob_up
            inward_sum += prob_up

        # Add edge below
        if row < dim - 1:
            down_index = i + dim
            prob_down = random.uniform(0, 1 - inward_sum - outward_sum)
            row_dict[down_index] = prob_down
            inward_sum += prob_down
            
        
        # Add probability to self
        prob_self = random.uniform(0, 1 - inward_sum - outward_sum)
        row_dict[i] = prob_self + inward_sum
        total_prob = inward_sum + outward_sum + prob_self
            
        for key in row_dict.keys():
            row_dict[key] /= total_prob

        dict_list.append(row_dict)
    return dict_list



def move(graph, cur_state, direction, gw):
    next_state = -1
    direction_dict = {
        0: 'up',
        1: 'down',
        2: 'left',
        3: 'right'
    }
    direction = direction_dict[direction]
    if direction == 'up':
        next_state = cur_state + gw
    elif direction == 'down':
        next_state = cur_state - gw
    elif direction == 'left':
        if cur_state % gw != 0:
            next_state = cur_state - 1
    elif direction == 'right':
        if (cur_state + 1) % gw != 0:
            next_state = cur_state + 1
            
    if next_state not in range(0, len(graph)):
        return False
    return next_state



def generate_sussessor(graph, current, gw):
    successor_list = []
    for i in range(4):
        suc = move(graph, current, i, gw)
        if suc is not False:
            successor_list.append(suc)
    return successor_list


def BFS(graph, gw, start_state):
    edge_list = []
    queue = []
    visited = []
    queue.append(start_state)
    while len(queue) != 0:
        node = queue.pop(0)
        successors = generate_sussessor(graph, node, gw)
        if len(successors) != 0:
            for s in successors:
                if s not in visited:
                    queue.append(s)
                    tuple_c = (node, s)
                    if tuple_c not in edge_list:
                        edge_list.append(tuple_c)
        visited.append(node)
    return edge_list


def obtain_values(tran_mat, node):
    true_list = []
    evidence = []
    evidence_card = []
    tran_mat_transpose = np.transpose(tran_mat)
   
    for index, element in enumerate(tran_mat_transpose[:][int(node)]):
        if element != 0:
            evidence.append(str(index))
            evidence_card.append(2)
            true_list.append(element)
            true_list.append(1 - element)
            
    false_list = [1.0 - true_list[i] for i in range(len(true_list))]
    return [true_list, false_list], evidence, evidence_card


def bayes_graph_structure_setup(size=5, init_location=0, transition_model=None, seed=4, diagonal=False):
    graph_use = [i for i in range(size * size)]
    list_of_edges = BFS(graph_use, size, init_location)
    print("Edge list:", list_of_edges)
    str_edge_list = []
    for element in list_of_edges:
        str_edge_list.append((str(element[0]), str(element[1])))
    model = BayesianNetwork(str_edge_list)
    print("Bayes model:", model)
    cpd_appending_order = [str(init_location)]
    for index, tuple in enumerate(list_of_edges):
        ele = tuple[1]
        if str(ele) not in cpd_appending_order:
            cpd_appending_order.append(str(ele))
    print("CPD appending order:", cpd_appending_order)
    state_all = [True, False] # Reach and unreach
    state = {}
    for i in range(size * size):
        state[i] = i
    if transition_model == None:
        if not diagonal:
            dict_list = random_graph_generator(size * size, size, seed)
            # Transition Matrix
            A = np.zeros((size * size, size * size))
            for index, dict in enumerate(dict_list):
                for key in dict.keys():
                    A[index][key] = dict[key]
            # Bayes Matrix
            B = np.zeros((size * size, size * size))
            for index, (s, e) in enumerate(list_of_edges):
                B[s, e] = A[s, e]
            transition_model = B
        else:
            dict_list = random_graph_generator_self(size * size, size, seed)
            # Transition Matrix
            A = np.zeros((size * size, size * size))
            for index, dict in enumerate(dict_list):
                for key in dict.keys():
                    A[index][key] = dict[key]
            # Bayes Matrix
            B = np.zeros((size * size, size * size))
            for index, (s, e) in enumerate(list_of_edges):
                B[s, e] = A[s, e]
            transition_model = B
    else:
        B = transition_model
    
    for index, element in enumerate(cpd_appending_order):
        if index == 0:
            cpd = TabularCPD(variable=element, 
                    variable_card=2, 
                    values=[[0.5], [0.5]], 
                    state_names={element: state_all})
            model.add_cpds(cpd)
        else:
            values, evidence, evidence_card = obtain_values(tran_mat=B, node=element)
            state_name = {element: state_all}
            for index, element_evid in enumerate(evidence):
                state_name[element_evid] = state_all
            cpd = TabularCPD(variable=element, 
                    variable_card=2, 
                    values=values, 
                    evidence=evidence, 
                    evidence_card=evidence_card,
                    state_names=state_name)
            model.add_cpds(cpd)
            
    model.check_model()
    return model, A
            
    