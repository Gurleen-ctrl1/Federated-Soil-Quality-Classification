# without edge server but with deduction logic
'''
import copy
import torch

class Server:
    def __init__(self, input_size):
        self.global_model = None
        self.input_size = input_size

    def aggregate(self, client_models):
        global_dict = copy.deepcopy(client_models[0])
        for key in global_dict.keys():
            for i in range(1, len(client_models)):
                global_dict[key] += client_models[i][key]
            global_dict[key] = torch.div(global_dict[key], len(client_models))
        return global_dict

    def update_global_model(self, global_weights):
        self.global_model = global_weights
'''

# with edge server and basic deduction logic
'''
import copy
import torch

class Server:
    def __init__(self):
        self.global_model = None

    def aggregate_from_edges(self, edge_models):
        agg_model = copy.deepcopy(edge_models[0])
        for key in agg_model:
            for i in range(1, len(edge_models)):
                agg_model[key] += edge_models[i][key]
            agg_model[key] /= len(edge_models)
        return agg_model

    def update_global_model(self, global_weights):
        self.global_model = global_weights
'''
# without both edge server and deduction logic
# server.py

import copy
import torch

class Server:
    def __init__(self, input_size):
        self.global_model = None
        self.input_size = input_size

    def aggregate(self, client_models):
        global_dict = copy.deepcopy(client_models[0])
        for key in global_dict.keys():
            for i in range(1, len(client_models)):
                global_dict[key] += client_models[i][key]
            global_dict[key] = torch.div(global_dict[key], len(client_models))
        return global_dict

    def update_global_model(self, global_weights):
        self.global_model = global_weights
