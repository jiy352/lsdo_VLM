import networkx as nx
from csdl import SimulatorBase, Model, Subgraph, Operation, ImplicitOperation
from csdl.core.output import Output
from csdl.core.input import Input
from csdl.core.variable import Variable
from csdl.core.concatenation import Concatenation
import numpy as np
# import matplotlib.pyplot as plt
# import networkx as nx
# import numpy as np
# import scipy.sparse as sp


def apply_uq(model, num_nodes, rv_list):

    print(f'APPLYING UQ TO {model}')

    ops_to_expand = {}
    new_shape_dict = {}

    # get graph
    model.define()
    graph = make_graph(model)

    # ---get graph IMPLICIT---
    # model.define()
    impnode_dict = {}
    graph_check = make_graph_fake(model, impnode_dict)

    # exit()

    for impnode in impnode_dict:
        impgraph = make_graph_fake(impnode_dict[impnode]._model, {})

        for real_node in graph_check.nodes:
            real_csdls = graph_check.nodes[real_node]['csdl_node']
            for imp_node in impgraph.nodes:
                imp_csdls = impgraph.nodes[imp_node]['csdl_node']
                if real_node == imp_node:
                    for real_csdl in real_csdls:
                        if real_csdl in imp_csdls:
                            print('SAME NODE FOUND: ', real_csdl, real_node, imp_node)
                            print('IN:  ', real_csdl.dependencies)
                            print('OUT: ', real_csdl.dependents)
                            for rem_node in real_csdl.dependents:
                                rem_node_name = rem_node.name
                                graph.remove_edges_from([(real_node, rem_node_name)])
                            for rem_node in real_csdl.dependencies:
                                rem_node_name = rem_node.name
                                graph.remove_edges_from([(rem_node_name, real_node)])
    # ---get graph IMPLICIT---
    print('IS DAG? ', nx.is_directed_acyclic_graph(graph))
    if not nx.is_directed_acyclic_graph(graph):
        print(nx.find_cycle(graph))
        exit()

    node_list = []
    for node in graph.nodes:
        node_list.append(node)
    dependency_dict = get_influences(graph, node_list, rv_list)

    for node in dependency_dict:
        csdl_node = graph.nodes[node]['csdl_node']
        if isinstance(csdl_node, Operation):
            predecessors = list(graph.predecessors(node))
            if len(predecessors) > 1:
                current_dependency = dependency_dict[node]
                print('MULTIVARIATE OPERATION: ', node, current_dependency)
                for pred in predecessors:
                    csdl_pred = graph.nodes[pred]['csdl_node']
                    pred_dependency = dependency_dict[pred]
                    print('\tPREDECESSOR: ', pred, pred_dependency)
                    if current_dependency != pred_dependency:

                        pred_sum = 0.0
                        for key in pred_dependency:
                            pred_sum += pred_dependency[key]
                        cur_sum = 0.0
                        for key in current_dependency:
                            cur_sum += current_dependency[key]

                        if pred_sum == 0.0 and cur_sum == 1.0:
                            continue

                        if node not in ops_to_expand:
                            ops_to_expand[node] = {}
                            ops_to_expand[node]['rv_predecessor'] = {}
                        ops_to_expand[node]['rv_predecessor'][pred] = {}
                        ops_to_expand[node]['rv_predecessor'][pred]['es_string'] = get_es_string(pred_dependency)
                        temp_shape = list(csdl_pred.shape)
                        temp_shape[0] = temp_shape[0]*num_nodes
                        ops_to_expand[node]['rv_predecessor'][pred]['new_shape'] = tuple(temp_shape)
                        print('\t\tEXPANSION NEEDED: ', ops_to_expand[node]['rv_predecessor'][pred]['es_string'])

                if node in ops_to_expand:
                    print('\tEXPAND OPERATION: ', node)
                    change_shape(graph, node, new_shape_dict, num_nodes)

    # print(dependency_dict)
    # find expansion operations:

    # for rv in rv_list:
    #     iterate_uq(graph, rv, rv, ops_to_expand, new_shape_dict, num_nodes, rv_list)

    # for op in ops_to_expand:
    #     ops_to_expand[op]['rv_predecessor']
    #     for pred_node in ops_to_expand[op]['rv_predecessor']:
    #         dict = ops_to_expand[op]['rv_predecessor'][pred_node]
    #         ops_to_expand[op]['rv_predecessor'][pred_node]['es_string'] = get_es_string(dict['rv'], rv_list)

    model.uq_info = [new_shape_dict, ops_to_expand]
    # PRINTS:
    print()
    num_vars = 0
    for node in graph:
        if isinstance(graph.nodes[node]['csdl_node'], Variable):
            num_vars += 1
    print(f'total number of variables:                  \t{num_vars}')

    num_new_shapes = []
    num_expanded = 0
    for rv in rv_list:
        num_new_shapes.append(0)
    for var in new_shape_dict:
        num_expanded += 1
    print(f'total number of expanded variables:         \t{num_expanded}')
    print()
    print(f'total number of multivariate ops:           \t{len(ops_to_expand)}')


def iterate_uq(nxgraph, node, rv, ops_to_expand, new_shape_dict, num_nodes, rv_list):
    for next_node in nxgraph[node]:
        # print(next_node, node, rv)
        csdl_node = nxgraph.nodes[next_node]['csdl_node']
        if isinstance(csdl_node, Operation):

            expansion_bool = False
            if len(list(nxgraph.predecessors(next_node))) > 1:

                num_rv = 0
                for pred in nxgraph.predecessors(next_node):
                    for other_rv in rv_list:
                        if other_rv == rv:
                            continue
                        ancestors = nx.ancestors(nxgraph, pred)
                        if (other_rv in ancestors) and (rv not in ancestors):

                            expansion_bool = True

            if expansion_bool:
                print('MULTIVARIATE OPERATION: ', next_node)
                if next_node not in ops_to_expand:
                    ops_to_expand[next_node] = {}
                    ops_to_expand[next_node]['rv_predecessor'] = {}
                    change_shape(nxgraph, next_node, rv, new_shape_dict, num_nodes)
                ops_to_expand[next_node]['rv_predecessor'][node] = {}
                ops_to_expand[next_node]['rv_predecessor'][node]['rv'] = rv
                continue

        iterate_uq(nxgraph, next_node, rv, ops_to_expand, new_shape_dict, num_nodes, rv_list)


def get_es_string(rv_dict):

    dep_list = []
    for rv in rv_dict:
        dep_list.append(rv_dict[rv])

    if dep_list == [1.0, 0.0]:
        string = 'i...,p...->ip...'
    elif dep_list == [0.0, 1.0]:
        string = 'i...,p...->pi...'
    elif dep_list == [0.0, 0.0]:
        string = 'i...,p...->pi...'
    else:
        raise(ValueError())

    return string


def get_es_string_old(rv, rv_list):

    if rv == rv_list[0]:
        string = 'i...,p...->ip...'
    elif rv == rv_list[1]:
        string = 'i...,p...->pi...'
    return string


def change_shape_old(nxgraph, node, rv, new_shape_dict, num_nodes):

    print('Changing downstream shape of ', node)
    csdl_node = nxgraph.nodes[node]['csdl_node']
    if isinstance(csdl_node, Variable):
        new_shape_dict[node] = {}
        temp_shape = list(csdl_node.shape)
        temp_shape[0] = temp_shape[0]*num_nodes
        new_shape_dict[node]['new_shape'] = tuple(temp_shape)
        new_shape_dict[node]['touched_by'] = rv

    for down_node in nx.descendants(nxgraph, node):

        csdl_node = nxgraph.nodes[down_node]['csdl_node']

        if isinstance(csdl_node, Variable):

            if down_node not in new_shape_dict:
                new_shape_dict[down_node] = {}
                new_shape_dict[down_node]['touched_by'] = []
                old_shape = csdl_node.shape

            else:
                old_shape = new_shape_dict[down_node]['new_shape']

            if rv in new_shape_dict[down_node]['touched_by']:
                continue
            else:
                new_shape_dict[down_node]['touched_by'].append(rv)

            temp_shape = list(old_shape)
            temp_shape[0] = old_shape[0]*num_nodes
            new_shape_dict[down_node]['new_shape'] = tuple(temp_shape)

            # tile_shape = zip(tuple(temp_shape),old_shape)
            tile_shape = tuple([int(a/b) for a, b in zip(tuple(temp_shape), old_shape)])
            new_shape_dict[down_node]['new_val'] = np.tile(csdl_node.val, tile_shape)
            # print(f'for rv {rv}, ', down_node, ': ', old_shape, '->', new_shape_dict[down_node]['new_shape'])

            # def insert_node(self, node)


def change_shape(nxgraph, node, new_shape_dict, num_nodes):

    print('Changing downstream shape of ', node)
    csdl_node = nxgraph.nodes[node]['csdl_node']
    if isinstance(csdl_node, Variable):
        new_shape_dict[node] = {}
        temp_shape = list(csdl_node.shape)
        temp_shape[0] = temp_shape[0]*num_nodes
        new_shape_dict[node]['new_shape'] = tuple(temp_shape)
        tile_shape = tuple([int(a/b) for a, b in zip(tuple(temp_shape), csdl_node.shape)])
        new_shape_dict[node]['new_val'] = np.tile(csdl_node.val, tile_shape)

    for down_node in nx.descendants(nxgraph, node):

        csdl_node = nxgraph.nodes[down_node]['csdl_node']
        if isinstance(csdl_node, Variable):
            if down_node not in new_shape_dict:
                new_shape_dict[down_node] = {}
                temp_shape = list(csdl_node.shape)
                temp_shape[0] = temp_shape[0]*num_nodes
                new_shape_dict[down_node]['new_shape'] = tuple(temp_shape)
                tile_shape = tuple([int(a/b) for a, b in zip(tuple(temp_shape), csdl_node.shape)])
                new_shape_dict[down_node]['new_val'] = np.tile(csdl_node.val, tile_shape)


def make_graph(model):
    '''
    Given sorted nodes, return a directed Networkx graph containing nodes with only operations/variables.
    Creates graph sequentially by looping through nodes and adding edges through node.dependencies and node.dependents.
    When a node is a Model, recursively expand to only operations/variables and insert to larger graph.
    The nodes of the networkx graph is a string of the csdl node name. The 'csdl_node' attribute contains the csdl node object.
    '''

    # initialize directed graph and reversed nodes
    DG = nx.DiGraph()
    sorted_nodes = list(reversed(model.sorted_nodes))

    # Add in inputs that may not be part of sorted_nodes
    for node in model.inputs:
        DG.add_node(node.name)
        DG.nodes[node.name]['csdl_node'] = node

    # loop through nodes and add edges
    # node is a csdl node object.
    # TODO: networkx DiGraph allows the same edge to be added multiple times. This is done to make sure no edges are 'missed' but can be optimized so all edges are added only once.
    for node in sorted_nodes:

        with open('hello.txt', 'a') as f:
            # f.write('readme')
            f.write(f'{node.name}, {node}, {model} \n')

        # print(node.name, node, model)

        # if node.name in ['gamma_b', '_00gw']:
        #     print('\n', node.name, node, model)
        #     for dependent in node.dependents:
        #         print('\tdependent', dependent, dependent.name)
        #     for dependency in node.dependencies:
        #         print('\tdependency', dependency, dependency.name)

        # if node.name == '_00gw':
        #     print(node.name, node, node.name)
        #     for dependent in node.dependents:
        #         print('\tdependent', dependent, dependent.name)
        #     for dependency in node.dependencies:
        #         print('\tdependency', dependency, dependency.name)
        # if isinstance(node, Concatenation):
        #     print('CONCATENATION: ', node.name)

        # if output, add node dependents as edges.
        if isinstance(node, Output):
            for dependent in node.dependents:

                # ignore dependents
                if isinstance(dependent, Subgraph):
                    continue

                DG.add_edges_from([(node.name, dependent.name)])

            # node attributes
            DG.nodes[node.name]['csdl_node'] = node

        # if input, add node dependents as edges.
        elif isinstance(node, Input):
            for dependent in node.dependents:

                # ignore model nodes
                if isinstance(dependent, Subgraph):
                    continue

                DG.add_edges_from([(node.name, dependent.name)])

            # node attributes
            DG.nodes[node.name]['csdl_node'] = node

        # if operation, add both dependents and dependencies as edges.
        elif isinstance(node, Operation):
            if isinstance(node, ImplicitOperation):
                print('IMPLICIT_OPERATION:', node.name)
                # self.make_graph(node._model)

            # if type(node) not in self.operation_analytics:
            #     self.operation_analytics[type(node)] = {}
            #     self.operation_analytics[type(node)]['count'] = 0
            # self.operation_analytics[type(node)]['count'] += 1

            for dependent in node.dependents:
                DG.add_edges_from([(node.name, dependent.name)])

                DG.nodes[dependent.name]['csdl_node'] = dependent

            for dependency in node.dependencies:
                DG.add_edges_from([(dependency.name, node.name)])

                DG.nodes[dependency.name]['csdl_node'] = dependency

            # node attributes
            DG.nodes[node.name]['csdl_node'] = node

        # if the node is a model, make_graph(model) and DG = union(DG, makegraph(model))
        elif isinstance(node, Subgraph):

            DG_temp = make_graph(node.submodel)

            DG = nx.compose(DG, DG_temp)

    # Return graph
    return DG


def make_graph_fake(model, impnode):
    '''
    Given sorted nodes, return a directed Networkx graph containing nodes with only operations/variables.
    Creates graph sequentially by looping through nodes and adding edges through node.dependencies and node.dependents.
    When a node is a Model, recursively expand to only operations/variables and insert to larger graph.
    The nodes of the networkx graph is a string of the csdl node name. The 'csdl_node' attribute contains the csdl node object.
    '''

    # initialize directed graph and reversed nodes
    DG = nx.DiGraph()
    sorted_nodes = list(reversed(model.sorted_nodes))

    # Add in inputs that may not be part of sorted_nodes
    for node in model.inputs:
        DG.add_node(node.name)
        if 'csdl_node' not in DG.nodes[node.name]:
            DG.nodes[node.name]['csdl_node'] = [node]
        elif node not in DG.nodes[node.name]['csdl_node']:
            DG.nodes[node.name]['csdl_node'].append(node)

    # loop through nodes and add edges
    # node is a csdl node object.
    # TODO: networkx DiGraph allows the same edge to be added multiple times. This is done to make sure no edges are 'missed' but can be optimized so all edges are added only once.
    for node in sorted_nodes:

        with open('hello.txt', 'a') as f:
            # f.write('readme')
            f.write(f'{node.name}, {node}, {model} \n')

        # print(node.name, node, model)

        # if node.name in ['gamma_b', '_00gw']:
        #     print('\n', node.name, node, model)
        #     for dependent in node.dependents:
        #         print('\tdependent', dependent, dependent.name)
        #     for dependency in node.dependencies:
        #         print('\tdependency', dependency, dependency.name)

        # if node.name == '_00gw':
        #     print(node.name, node, node.name)
        #     for dependent in node.dependents:
        #         print('\tdependent', dependent, dependent.name)
        #     for dependency in node.dependencies:
        #         print('\tdependency', dependency, dependency.name)
        # if isinstance(node, Concatenation):
        #     print('CONCATENATION: ', node.name)

        # if output, add node dependents as edges.
        if isinstance(node, Output):
            for dependent in node.dependents:

                # ignore dependents
                if isinstance(dependent, Subgraph):
                    continue

                DG.add_edges_from([(node.name, dependent.name)])

                try:
                    DG.nodes[dependent.name]['csdl_node'].append(dependent)
                except:
                    DG.nodes[dependent.name]['csdl_node'] = [dependent]

            # node attributes
            try:
                DG.nodes[node.name]['csdl_node'].append(node)
            except:
                DG.nodes[node.name]['csdl_node'] = [node]
            # DG.nodes[node.name]['csdl_node'] = node

        # if input, add node dependents as edges.
        elif isinstance(node, Input):
            for dependent in node.dependents:

                # ignore model nodes
                if isinstance(dependent, Subgraph):
                    continue

                DG.add_edges_from([(node.name, dependent.name)])

                try:
                    DG.nodes[dependent.name]['csdl_node'].append(dependent)
                except:
                    DG.nodes[dependent.name]['csdl_node'] = [dependent]

            # node attributes
            try:
                DG.nodes[node.name]['csdl_node'].append(node)
            except:
                DG.nodes[node.name]['csdl_node'] = [node]
            # DG.nodes[node.name]['csdl_node'] = node

        # if operation, add both dependents and dependencies as edges.
        elif isinstance(node, Operation):
            if isinstance(node, ImplicitOperation):
                impnode[node.name] = node
                print('---------IMPLICIT_OPERATION:', node.name)
                print('OUT')
                for dependent in node.dependents:
                    print(dependent, dependent.name)
                print('IN')
                for dependency in node.dependencies:
                    print(dependency, dependency.name)
                print('---------IMPLICIT_OPERATION: ', node.name)

                # self.make_graph(node._model)

            # if type(node) not in self.operation_analytics:
            #     self.operation_analytics[type(node)] = {}
            #     self.operation_analytics[type(node)]['count'] = 0
            # self.operation_analytics[type(node)]['count'] += 1

            for dependent in node.dependents:
                DG.add_edges_from([(node.name, dependent.name)])

                # DG.nodes[dependent.name]['csdl_node'] = dependent

                # if 'csdl_node' not in DG.nodes[dependent.name]:
                #     DG.nodes[dependent.name]['csdl_node'] = [dependent]
                # elif dependent not in DG.nodes[dependent.name]['csdl_node']:

                try:
                    DG.nodes[dependent.name]['csdl_node'].append(dependent)
                except:
                    DG.nodes[dependent.name]['csdl_node'] = [dependent]

            for dependency in node.dependencies:
                DG.add_edges_from([(dependency.name, node.name)])

                # DG.nodes[dependency.name]['csdl_node'] = dependency

                # if 'csdl_node' not in DG.nodes[dependency.name]:
                #     DG.nodes[dependency.name]['csdl_node'] = [dependency]
                # elif dependency not in DG.nodes[dependency.name]['csdl_node']:
                #     DG.nodes[dependency.name]['csdl_node'].append(dependency)

                try:
                    DG.nodes[dependency.name]['csdl_node'].append(dependency)
                except:
                    DG.nodes[dependency.name]['csdl_node'] = [dependency]

                # if dependency.name == 'gamma_b':
                #     DG.nodes[dependency.name]['csdl_node'].append(dependency)
                #     print(DG.nodes[dependency.name]['csdl_node'])
            # node attributes
            # DG.nodes[node.name]['csdl_node'] = node
            try:
                DG.nodes[node.name]['csdl_node'].append(node)
            except:
                DG.nodes[node.name]['csdl_node'] = [node]
            # if 'csdl_node' not in DG.nodes[node.name]:
            #     DG.nodes[node.name]['csdl_node'] = [node]
            # elif node not in DG.nodes[node.name]['csdl_node']:
            #     DG.nodes[node.name]['csdl_node'].append(node)
        # if the node is a model, make_graph(model) and DG = union(DG, makegraph(model))
        elif isinstance(node, Subgraph):

            DG_temp = make_graph_fake(node.submodel, impnode)

            for node_sub in DG_temp:
                if node_sub in DG:
                    DG_temp.nodes[node_sub]['csdl_node'].extend(DG.nodes[node_sub]['csdl_node'])

            DG = nx.compose(DG, DG_temp)

    # Return graph
    return DG


def get_influences(G, outputs, inputs, return_type='dict'):

    if return_type == 'dict':
        influences = {}
    elif return_type == 'array':
        influences = np.zeros((len(outputs), len(inputs)))
    else:
        raise(ValueError)

    output_index = 0
    for output_node in outputs:
        if return_type == 'dict':
            influences[output_node] = {}

        depends_on = list(nx.ancestors(G, output_node))

        for input_node in inputs:
            if return_type == 'dict':
                influences[output_node][input_node] = 0.0
            if input_node in depends_on:
                if return_type == 'array':
                    input_index = inputs.index(input_node)
                    influences[output_index, input_index] = 1.0
                elif return_type == 'dict':
                    influences[output_node][input_node] = 1.0
        output_index += 1

    return influences
# gamma_b <csdl.core.declared_variable.DeclaredVariable object at 0x7fbbc1ebfb50> (3, 16) <VLM_package.VLM_system.solve_circulations.solve_group.SolveMatrix object at 0x7fbbc1b91220>
# IN:
# OUT:
#          <csdl.operations.einsum.einsum object at 0x7fbbc1ebfbb0> _00Hs
#          <csdl.core.implicit_operation.ImplicitOperation object at 0x7fbbc1eef1c0> _00Hy

# gamma_b <csdl.core.output.Output object at 0x7fbbc1eef2e0> (3, 16) <VLM_package.VLM_system.solve_circulations.solve_group.SolveMatrix object at 0x7fbbc1b91220>
# IN:
#          <csdl.core.implicit_operation.ImplicitOperation object at 0x7fbbc1eef1c0> _00Hy
# OUT:
