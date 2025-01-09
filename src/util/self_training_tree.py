from anytree import Node
import numpy as np


class SelfTrainingTree(object):

    def __init__(self, init_p_inds, root_ind=-1):

        self.nodes = [Node(root_ind, ind=root_ind)]
        self.ind_pos_in_nodes = {root_ind: 0}

        for p_ind in init_p_inds:
            self.append(p_ind, root_ind)

        self.root_ind = root_ind

    def single_ind_to_node(self, ind):
        return self.nodes[self.ind_pos_in_nodes[ind]]

    def multi_inds_to_nodes(self, inds):
        return [self.single_ind_to_node(ind) for ind in inds]

    def get_parent_ind(self, ind):
        return self.single_ind_to_node(ind).parent.ind

    def get_children_inds(self, ind):
        return self.nodes_to_inds(self.single_ind_to_node(ind).children)

    def poses_to_nodes(self, poses_in_nodes):
        return [self.nodes[pos] for pos in poses_in_nodes]

    def nodes_to_inds(self, nodes):
        return np.array([node.ind for node in nodes])

    def poses_to_inds(self, poses_in_nodes):
        return self.nodes_to_inds(self.poses_to_nodes(poses_in_nodes))

    def append(self, ind, nn_ind):

        parent = self.nodes[self.ind_pos_in_nodes[nn_ind]]
        self.nodes.append(
            Node(ind, ind=ind, parent=parent)
        )
        self.ind_pos_in_nodes[ind] = len(self.nodes) - 1

    def delete(self, ind=None, pos_in_nodes=None):
        if pos_in_nodes is None:
            # exec(gen_cmd_print_variables("self.ind_pos_in_nodes"))
            pos_in_nodes = self.ind_pos_in_nodes[ind]
        # ind = self.nodes[pos_in_nodes].ind

        poses_to_del = []
        self._recursive_delete(poses_to_del, pos_in_nodes=pos_in_nodes)

        poses_to_del = np.array(poses_to_del)
        inds_deleted = self.poses_to_inds(poses_to_del)
        self.nodes = [self.nodes[i] for i in range(len(self.nodes)) if i not in poses_to_del]
        self.ind_pos_in_nodes = dict(zip(self.nodes_to_inds(self.nodes), np.arange(len(self.nodes))))

        return inds_deleted

    def _recursive_delete(self, poses_to_del: list, ind=None, pos_in_nodes=None):

        if pos_in_nodes is None:
            pos_in_nodes = self.ind_pos_in_nodes[ind]

        self.nodes[pos_in_nodes].parent = None
        poses_to_del.append(pos_in_nodes)

        for child_node in self.nodes[pos_in_nodes].children:
            self._recursive_delete(poses_to_del, ind=child_node.ind)

    def print_tree(self):
        for node in self.nodes:
            print(f"{node.ind}: "
                  f"parent {node.parent.ind if node.parent is not None else None}, "
                  f"children {[child.ind for child in node.children]}")

    def get_leaves_no_init_p(self, ret_type="ind"):

        leave_nodes = [node for node in self.nodes if not node.children and node.parent.ind != self.root_ind]
        if ret_type == "node":
            return leave_nodes
        return self.nodes_to_inds(leave_nodes)

    def get_chains_with_n(self, labels_by_ind):

        chains_with_redundancy = {}
        for node in self.nodes[1:]:
            ind = node.ind
            if labels_by_ind[ind] not in (0, -1):
                continue

            parent_ind = node.parent.ind

            if labels_by_ind[parent_ind] not in (0, -1):
                pos_first_n_in_chain = 0 if labels_by_ind[ind] == 0 else -1
                chains_with_redundancy[ind] = (np.array([ind]), pos_first_n_in_chain)
                continue

            ori_chain, ori_pos_first_n_in_chain = chains_with_redundancy[parent_ind]
            if ori_pos_first_n_in_chain != -1:
                pos_first_n_in_chain = ori_pos_first_n_in_chain
            elif ori_pos_first_n_in_chain == -1 and labels_by_ind[ind] == -1:
                pos_first_n_in_chain = -1
            else:
                pos_first_n_in_chain = len(ori_chain)
            chains_with_redundancy[ind] = (np.append(ori_chain, ind), pos_first_n_in_chain)

        leaf_inds = self.get_leaves_no_init_p()
        leaf_inds_to_keep = leaf_inds[np.isin(labels_by_ind[leaf_inds], [0, -1])]
        chain_vals = [chains_with_redundancy[ind][0] for ind in leaf_inds_to_keep]
        all_pos_first_n_in_chain = [chains_with_redundancy[ind][1] for ind in leaf_inds_to_keep]

        chains = dict(zip(leaf_inds_to_keep, chain_vals))
        dict_pos_first_n_in_chain = dict(zip(leaf_inds_to_keep, all_pos_first_n_in_chain))

        return chains, dict_pos_first_n_in_chain

    def get_chains(self, labels_by_ind):
        '''
        get chains
        '''

        chains_with_redundancy = {}
        for node in self.nodes[1:]:
            ind = node.ind
            assert labels_by_ind[ind] != 0
            if labels_by_ind[ind] != -1:
                continue

            parent_ind = node.parent.ind
            assert labels_by_ind[parent_ind] in (-1, 1, 2)

            if labels_by_ind[parent_ind] != -1:
                chains_with_redundancy[ind] = np.array([ind])
                continue

            chains_with_redundancy[ind] = np.append(chains_with_redundancy[parent_ind], ind)

        leaf_inds = self.get_leaves_no_init_p()
        unknown_leaf_inds = leaf_inds[labels_by_ind[leaf_inds] == -1]
        chain_vals = [chains_with_redundancy[ind] for ind in unknown_leaf_inds]
        chains = dict(zip(unknown_leaf_inds, chain_vals))

        return chains

    def get_inf_by_ind(self, chains: dict, n_examples):

        leaf_inds = np.array(list(chains.keys()))

        inf_by_ind = np.zeros(n_examples, dtype=int)
        for chain_val in chains.values():
            if len(chain_val) == 1:
                continue

            cur_inf_by_ind = inf_by_ind[chain_val[:-1]]
            i_subtree_root = np.argmin(cur_inf_by_ind)
            if cur_inf_by_ind[i_subtree_root] != 0:
                continue
            subtree_root_ind = chain_val[i_subtree_root]
            inf_by_ind = self._recursive_get_inf(subtree_root_ind, leaf_inds, inf_by_ind)

        inf_by_ind[inf_by_ind == 0] = int(1e10)
        inf_by_ind[leaf_inds] = 0

        return inf_by_ind

    def _recursive_get_inf(self, subtree_root_ind, leaf_inds, inf_by_ind):

        child_inds = self.get_children_inds(subtree_root_ind)
        inf_by_ind[subtree_root_ind] = len(child_inds)

        non_leaf_child_inds = np.array([
            ind for ind in child_inds if ind not in leaf_inds
        ])
        if len(non_leaf_child_inds) == 0:
            return inf_by_ind

        for child_ind in non_leaf_child_inds:
            inf_by_ind = self._recursive_get_inf(child_ind, leaf_inds, inf_by_ind)
        inf_by_ind[subtree_root_ind] += np.sum(inf_by_ind[non_leaf_child_inds])

        return inf_by_ind





