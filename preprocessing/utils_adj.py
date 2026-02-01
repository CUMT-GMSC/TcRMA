# from load_data import Data
import numpy as np
# import time
import torch
import os
from collections import defaultdict, Counter
import scipy.sparse as sp
from scipy import sparse
from sklearn.metrics.pairwise import cosine_similarity

torch.manual_seed(1337)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(1337)
np.random.seed(1337)


def normalize_sparse(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_pow_half = np.power(rowsum, -1/2).flatten() # D^(-1/2)
    r_pow_half[np.isinf(r_pow_half)] = 0.
    r_mat_inv = sp.diags(r_pow_half) # D^(-1/2)
    mx = r_mat_inv.dot(mx.dot(r_mat_inv)) # D^(-1/2) * A * D^(-1/2)
    return mx


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse_coo_tensor(indices, values, shape, dtype=torch.float32)

#约束规则
def create_adj_from_dict(triples, num_ents, num_rels, threshold):
    """
    Parameter:
    triples: list
        - triples (each triple is a list of 3 elements)
    num_ents: int
        - number of entities
    num_rels: int
        - number of relations

    Return: 
    edge_mat: torch sparse tensor
        - adjacency matrix with selfloop
    """
    r1r2_set = Counter(["{}_{}".format(ele[0], ele[2]) for ele in triples]) # number of times h and r appear together
    co_times = np.sort(list(r1r2_set.values()))

    num_pairs = len(co_times)
    num_kept = int(num_pairs * threshold) # what is this threshold in the paper?

    if num_kept > 0:
        minval = co_times[-num_kept]
    else:
        minval = max(co_times) + 1

    seen_r1e = set()
    seen_er2 = set()
    seen_r1r2 = set()

    row_indxs = []
    col_indxs = []
    dat_values = []

    for triple in triples:
        r1, e, r2 = triple 
        if r1r2_set["{}_{}".format(r1, r2)] < minval:
            continue

        value1, value2, value3 = 1, 1, 1

        if "{}_{}".format(r1, e) in seen_r1e:
            value1 = 0
        else:
            seen_r1e.add("{}_{}".format(r1, e))
        
        if "{}_{}".format(e, r2) in seen_er2:
            value2 = 0
        else:
            seen_er2.add("{}_{}".format(e, r2))
        
        if "{}_{}".format(r1, r2) in seen_r1r2:
            value3 = 0
        else:
            seen_r1r2.add("{}_{}".format(r1, r2))

        list_1 = [r1 + num_ents, e, r1 + num_ents]
        list_2 = [e, r2 + num_ents, r2 + num_ents]

        values = [value1, value2, value3]

        for v_index in values:
            if values[v_index] != 0:
                row_indxs.append(list_1[v_index])
                row_indxs.append(list_2[v_index])
                col_indxs.append(list_2[v_index])
                col_indxs.append(list_1[v_index])
                dat_values.append(values[v_index])
                dat_values.append(values[v_index])

    edge_mat = sparse.csr_matrix((dat_values, (row_indxs, col_indxs)), shape=(num_ents + num_rels, num_ents + num_rels))
    edge_mat = edge_mat + sparse.eye(edge_mat.shape[0], format="csr")
    edge_mat[edge_mat > 1] = 1

    edge_mat = normalize_sparse(edge_mat)
    edge_mat = sparse_mx_to_torch_sparse_tensor(edge_mat).to(device)
    return edge_mat

def construct_relation_focus_matrix_old(data, entity_idxs, rel_idxs, threshold):
    """
    construct the relation-focus graph
    """
    num_relations = int(len(rel_idxs) / 2) # does not count the inversed relations

    head_rel_dict = defaultdict(set) # {ent: {rel of triples having ent as head}}
    tail_rel_dict = defaultdict(set) # {ent: {rel of triples having ent as tail}}

    ent_set = set() # set of all entities

    new_triple = [] # <rel, ent, rel> triple
    new_triple_text = set()

    for triple in data:
        head, rel, tail = triple
        head = entity_idxs[head]
        rel = rel_idxs[rel]
        tail = entity_idxs[tail]

        ent_set.update([head, tail])
        head_rel_dict[head].add(rel)
        tail_rel_dict[tail].add(rel)
    
    for ent in ent_set:
        as_tail_set = head_rel_dict[ent] # relations as tail
        as_head_set = tail_rel_dict[ent] # relations as head

        for ele in as_tail_set:
            for ele2 in as_head_set:
                str_key = "{}_{}_{}".format(ele, ent, ele2)
                # ignore inverse relations connections
                if (ele % num_relations) == (ele2 % num_relations):
                    # print(ele, ele2)
                    continue
                if str_key not in new_triple_text:
                    new_triple_text.add(str_key)
                    new_triple.append([ele, ent, ele2])

    print("Number of ori triples: {}".format(len(data)))
    print("Number of rel triples: {}".format(len(new_triple)))
    edge_mat = create_adj_from_dict(new_triple, len(entity_idxs), len(rel_idxs), threshold)    
    return edge_mat


def construct_relation_focus_matrix_nosim(data, entity_idxs, rel_idxs, dataset=None):
    """
    实体、关系作为节点 根据共现
    """
    num_relations = int(len(rel_idxs) / 2) # 不包含逆关系的数量

    # --- 构建结构共现信息 ---
    head_rel_dict = defaultdict(set)
    tail_rel_dict = defaultdict(set)
    ent_set = set()
    new_triple = [] 
    new_triple_text = set()

    for triple in data:
        head, rel, tail = triple
        head = entity_idxs[head]
        rel = rel_idxs[rel]
        tail = entity_idxs[tail]
        ent_set.update([head, tail])
        head_rel_dict[head].add(rel)
        tail_rel_dict[tail].add(rel)
    
    for ent in ent_set:
        as_tail_set = head_rel_dict[ent]
        as_head_set = tail_rel_dict[ent]
        for ele in as_tail_set:
            for ele2 in as_head_set:
                # 忽略逆关系自连
                if (ele % num_relations) == (ele2 % num_relations):
                    continue
                str_key = "{}_{}_{}".format(ele, ent, ele2)
                if str_key not in new_triple_text:
                    new_triple_text.add(str_key)
                    new_triple.append([ele, ent, ele2])

    edge_index = []
    edge_type = []
    
    seen_r1e = set()
    seen_er2 = set()
    seen_r1r2 = set()
    
    for triple in new_triple:
        r1, e, r2 = triple
        
        value1, value2, value3 = 1, 1, 1 

        # 去重逻辑
        if "{}_{}".format(r1, e) in seen_r1e: value1 = 0
        else: seen_r1e.add("{}_{}".format(r1, e))
        
        if "{}_{}".format(e, r2) in seen_er2: value2 = 0
        else: seen_er2.add("{}_{}".format(e, r2))
        
        if "{}_{}".format(r1, r2) in seen_r1r2: value3 = 0
        else: seen_r1r2.add("{}_{}".format(r1, r2))

        # 添加 Type 0: r1 - e
        if value1 > 0:
            edge_index.append((r1 + len(entity_idxs), e))
            edge_type.append(0)
            edge_index.append((e, r1 + len(entity_idxs)))
            edge_type.append(0)
            
        # 添加 Type 1: e - r2
        if value2 > 0:
            edge_index.append((e, r2 + len(entity_idxs)))
            edge_type.append(1)
            edge_index.append((r2 + len(entity_idxs), e))
            edge_type.append(1)
            
        # 添加 Type 2: r1 - r2 (结构共现)
        if value3 > 0:
            edge_index.append((r1 + len(entity_idxs), r2 + len(entity_idxs)))
            edge_type.append(2)
            edge_index.append((r2 + len(entity_idxs), r1 + len(entity_idxs)))
            edge_type.append(2)

    # print(f"仅结构边: 添加了 {len(new_triple)} 个关系三元组")

    # 转换为 Tensor
    edge_index = torch.LongTensor(edge_index).to(device).t()
    edge_type = torch.LongTensor(edge_type).to(device)
    
    return edge_index, edge_type


def construct_entity_focus_matrix(data, entity_idxs, rel_idxs):
    """
    构建实体角度的邻接矩阵，返回edge_index和edge_type格式
    
    Parameters
    ----------
    data: list
        三元组数据 [(head, rel, tail), ...]
    entity_idxs: dict
        实体到ID的映射
    rel_idxs: dict
        关系到ID的映射
        
    Returns
    -------
    edge_index: torch.LongTensor
        shape (2, num_edges), 边的起始和目标节点
    edge_type: torch.LongTensor  
        shape (num_edges,), 每条边对应的关系类型
    """
    edge_index = []
    edge_type = []
    
    # 只添加原始方向的边
    for hrt in data:
        head = entity_idxs[hrt[0]]
        rel = rel_idxs[hrt[1]]
        tail = entity_idxs[hrt[2]]
        
        edge_index.append((head, tail))
        edge_type.append(rel)
    
    edge_index = torch.LongTensor(edge_index).to(device).t()
    edge_type = torch.LongTensor(edge_type).to(device)
    
    return edge_index, edge_type


def get_deg(train_data_idxs, n_entities):
    deg = Counter([ele[0] for ele in train_data_idxs] + [ele[-1] for ele in train_data_idxs])
    deg = np.array([deg[i] for i in range(n_entities)])
    return torch.FloatTensor(deg)

def get_er_vocab(data):
    er_vocab = defaultdict(list) # er_vocab is a dict of (h, r) and t_index, if r is a reverse relation ==> t_index += n_entities
    for triple in data:
        er_vocab[(triple[0], triple[1])].append(triple[2])
    return er_vocab

def get_batch(er_vocab, er_vocab_pairs, idx, batch_size, d):
    batch = er_vocab_pairs[idx:idx + batch_size]
    targets = np.zeros((len(batch), len(d.entities)))
    for idx, pair in enumerate(batch):
        targets[idx, er_vocab[pair]] = 1.
    targets = torch.FloatTensor(targets)
    return np.array(batch), targets.to(device)


