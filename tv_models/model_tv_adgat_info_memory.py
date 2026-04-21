from helper import *


from tv_models.ent_encoder_adgat import ent_adGat
from tv_models.rel_encoder_adgat_memory import rel_adGat_memory


import numpy as np
import os

def make_quaternion_mul(kernel):
    """" The constructed 'hamilton' W is a modified version of the quaternion representation,
    thus doing tf.matmul(Input,W) is equivalent to W * Inputs. """
    dim = kernel.size(1) // 4
    r, i, j, k = torch.split(kernel, [dim, dim, dim, dim], dim=1)
    r2 = torch.cat([r, -i, -j, -k], dim=0)  # 0, 1, 2, 3
    i2 = torch.cat([i, r, -k, j], dim=0)  # 1, 0, 3, 2
    j2 = torch.cat([j, k, r, -i], dim=0)  # 2, 3, 0, 1
    k2 = torch.cat([k, -j, i, r], dim=0)  # 3, 2, 1, 0
    hamilton = torch.cat([r2, i2, j2, k2], dim=1)
    assert kernel.size(1) == hamilton.size(1)
    return hamilton

class BaseModel(torch.nn.Module):
    def __init__(self, params):
        super(BaseModel, self).__init__()

        self.p		= params
        self.act	= torch.tanh
        self.bceloss	= torch.nn.BCELoss()

    def loss(self, pred, true_label):
        # return self.bceloss(pred, true_label)
        epsilon = self.p.lbl_smooth
        num_ent = self.p.num_ent
    
    # 对标签进行平滑处理
        true_label = true_label * (1.0 - epsilon) + (epsilon / num_ent)
    
    # 然后再计算 BCE 损失
        return self.bceloss(pred, true_label)
        
class CompGCNBase(BaseModel):

    def __init__(self, ent_edge_index, ent_edge_type, rel_edge_index, rel_edge_type, num_rel, params=None):
        super(CompGCNBase, self).__init__(params)
        
        
        self.p.gcn_dim		= self.p.embed_dim if self.p.gcn_layer == 1 else self.p.gcn_dim 

      
        self.ent_view_edge_index = ent_edge_index
        self.ent_view_edge_type = ent_edge_type
        self.rel_view_edge_index = rel_edge_index
        self.rel_view_edge_type = rel_edge_type
        self.device = self.ent_view_edge_index.device
        
      
        if isinstance(self.ent_view_edge_index, torch.Tensor) and self.ent_view_edge_index.dim() == 2:
            num_nodes = self.p.num_ent
            values = torch.ones(self.ent_view_edge_index.size(1), device=self.ent_view_edge_index.device)
            self.ent_adj_sparse = torch.sparse_coo_tensor(self.ent_view_edge_index, values, (num_nodes, num_nodes)).coalesce()
        else:
            self.ent_adj_sparse = None
        if isinstance(self.rel_view_edge_index, torch.Tensor) and self.rel_view_edge_index.dim() == 2:
            num_rel_nodes = self.p.num_rel
            values = torch.ones(self.rel_view_edge_index.size(1), device=self.rel_view_edge_index.device)
            self.rel_adj_sparse = torch.sparse_coo_tensor(self.rel_view_edge_index, values, (num_rel_nodes, num_rel_nodes)).coalesce()
        else:
            self.rel_adj_sparse = None

        self.type_embeddings = self.load_type_embeddings()
        self.text_embeddings = self.load_text_embeddings()
        self.relation_fingerprints = self.load_relation_fingerprints()
        
       
        if self.relation_fingerprints is not None:
            self.relation_fingerprints = self.relation_fingerprints.to(self.device)

        self.entity_embed_dim = self.p.init_dim  
        self.type_embed_dim = self.p.init_dim  #
        self.semantic_embed_dim = self.p.init_dim 
        self.total_embed_dim = self.entity_embed_dim + self.type_embed_dim + self.semantic_embed_dim  #

        self.proj_str = torch.nn.Sequential(
        torch.nn.Linear(self.entity_embed_dim, self.p.init_dim),
        torch.nn.LayerNorm(self.p.init_dim )
        )
        self.proj_top = torch.nn.Sequential(
         torch.nn.Linear(self.type_embed_dim, self.p.init_dim ),
         torch.nn.LayerNorm(self.p.init_dim )
        )
        self.proj_txt = torch.nn.Sequential(
        torch.nn.Linear(self.semantic_embed_dim, self.p.init_dim ),
        torch.nn.LayerNorm(self.p.init_dim)
        )

     
        self.gate_network = torch.nn.Sequential(
        torch.nn.Linear(self.total_embed_dim, self.total_embed_dim),
        torch.nn.ReLU(),
        torch.nn.Linear(self.total_embed_dim, 3), 
        torch.nn.Softmax(dim=-1)
          )
        

        self.init_embed = get_param((self.p.num_ent, self.entity_embed_dim))
        self.topic_embed = self.initialize_type_embeddings()
        self.text_embed = self.initialize_text_embeddings()

      
        if self.p.score_func == 'transe':
            self.init_rel = get_param((num_rel, self.total_embed_dim))
        else:
            self.init_rel = get_param((num_rel*2, self.total_embed_dim))  #



        self.input_dropout = torch.nn.Dropout(self.p.gnn_input_dropout)
        self.gcn_dropout = torch.nn.Dropout(self.p.gnn_output_dropout)

        self.conv_r = torch.nn.ModuleList()
        self.conv_e = torch.nn.ModuleList()

        for _layer in range(self.p.gcn_layer):
            self.conv_e.append(ent_adGat(self.total_embed_dim, self.p.gcn_dim, num_rel, params=self.p))
            self.conv_r.append(rel_adGat_memory(self.total_embed_dim, self.p.gcn_dim, num_rel, params=self.p))

        self.linear_ents = torch.nn.ParameterList()
        self.linear_ents_cor = torch.nn.ParameterList()

      
        self.attention_weights = torch.nn.ParameterList()
        self.attention_bias = torch.nn.ParameterList()

       
        self.gate_weights_ef = torch.nn.ParameterList()
        self.gate_weights_rf = torch.nn.ParameterList()
        self.gate_bias = torch.nn.ParameterList()

        for _layer in range(self.p.gcn_layer):
            self.linear_ents.append(Parameter(torch.FloatTensor(self.total_embed_dim // 2,self.total_embed_dim)))
            self.linear_ents_cor.append(Parameter(torch.FloatTensor(self.total_embed_dim // 4, self.total_embed_dim)))

           
            self.attention_weights.append(Parameter(torch.FloatTensor(self.p.gcn_dim, self.p.gcn_dim)))
            self.attention_bias.append(Parameter(torch.zeros(self.p.gcn_dim)))

           
            self.gate_weights_ef.append(Parameter(torch.FloatTensor(self.p.gcn_dim, self.p.gcn_dim)))
            self.gate_weights_rf.append(Parameter(torch.FloatTensor(self.p.gcn_dim, self.p.gcn_dim)))
            self.gate_bias.append(Parameter(torch.zeros(self.p.gcn_dim)))


        self.register_parameter('bias', Parameter(torch.zeros(self.p.num_ent)))

    def load_type_embeddings(self):
        type_embed_path = f'./data/{self.p.dataset}/info/entity_type_embeddings.npy'

        if os.path.exists(type_embed_path):
            try:
                type_embeddings = np.load(type_embed_path)
                return torch.FloatTensor(type_embeddings)
            except Exception as e:
                return None
        else:
            return None

    def load_relation_fingerprints(self):
        fingerprints_path = f'./data/{self.p.dataset}/info/relation_fingerprints.pt'
        
        if os.path.exists(fingerprints_path):
            try:
                fingerprints = torch.load(fingerprints_path, weights_only=True)
                return fingerprints
            except Exception as e:
                return None
        else:
            return None

    def load_text_embeddings(self):
        text_embed_path = f'./data/{self.p.dataset}/info/entity_semantic_embeddings.npy'
        
        if os.path.exists(text_embed_path):
            try:
                text_embeddings = np.load(text_embed_path)
                return torch.FloatTensor(text_embeddings)
            except Exception as e:
                return None
        else:
            return None

    def initialize_type_embeddings(self):
        if self.type_embeddings is not None:
            type_embeddings = self.type_embeddings

            if type_embeddings.size(0) != self.p.num_ent:
                if type_embeddings.size(0) > self.p.num_ent:
                    type_embeddings = type_embeddings[:self.p.num_ent]
                else:
                    padded = torch.zeros((self.p.num_ent, type_embeddings.size(1)))
                    padded[:type_embeddings.size(0)] = type_embeddings
                    type_embeddings = padded

            if type_embeddings.size(1) != self.type_embed_dim:
                if type_embeddings.size(1) > self.type_embed_dim:
                    type_embeddings = type_embeddings[:, :self.type_embed_dim]
                else:
                    padded = torch.zeros((self.p.num_ent, self.type_embed_dim))
                    padded[:, :type_embeddings.size(1)] = type_embeddings
                    type_embeddings = padded

            
            return Parameter(type_embeddings.clone())

        
        return get_param((self.p.num_ent, self.type_embed_dim))

    def initialize_text_embeddings(self):
        if self.text_embeddings is not None:
          
            if self.text_embeddings.size(1) == self.semantic_embed_dim:
                if self.text_embeddings.size(0) == self.p.num_ent:
                    return Parameter(self.text_embeddings.clone())
                else:
                    if self.text_embeddings.size(0) > self.p.num_ent:
                        text_embed = self.text_embeddings[:self.p.num_ent]
                    else:
                        text_embed = torch.zeros((self.p.num_ent, self.semantic_embed_dim))
                        text_embed[:self.text_embeddings.size(0)] = self.text_embeddings
                    return Parameter(text_embed)
            else:
                print(f"文本嵌入维度不匹配")
        
        return Parameter(torch.zeros((self.p.num_ent, self.semantic_embed_dim)))

    def get_combined_embeddings(self):
    
      
        init_embed = self.init_embed.to(self.device)
        topic_embed = self.topic_embed.to(self.device)
        text_embed = self.text_embed.to(self.device)
    
        combined_embeddings = torch.cat([init_embed, topic_embed, text_embed], dim=1)
        
        return combined_embeddings

    def attention_fusion(self, Xef, Xrf, layer_idx):
        attention_ef = torch.tanh(torch.mm(Xef, self.attention_weights[layer_idx]) + self.attention_bias[layer_idx])
        attention_rf = torch.tanh(torch.mm(Xrf, self.attention_weights[layer_idx]) + self.attention_bias[layer_idx])
        
        attention_scores = torch.stack([attention_ef, attention_rf], dim=2)  # [num_entities, gcn_dim, 2]
        attention_weights = F.softmax(attention_scores, dim=2)  # [num_entities, gcn_dim, 2]
        
        weight_ef = attention_weights[:, :, 0]  # [num_entities, gcn_dim]
        weight_rf = attention_weights[:, :, 1]  # [num_entities, gcn_dim]
        
        fused_features = weight_ef * Xef + weight_rf * Xrf
        
        return fused_features

    def gate_fusion(self, Xef, Xrf, layer_idx):
        gate_ef = torch.mm(Xef, self.gate_weights_ef[layer_idx])
        gate_rf = torch.mm(Xrf, self.gate_weights_rf[layer_idx])
        gate = torch.sigmoid(gate_ef + gate_rf + self.gate_bias[layer_idx])
        
        fused_features = gate * Xef + (1 - gate) * Xrf
        
        return fused_features

    def forward_base(self, sub, rel):
       
        r   = self.init_rel if self.p.score_func != 'transe' else torch.cat([self.init_rel, -self.init_rel], dim=0)
        
        X   = self.get_combined_embeddings()
        X   = self.input_dropout(X)
        R   = r
    
        for _layer in range(self.p.gcn_layer):
            XR = torch.cat((X, R), dim=0)
            Xef, r ,_= self.conv_e[_layer](X, self.ent_view_edge_index, self.ent_view_edge_type, r)
            fingerprints = self.relation_fingerprints 
            XRrf, r,_,_ = self.conv_r[_layer](XR, self.rel_view_edge_index, self.rel_view_edge_type, r, fingerprints)
           
            Xrf = XRrf[:X.size(0)]
            r = XRrf[X.size(0):]
            
            if self.p.combine_type == "sum":
                X = Xef + Xrf
            elif self.p.combine_type == "corr":
                X = Xef * Xrf
            elif self.p.combine_type=="concat":
                X = torch.cat([Xef, Xrf], dim=1)
                X = torch.mm(X, self.linear_ents[_layer])
            elif self.p.combine_type == "attention":
                X = self.attention_fusion(Xef, Xrf, _layer)
            elif self.p.combine_type == "gate":
                X = self.gate_fusion(Xef, Xrf, _layer)
            
            X = self.gcn_dropout(X)


        sub_emb	= torch.index_select(X, 0, sub)
        rel_emb	= torch.index_select(r, 0, rel) 

    
        return sub_emb, rel_emb, X


class TV_adGAT_info_memory_TransE(CompGCNBase):
    def __init__(self, ent_edge_index, ent_edge_type, rel_edge_index, rel_edge_type, params=None):
        super(self.__class__, self).__init__(ent_edge_index, ent_edge_type, rel_edge_index, rel_edge_type, params.num_rel, params)
        # self.drop = torch.nn.Dropout(self.p.hid_drop)
        gamma_init = torch.FloatTensor([self.p.gamma])
        self.register_parameter('gamma', Parameter(gamma_init))

    def forward(self, sub, rel):
        sub_emb, rel_emb, all_ent = self.forward_base(sub, rel)
        
        # TransE: h + r ≈ t
        obj_emb = sub_emb + rel_emb
        
        # Memory-efficient approximation of TransE distance
        # Instead of computing ||h+r-t||_1 directly, use L2 distance via dot product
        # ||a-b||_2^2 = ||a||_2^2 + ||b||_2^2 - 2<a,b>
        
        obj_norm = torch.sum(obj_emb * obj_emb, dim=1, keepdim=True)  # [batch_size, 1]
        ent_norm = torch.sum(all_ent * all_ent, dim=1, keepdim=True)  # [num_entities, 1]
        
        # Compute dot products efficiently
        dot_products = torch.mm(obj_emb, all_ent.transpose(1, 0))  # [batch_size, num_entities]
        
        # Compute squared L2 distances: ||obj_emb||^2 + ||all_ent||^2 - 2 * dot_product
        distances_sq = obj_norm + ent_norm.transpose(1, 0) - 2 * dot_products
        
        # Convert to scores (higher score = lower distance)
        x = self.gamma - torch.sqrt(torch.clamp(distances_sq, min=1e-8))  # Clamp to avoid NaN
        
        x += self.bias.expand_as(x)
        score = torch.sigmoid(x)
        
        return score

class TV_adGAT_info_memory_DistMult(CompGCNBase):
    def __init__(self, ent_edge_index, ent_edge_type, rel_edge_index, rel_edge_type, params=None):
        super(self.__class__, self).__init__(ent_edge_index, ent_edge_type, rel_edge_index, rel_edge_type, params.num_rel, params)
        # self.drop = torch.nn.Dropout(self.p.hid_drop)

    def forward(self, sub, rel):

        sub_emb, rel_emb, all_ent	= self.forward_base(sub, rel)
        obj_emb				= sub_emb * rel_emb

        x = torch.mm(obj_emb, all_ent.transpose(1, 0))
        x += self.bias.expand_as(x)

        score = torch.sigmoid(x)
        return score

class TV_adGAT_info_memory_ConvE(CompGCNBase):
    def __init__(self, ent_edge_index, ent_edge_type, rel_edge_index, rel_edge_type, params=None):
        super(self.__class__, self).__init__(ent_edge_index, ent_edge_type, rel_edge_index, rel_edge_type, params.num_rel, params)

        self.bn0		= torch.nn.BatchNorm2d(1)
        self.bn1		= torch.nn.BatchNorm2d(self.p.num_filt)
        self.bn2		= torch.nn.BatchNorm1d(self.total_embed_dim)  
        
        # self.hidden_drop	= torch.nn.Dropout(self.p.hid_drop)
        self.hidden_drop2	= torch.nn.Dropout(self.p.ConvE_hid_drop)
        self.feature_drop	= torch.nn.Dropout(self.p.ConvE_feat_drop)
        self.m_conv1		= torch.nn.Conv2d(1, out_channels=self.p.num_filt, kernel_size=(self.p.ker_sz, self.p.ker_sz), stride=1, padding=0, bias=self.p.bias)

        flat_sz_h		= int(2*self.p.k_w) - self.p.ker_sz + 1
        flat_sz_w		= self.p.k_h 	    - self.p.ker_sz + 1
        self.flat_sz		= flat_sz_h*flat_sz_w*self.p.num_filt
        self.fc			= torch.nn.Linear(self.flat_sz, self.total_embed_dim) 

    def concat(self, e1_embed, rel_embed):
        e1_embed	= e1_embed.view(-1, 1, self.total_embed_dim)  #  [batch_size, 1, total_embed_dim]
        rel_embed	= rel_embed.view(-1, 1, self.total_embed_dim)  # [batch_size, 1, total_embed_dim]
        stack_inp	= torch.cat([e1_embed, rel_embed], 1)# [batch_size, 2, total_embed_dim]
        stack_inp	= torch.transpose(stack_inp, 2, 1).reshape((-1, 1, 2*self.p.k_w, self.p.k_h))#[batch_size, 1, 2*k_w, k_h]
        #让k_w * k_h = total_embed_dim 或者2 * k_w * k_h = total_embed_dim
        return stack_inp

    def forward(self, sub, rel):

        sub_emb, rel_emb, all_ent	= self.forward_base(sub, rel)#, self.hidden_drop, self.feature_drop
        
        stk_inp				= self.concat(sub_emb, rel_emb)
        x				= self.bn0(stk_inp)
        x				= self.m_conv1(x)
        x				= self.bn1(x)
        x				= F.relu(x)
        x				= self.feature_drop(x)
        x				= x.view(-1, self.flat_sz)
        x				= self.fc(x)
        x				= self.hidden_drop2(x)
        x				= self.bn2(x)
        x				= F.relu(x)
        

        x = torch.mm(x, all_ent.transpose(1,0))
        x += self.bias.expand_as(x)

        score = torch.sigmoid(x)
        
        return score

class TV_adGAT_info_memory_InteractE(CompGCNBase):
    def __init__(self, ent_edge_index, ent_edge_type, rel_edge_index, rel_edge_type, params=None):
        super(self.__class__, self).__init__(ent_edge_index, ent_edge_type, rel_edge_index, rel_edge_type, params.num_rel, params)
        
        # InteractE specific parameters
        self.inp_drop = torch.nn.Dropout(self.p.iinp_drop)
        self.feature_map_drop = torch.nn.Dropout2d(self.p.ifeat_drop)
        self.hidden_drop = torch.nn.Dropout(self.p.ihid_drop)

        self.bn0 = torch.nn.BatchNorm2d(self.p.iperm)
        self.bn1 = torch.nn.BatchNorm2d(self.p.inum_filt * self.p.iperm)
        self.bn2 = torch.nn.BatchNorm1d(self.total_embed_dim)

        self.padding = 0
        flat_sz_h = self.p.ik_h
        flat_sz_w = 2 * self.p.ik_w
        self.flat_sz = flat_sz_h * flat_sz_w * self.p.inum_filt * self.p.iperm
        self.fc = torch.nn.Linear(self.flat_sz, self.total_embed_dim)
        
        self.register_parameter('conv_filt', Parameter(torch.zeros(self.p.inum_filt, 1, self.p.iker_sz, self.p.iker_sz)))
        xavier_normal_(self.conv_filt)

        self.chequer_perm = self.get_chequer_perm()

    def circular_padding_chw(self, batch, padding):
        upper_pad = batch[..., -padding:, :]
        lower_pad = batch[..., :padding, :]
        temp = torch.cat([upper_pad, batch, lower_pad], dim=2)

        left_pad = temp[..., -padding:]
        right_pad = temp[..., :padding]
        padded = torch.cat([left_pad, temp, right_pad], dim=3)
        return padded

    def forward(self, sub, rel):
        sub_emb, rel_emb, all_ent = self.forward_base(sub, rel)
        
        # Reshape and combine embeddings
        sub_emb = sub_emb.view(-1, self.total_embed_dim)
        rel_emb = rel_emb.view(-1, self.total_embed_dim)
        
        comb_emb = torch.cat([sub_emb, rel_emb], dim=1)
        chequer_perm = comb_emb[:, self.chequer_perm]
        stack_inp = chequer_perm.reshape((-1, self.p.iperm, 2 * self.p.ik_w, self.p.ik_h))
        
        # InteractE convolution part
        x = self.bn0(stack_inp)
        x = self.circular_padding_chw(x, self.p.iker_sz // 2)
        x = F.conv2d(x, self.conv_filt.repeat(self.p.iperm, 1, 1, 1), padding=self.padding, groups=self.p.iperm)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.feature_map_drop(x)
        x = x.view(-1, self.flat_sz)
        x = self.fc(x)
        x = self.hidden_drop(x)
        x = self.bn2(x)
        x = F.relu(x)
        
        # Score against all entities
        x = torch.mm(x, all_ent.transpose(1, 0))
        x += self.bias.expand_as(x)
        score = torch.sigmoid(x)
        
        return score


    def get_chequer_perm(self):
       
        # Ensure embed_dim is compatible with InteractE's reshaping
        if self.total_embed_dim != self.p.ik_w * self.p.ik_h:
             raise ValueError(f"Total embedding dimension ({self.total_embed_dim}) must be equal to ik_w * ik_h ({self.p.ik_w * self.p.ik_h}) for InteractE.")
        
        ent_perm = np.int32([np.random.permutation(self.p.embed_dim) for _ in range(self.p.iperm)])
        rel_perm = np.int32([np.random.permutation(self.p.embed_dim) for _ in range(self.p.iperm)])

        comb_idx = []
        for k in range(self.p.iperm):
            temp = []
            ent_idx, rel_idx = 0, 0

            for i in range(self.p.ik_h):
                for j in range(self.p.ik_w):
                    if k % 2 == 0:
                        if i % 2 == 0:
                            temp.append(ent_perm[k, ent_idx])
                            ent_idx += 1
                            temp.append(rel_perm[k, rel_idx] + self.p.embed_dim)
                            rel_idx += 1
                        else:
                            temp.append(rel_perm[k, rel_idx] + self.p.embed_dim)
                            rel_idx += 1
                            temp.append(ent_perm[k, ent_idx])
                            ent_idx += 1
                    else:
                        if i % 2 == 0:
                            temp.append(rel_perm[k, rel_idx] + self.p.embed_dim)
                            rel_idx += 1
                            temp.append(ent_perm[k, ent_idx])
                            ent_idx += 1
                        else:
                            temp.append(ent_perm[k, ent_idx])
                            ent_idx += 1
                            temp.append(rel_perm[k, rel_idx] + self.p.embed_dim)
                            rel_idx += 1

            comb_idx.append(temp)

        chequer_perm = torch.LongTensor(np.int32(comb_idx)).to(self.device)
        return chequer_perm
