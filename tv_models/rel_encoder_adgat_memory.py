from helper import *
# from tv_models.message_passing import MessagePassing
import scipy.sparse as sp
import torch.nn as nn
from torch_geometric.utils import softmax
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
import math

class rel_adGat_memory(MessagePassing):
    def __init__(self, in_channels, out_channels,num_rels, act=lambda x:x, params=None, bias=None, beta=None):
        super(self.__class__, self).__init__(aggr="add")

        self.p 			= params
        self.in_channels	= in_channels
        self.out_channels	= out_channels
        self.num_rels 		= num_rels
       

        self.beta = beta
        self.bias = bias
        self.act 		= act 
        self.device		= None

      
        self.w_in = torch.nn.Linear(in_channels, out_channels).cuda()
        self.w_out = torch.nn.Linear(in_channels, out_channels).cuda()
        self.w_rel = torch.nn.Linear(in_channels, out_channels).cuda()
        self.w_att = torch.nn.Linear(3*in_channels, out_channels).cuda()
        self.a = torch.nn.Linear(out_channels, 1, bias=False).cuda()

        self.drop_ratio = self.p.inres_drop 
        self.drop = torch.nn.Dropout(self.drop_ratio, inplace=False)
        
        self.bn = torch.nn.BatchNorm1d(out_channels).to(torch.device("cuda"))
        self.res_w = torch.nn.Linear(in_channels, out_channels, bias=False)
        self.activation = torch.nn.Tanh()
        self.leaky_relu = torch.nn.LeakyReLU(negative_slope=0.2, inplace=True)
        
       
        self.gate_w_in = torch.nn.Linear(out_channels, out_channels).cuda()
        self.gate_w_loop = torch.nn.Linear(out_channels, out_channels).cuda()
        self.gate_att = torch.nn.Linear(2*out_channels, 2, bias=False).cuda()

    
        
        self.finger_scale = nn.Parameter(torch.ones(1)) 
        self.temperature = nn.Parameter(torch.ones(1) * 0.5) 
       
        # self.memory_bank = torch.nn.Parameter(torch.Tensor(self.num_memories, in_channels))
        # torch.nn.init.xavier_uniform_(self.memory_bank)
      
        self.query_proj = nn.Linear(in_channels, in_channels).cuda()
        self.key_proj   = nn.Linear(in_channels, in_channels).cuda()

       
        self.mem_gate = torch.nn.Linear(in_channels * 2, 1).cuda()
        self.layer_norm_mem = torch.nn.LayerNorm(in_channels).cuda()
     

        if self.p.bias: 
            self.register_parameter('bias_value', torch.nn.Parameter(torch.zeros(out_channels)))
        self.init_weight()

    def init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.fill_(0.0)
      

    def forward(self, x, edge_index, edge_type, rel_embed, fingerprints,adj_sparse=None, pre_alpha=None): 
        if self.device is None:
            self.device = edge_index.device
        num_ent = x.size(0)
      
        # fingerprints: [num_rels, 2*num_topics]
       
        # norm_finger: [num_rels, 2*num_topics]
        norm_finger = F.normalize(fingerprints, p=2, dim=1)
  
        struct_bias = torch.matmul(norm_finger, norm_finger.t())

        q = self.query_proj(rel_embed) # [num_rels, dim]
        k = self.key_proj(rel_embed)   # [num_rels, dim]
        
      
        scores = torch.matmul(q, k.t()) / (math.sqrt(self.in_channels) * F.softplus(self.temperature))
        
        total_scores = scores + self.finger_scale * struct_bias
        
        attn_weights = F.softmax(total_scores, dim=1)
        
       
        global_rel_info = torch.matmul(attn_weights, rel_embed) # [num_rels, dim]

        combined = torch.cat([rel_embed, global_rel_info], dim=1)
        alpha = torch.sigmoid(self.mem_gate(combined))
        rel_embed_enhanced = self.layer_norm_mem(rel_embed + alpha * global_rel_info)

        in_res = self.propagate(edge_index=edge_index, x=x, edge_type=edge_type, rel_embed=rel_embed_enhanced, pre_alpha=pre_alpha)
        loop_res = self.res_w(x)
        
       
        in_res_dropped = self.drop(in_res)
        loop_res_dropped = self.drop(loop_res)
        
       
        gate_in = self.gate_w_in(in_res_dropped)
        gate_loop = self.gate_w_loop(loop_res_dropped)
        gate_concat = torch.cat([gate_in, gate_loop], dim=1)
        gate_weights = torch.softmax(self.gate_att(gate_concat), dim=1)
        
       
        out = gate_weights[:, 0:1] * in_res_dropped + gate_weights[:, 1:2] * loop_res_dropped

        if self.bias:
            out = out + self.bias_value
        
        out = self.bn(out)
        out = self.activation(out)
       
        rel_out = self.w_rel(rel_embed_enhanced)

        return out, rel_out, self.alpha.detach(), attn_weights

    
    def message(self, x_i, x_j, edge_type, rel_embed, ptr, index, size_i, pre_alpha):
        rel_embed = torch.index_select(rel_embed, 0, edge_type)
       
        num_edge = x_j.size(0)//2

        in_message = x_j[:num_edge]
        out_message = x_j[num_edge:]
                
        trans_in = self.w_in(in_message)
        trans_out = self.w_out(out_message)
        
        out = torch.cat((trans_in, trans_out), dim=0)
        
        b = self.leaky_relu(self.w_att(torch.cat((x_i, rel_embed, x_j), dim=1))).cuda()
        b = self.a(b).float()
        alpha = softmax(b, index, ptr, size_i)
        alpha = F.dropout(alpha, p=self.drop_ratio, training=self.training, inplace=False)
      
        if pre_alpha!=None and self.beta != 0:
            self.alpha = alpha*(1-self.beta) + pre_alpha*(self.beta)
        else:
            self.alpha = alpha
        out = out * alpha.view(-1,1)

        return out
    def update(self, aggr_out):
        return aggr_out

    def __repr__(self):
        return '{}({}, {}, num_rels={})'.format(
            self.__class__.__name__, self.in_channels, self.out_channels, self.num_rels)
