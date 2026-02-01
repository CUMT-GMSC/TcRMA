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
        self.act 		= act #激活函数
        self.device		= None

        # 使用torch.nn.Linear替代get_param
        self.w_in = torch.nn.Linear(in_channels, out_channels).cuda()
        self.w_out = torch.nn.Linear(in_channels, out_channels).cuda()
        self.w_rel = torch.nn.Linear(in_channels, out_channels).cuda()
        self.w_att = torch.nn.Linear(3*in_channels, out_channels).cuda()
        self.a = torch.nn.Linear(out_channels, 1, bias=False).cuda()

        self.drop_ratio = self.p.inres_drop 
        self.drop = torch.nn.Dropout(self.drop_ratio, inplace=False)
        
        self.bn = torch.nn.BatchNorm1d(out_channels).to(torch.device("cuda"))#批归一化层
        self.res_w = torch.nn.Linear(in_channels, out_channels, bias=False)
        self.activation = torch.nn.Tanh()
        self.leaky_relu = torch.nn.LeakyReLU(negative_slope=0.2, inplace=True)
        
        # 添加用于动态聚合的注意力层
        self.gate_w_in = torch.nn.Linear(out_channels, out_channels).cuda()
        self.gate_w_loop = torch.nn.Linear(out_channels, out_channels).cuda()
        self.gate_att = torch.nn.Linear(2*out_channels, 2, bias=False).cuda()

        # Memory Network 组件（只用于关系）
        
        self.finger_scale = nn.Parameter(torch.ones(1)) # 可学习的缩放系数
        self.temperature = nn.Parameter(torch.ones(1) * 0.5) # 可学习的温度系数
        # 关系记忆库
        # self.memory_bank = torch.nn.Parameter(torch.Tensor(self.num_memories, in_channels))
        # torch.nn.init.xavier_uniform_(self.memory_bank)
      
        self.query_proj = nn.Linear(in_channels, in_channels).cuda()
        self.key_proj   = nn.Linear(in_channels, in_channels).cuda()

        ## 融合门控
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
        # 专门初始化 mem_gate 的 bias 为负数 (例如 -2.0)
        # Sigmoid(-2.0) ≈ 0.12，Sigmoid(-5.0) ≈ 0.006
        # 这样初始 alpha 会很小，模型先学好 GAT 本身，再慢慢引入 Memory
        # nn.init.constant_(self.mem_gate.bias, -2.0)

    def forward(self, x, edge_index, edge_type, rel_embed, fingerprints,adj_sparse=None, pre_alpha=None): 
        if self.device is None:
            self.device = edge_index.device
        num_ent = x.size(0)
     
        # fingerprints: [num_rels, 2*num_topics]
        # norm_finger: [num_rels, 2*num_topics]
        norm_finger = F.normalize(fingerprints, p=2, dim=1)#p=2使用L2范数归一化
        # struct_bias: [num_rels, num_rels] -> 值在 [0, 1] 之间
        struct_bias = torch.matmul(norm_finger, norm_finger.t())

        q = self.query_proj(rel_embed) # [num_rels, dim]
        k = self.key_proj(rel_embed)   # [num_rels, dim]
        
        # 语义相似度
        scores = torch.matmul(q, k.t()) / (math.sqrt(self.in_channels) * F.softplus(self.temperature))
        
        # 融合偏置：语义分 + (权重 * 结构先验)
        total_scores = scores + self.finger_scale * struct_bias
        
        attn_weights = F.softmax(total_scores, dim=1)
        
        # 融合了全局相关关系的表征
        global_rel_info = torch.matmul(attn_weights, rel_embed) # [num_rels, dim]

        combined = torch.cat([rel_embed, global_rel_info], dim=1)
        alpha = torch.sigmoid(self.mem_gate(combined))
        rel_embed_enhanced = self.layer_norm_mem(rel_embed + alpha * global_rel_info)

        in_res = self.propagate(edge_index=edge_index, x=x, edge_type=edge_type, rel_embed=rel_embed_enhanced, pre_alpha=pre_alpha)
        loop_res = self.res_w(x)
        
        # 通过注意力机制动态聚合in_res和loop_res
        in_res_dropped = self.drop(in_res)
        loop_res_dropped = self.drop(loop_res)
        
        # 计算注意力权重
        gate_in = self.gate_w_in(in_res_dropped)
        gate_loop = self.gate_w_loop(loop_res_dropped)
        gate_concat = torch.cat([gate_in, gate_loop], dim=1)
        gate_weights = torch.softmax(self.gate_att(gate_concat), dim=1)
        
        # 动态聚合
        out = gate_weights[:, 0:1] * in_res_dropped + gate_weights[:, 1:2] * loop_res_dropped

        if self.bias:
            out = out + self.bias_value
        
        out = self.bn(out)
        out = self.activation(out)
        # 对增强后的关系嵌入进行变换输出
        rel_out = self.w_rel(rel_embed_enhanced)

        return out, rel_out, self.alpha.detach(), attn_weights

     #消息传递函数，使用注意力机制
    def message(self, x_i, x_j, edge_type, rel_embed, ptr, index, size_i, pre_alpha):
        rel_embed = torch.index_select(rel_embed, 0, edge_type)
        # 直接使用x_j，不进行关系变换
        num_edge = x_j.size(0)//2

        in_message = x_j[:num_edge]
        out_message = x_j[num_edge:]
                
        trans_in = self.w_in(in_message)
        trans_out = self.w_out(out_message)
        
        out = torch.cat((trans_in, trans_out), dim=0)
        
        b = self.leaky_relu(self.w_att(torch.cat((x_i, rel_embed, x_j), dim=1))).cuda()#注意力计算
        b = self.a(b).float()#再经过一个线性层 self.a（输出为 1 维），将注意力特征压缩为一个标量
        alpha = softmax(b, index, ptr, size_i)#对所有入边的注意力分数做 softmax，归一化为概率分布，得到每条边的注意力权重 α index, ptr, size_i 用于指定 softmax 的分组（即每个节点的入边）
        alpha = F.dropout(alpha, p=self.drop_ratio, training=self.training, inplace=False)
        #如果有前一层的注意力分布 pre_alpha 且 β 不为 0，则用 β 做加权融合（残差机制），否则直接用当前 α。
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
