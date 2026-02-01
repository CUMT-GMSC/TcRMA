from helper import *
# from gnn_model.message_passing import MessagePassing
import scipy.sparse as sp
import torch.nn as nn
from torch_geometric.utils import  softmax
from torch_geometric.nn import MessagePassing
# from tv_models.message_passing import MessagePassing

class ent_adGat(MessagePassing):
	def __init__(self, in_channels, out_channels, num_rels, act=lambda x:x, params=None, bias=None, beta=None):
		super(ent_adGat, self).__init__(aggr="add")

		self.p 			= params
		self.in_channels	= in_channels
		self.out_channels	= out_channels
		self.num_rels 		= num_rels

		self.beta = beta
		self.bias = bias
		self.act 		= act #激活函数
		self.device		= None

		# self.w_loop = torch.nn.Linear(in_channels, out_channels).cuda()
		self.w_in = torch.nn.Linear(in_channels, out_channels).cuda()
		self.w_out = torch.nn.Linear(in_channels, out_channels).cuda()
		self.w_rel = torch.nn.Linear(in_channels, out_channels).cuda()
		self.w_att = torch.nn.Linear(3*in_channels, out_channels).cuda()
		self.a = torch.nn.Linear(out_channels, 1, bias=False).cuda()
        # self.loop_rel = torch.nn.Parameter(torch.Tensor(1, in_channels)).cuda()
        # torch.nn.init.xavier_uniform_(self.loop_rel)


		self.drop_ratio = self.p.inres_drop 
		self.drop = torch.nn.Dropout(self.drop_ratio, inplace=False)
		
		self.bn			= torch.nn.BatchNorm1d(out_channels).to(torch.device("cuda"))#批归一化层
		self.res_w = torch.nn.Linear(in_channels, out_channels, bias=False)
		self.activation = torch.nn.Tanh() #torch.nn.Tanh()
		self.leaky_relu = torch.nn.LeakyReLU(negative_slope=0.2, inplace=True)
		
		# 添加用于动态聚合的注意力层
		self.gate_w_in = torch.nn.Linear(out_channels, out_channels).cuda()
		self.gate_w_loop = torch.nn.Linear(out_channels, out_channels).cuda()
		self.gate_att = torch.nn.Linear(2*out_channels, 2, bias=False).cuda()
		
		# 关系嵌入维度投影层（用于处理rel_embed维度不匹配的情况）
		# 在message中，rel_embed可能来自关系编码器的输出（out_channels），需要投影到in_channels
		self.rel_proj = torch.nn.Linear(out_channels, in_channels, bias=False).cuda()

		if self.p.bias: 
			self.register_parameter('bias_value', torch.nn.Parameter(torch.zeros(out_channels)))
		self.init_weight()

	def init_weight(self):
		for m in self.modules():
			if isinstance(m, nn.Linear):
				torch.nn.init.xavier_uniform_(m.weight.data)
				if m.bias is not None:
					m.bias.data.fill_(0.0)

	def forward(self, x, edge_index, edge_type, rel_embed, adj_sparse=None, pre_alpha=None): 
		if self.device is None:
			self.device = edge_index.device

		num_ent = x.size(0)
        # loop_index = torch.stack([torch.arange(num_ent), torch.arange(num_ent)]).cuda()
        # loop_type   = torch.full((num_ent,), rel_emb.size(0)-1, dtype=torch.long).cuda()

		in_res = self.propagate(edge_index=edge_index, x=x, edge_type=edge_type, rel_embed=rel_embed, pre_alpha=pre_alpha)
        # loop_res = self.propagate(edge_index=loop_index, x=x, edge_type=loop_type, rel_emb=rel_emb, pre_alpha=pre_alpha, mode="loop")
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
		
		if rel_embed.size(1) == self.in_channels:
			processed_rel = self.w_rel(rel_embed)
		else:
			processed_rel = rel_embed

		return out, processed_rel, self.alpha.detach()
	
	 #消息传递函数，根据模式选择权重，应用关系变换和归一化
	def message(self,x_i, x_j, edge_type, rel_embed, ptr, index, size_i, pre_alpha):
		rel_embed = torch.index_select(rel_embed, 0, edge_type)
		
		# 如果关系嵌入维度与实体嵌入维度不匹配，进行投影
		if rel_embed.size(1) != x_j.size(1):
			rel_embed = self.rel_proj(rel_embed)
		
		xj_rel = self.rel_transform(x_j, rel_embed)
		num_edge = xj_rel.size(0)//2

		in_message = xj_rel[:num_edge]
		out_message = xj_rel[num_edge:]
                
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
	
    #关系变换方法，支持三种操作模式：相关、减法和乘法
	def rel_transform(self, ent_embed, rel_embed):
		if   self.p.opn == 'corr': 	
			trans_embed  = ccorr(ent_embed, rel_embed)
		elif self.p.opn == 'sub': 	
			trans_embed  = ent_embed - rel_embed
		elif self.p.opn == 'mult': 	
			trans_embed  = ent_embed * rel_embed
		elif self.p.opn == "corr_new":
			trans_embed = ccorr_new(ent_embed, rel_embed)
		elif self.p.opn == "conv":
			trans_embed = cconv(ent_embed, rel_embed)
		elif self.p.opn == "conv_new":
			trans_embed = cconv_new(ent_embed, rel_embed)
		elif self.p.opn == 'cross':
			trans_embed = ent_embed * rel_embed + ent_embed
		elif self.p.opn == "corr_plus":
			trans_embed = ccorr_new(ent_embed, rel_embed) + ent_embed
		elif self.p.opn == "rotate":
			trans_embed = rotate(ent_embed, rel_embed)
		else: raise NotImplementedError

		return trans_embed
   

	def update(self, aggr_out):
		return aggr_out

	def __repr__(self):
		return '{}({}, {}, num_rels={})'.format(
			self.__class__.__name__, self.in_channels, self.out_channels, self.num_rels)
