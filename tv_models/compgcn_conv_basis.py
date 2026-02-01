from helper import *
from gnn_model.message_passing import MessagePassing

#基础向量分解部分 本质是进行参数压缩
class CompGCNConvBasis(MessagePassing):
	def __init__(self, in_channels, out_channels, num_rels, num_bases, act=lambda x:x, cache=True, params=None):
		super(self.__class__, self).__init__()

		self.p 			= params
		self.in_channels	= in_channels
		self.out_channels	= out_channels
		self.num_rels 		= num_rels
		self.num_bases 		= num_bases#基础向量数量
		self.act 		= act
		self.device		= None
		self.cache 		= cache			# Should be False for graph classification tasks
        #定义三种方向上的卷积核 自环 入边  出边
		self.w_loop		= get_param((in_channels, out_channels));
		self.w_in		= get_param((in_channels, out_channels));
		self.w_out		= get_param((in_channels, out_channels));

		self.rel_basis 		= get_param((self.num_bases, in_channels))#基础向量
		self.rel_wt 		= get_param((self.num_rels*2, self.num_bases))#正向+反向关系的组合
		self.w_rel 		= get_param((in_channels, out_channels))#用于更新关系向量
		self.loop_rel 		= get_param((1, in_channels));

		self.drop		= torch.nn.Dropout(self.p.dropout)
		self.bn			= torch.nn.BatchNorm1d(out_channels)
		
		self.in_norm, self.out_norm,
		self.in_index, self.out_index,
		self.in_type, self.out_type,
		self.loop_index, self.loop_type = None, None, None, None, None, None, None, None

		if self.p.bias: self.register_parameter('bias', Parameter(torch.zeros(out_channels)))

	def forward(self, x, edge_index, edge_type, edge_norm=None, rel_embed=None):
		#edge_index：边的起止节点 
		if self.device is None:
			self.device = edge_index.device
        #利用基础向量生成所有关系的嵌入；末尾追加一个 loop 关系。
		rel_embed = torch.mm(self.rel_wt, self.rel_basis)
		rel_embed = torch.cat([rel_embed, self.loop_rel], dim=0)

		num_edges = edge_index.size(1) // 2 #边的数量，假设每条边有两个方向（入边和出边）
		num_ent   = x.size(0)
         # 如果没有缓存或归一化系数未计算，则计算入边、出边和自环的索引和类型
		if not self.cache or self.in_norm == None:
			self.in_index, self.out_index = edge_index[:, :num_edges], edge_index[:, num_edges:]
			self.in_type,  self.out_type  = edge_type[:num_edges], 	 edge_type [num_edges:]

			self.loop_index  = torch.stack([torch.arange(num_ent), torch.arange(num_ent)]).to(self.device)
			self.loop_type   = torch.full((num_ent,), rel_embed.size(0)-1, dtype=torch.long).to(self.device)

			self.in_norm     = self.compute_norm(self.in_index,  num_ent)
			self.out_norm    = self.compute_norm(self.out_index, num_ent)
		#三个方向的信息传播结果
		in_res		= self.propagate('add', self.in_index,   x=x, edge_type=self.in_type,   rel_embed=rel_embed, edge_norm=self.in_norm, 	mode='in')
		loop_res	= self.propagate('add', self.loop_index, x=x, edge_type=self.loop_type, rel_embed=rel_embed, edge_norm=None, 		mode='loop')
		out_res		= self.propagate('add', self.out_index,  x=x, edge_type=self.out_type,  rel_embed=rel_embed, edge_norm=self.out_norm,	mode='out')
		out		= self.drop(in_res)*(1/3) + self.drop(out_res)*(1/3) + loop_res*(1/3)#三个方向的结果进行加权平均，作为最终的输出

		if self.p.bias: out = out + self.bias
		if self.b_norm: out = self.bn(out)

		return self.act(out), torch.matmul(rel_embed, self.w_rel)[:-1]
    #关系变换方法，支持三种操作模式：相关、减法和乘法
	def rel_transform(self, ent_embed, rel_embed):
		if   self.p.opn == 'corr': 	trans_embed  = ccorr(ent_embed, rel_embed)
		elif self.p.opn == 'sub': 	trans_embed  = ent_embed - rel_embed
		elif self.p.opn == 'mult': 	trans_embed  = ent_embed * rel_embed
		else: raise NotImplementedError

		return trans_embed

	def message(self, x_j, edge_type, rel_embed, edge_norm, mode):
		weight 	= getattr(self, 'w_{}'.format(mode))#动态选择当前卷积方向的权重矩阵
		rel_emb = torch.index_select(rel_embed, 0, edge_type)#对于每一条边，根据它的 edge_type 从 rel_embed 中取出该关系类型的嵌入；
		xj_rel  = self.rel_transform(x_j, rel_emb)#实体 和关系的组合 组合方式rel_transform()
		out	= torch.mm(xj_rel, weight)#将组合后的向量乘以方向对应的权重矩阵，完成一次线性映射

		return out if edge_norm is None else out * edge_norm.view(-1, 1)

	def update(self, aggr_out):
		return aggr_out
    #边归一化
	def compute_norm(self, edge_index, num_ent):
		row, col	= edge_index
		edge_weight 	= torch.ones_like(row).float()
		deg		= scatter_add( edge_weight, row, dim=0, dim_size=num_ent)	# Summing number of weights of the edges [Computing out-degree] [Should be equal to in-degree (undireted graph)]
		deg_inv		= deg.pow(-0.5)							# D^{-0.5}
		deg_inv[deg_inv	== float('inf')] = 0
		norm		= deg_inv[row] * edge_weight * deg_inv[col]			# D^{-0.5}

		return norm

	def __repr__(self):
		return '{}({}, {}, num_rels={})'.format(
			self.__class__.__name__, self.in_channels, self.out_channels, self.num_rels)
