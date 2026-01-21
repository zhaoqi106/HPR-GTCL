import torch
import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):
    def __init__(self, in_dim, hidden_dims, activation=nn.GELU, dropout=0.0):
        super().__init__()
        layers = []
        dims = [in_dim] + hidden_dims
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            layers.append(nn.LayerNorm(dims[i + 1]))
            layers.append(activation())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

class GraphAwareAttention(nn.Module):
    def __init__(self, dim, heads=4, dropout=0.0, use_graph_bias=True):
        super().__init__()
        assert dim % heads == 0
        self.heads = heads
        self.hd = dim // heads
        self.scale = self.hd ** -0.5
        self.use_graph_bias = use_graph_bias

        self.qkv = nn.Linear(dim, dim * 3)
        self.out = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)

        if use_graph_bias:
            self.graph_bias = nn.Parameter(torch.zeros(1, 1, 1))

    def forward(self, x, adj_matrix=None, attn_mask=None):
        N, D = x.shape
        qkv = self.qkv(x)
        q, k, v = torch.chunk(qkv, 3, dim=-1)

        def split_heads(t):
            return t.view(N, self.heads, self.hd).permute(1, 0, 2)

        qh, kh, vh = split_heads(q), split_heads(k), split_heads(v)
        scores = torch.matmul(qh, kh.transpose(-2, -1)) * self.scale

        if adj_matrix is not None and self.use_graph_bias:
            if adj_matrix.is_sparse:
                adj_matrix = adj_matrix.to_dense()
            graph_bias = self.graph_bias * adj_matrix.unsqueeze(0)
            graph_bias = graph_bias.expand(self.heads, -1, -1)
            scores = scores + graph_bias

        if attn_mask is not None:
            mask = (attn_mask == 0).unsqueeze(0).unsqueeze(0)
            scores = scores.masked_fill(mask, float('-1e9'))

        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        out = torch.matmul(attn, vh)

        if out.dim() != 3:
            out = out.contiguous().view(self.heads, N, self.hd)

        out = out.permute(1, 0, 2).contiguous().view(N, D)
        return self.out(out)

class GraphTransformerLayer(nn.Module):
    def __init__(self, dim, heads=4, dropout=0.1, use_graph_bias=True):
        super().__init__()
        self.attention = GraphAwareAttention(dim, heads, dropout, use_graph_bias)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, adj_matrix=None, attn_mask=None):
        attn_out = self.attention(self.norm1(x), adj_matrix, attn_mask)
        x = x + attn_out
        ffn_out = self.ffn(self.norm2(x))
        x = x + ffn_out
        return x

class GraphPropLayer(nn.Module):
    def __init__(self, dim, dropout=0.1):
        super().__init__()
        self.lin_self = nn.Linear(dim, dim)
        self.lin_neigh = nn.Linear(dim, dim)
        self.norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, h, adj):
        if adj.is_sparse:
            adj = adj.to_dense()
        neigh = torch.matmul(adj, h)
        out = self.lin_self(h) + self.lin_neigh(neigh)
        out = F.gelu(out)
        out = self.dropout(out)
        return self.norm(h + out)

class CrossGraphAttention(nn.Module):
    def __init__(self, dim, heads=4, dropout=0.1):
        super().__init__()
        assert dim % heads == 0
        self.heads = heads
        self.hd = dim // heads
        self.scale = self.hd ** -0.5
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.out = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)

    def _split_heads(self, t):
        N, D = t.shape
        return t.view(N, self.heads, self.hd).permute(1, 0, 2)

    def forward(self, query, key_value):
        q = self._split_heads(self.q(query))
        k = self._split_heads(self.k(key_value))
        v = self._split_heads(self.v(key_value))

        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = out.permute(1, 0, 2).contiguous().view(query.shape[0], -1)
        return self.out(out)

class GatedFusion(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.Sigmoid()
        )

    def forward(self, a, b):
        g = self.gate(torch.cat([a, b], dim=-1))
        return g * a + (1.0 - g) * b

class Projector(nn.Module):
    def __init__(self, in_dim, proj_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, in_dim),
            nn.ReLU(),
            nn.Linear(in_dim, proj_dim)
        )

    def forward(self, x):
        return F.normalize(self.net(x), dim=-1)

class HPRGTCL(nn.Module):
    def __init__(self,
                 symptom_feat_dim: int,
                 herb_feat_dim: int,
                 embed_dim: int = 256,
                 transformer_heads: int = 4,
                 transformer_layers: int = 2,
                 prop_steps: int = 2,
                 dropout: float = 0.2,
                 proj_dim: int = 128,
                 use_graph_bias: bool = True):
        super().__init__()

        self.sym_input = MLP(symptom_feat_dim, [embed_dim], dropout=dropout)
        self.herb_input = MLP(herb_feat_dim, [embed_dim], dropout=dropout)

        self.sym_transformers = nn.ModuleList([
            GraphTransformerLayer(embed_dim, transformer_heads, dropout, use_graph_bias)
            for _ in range(transformer_layers)
        ])
        self.herb_transformers = nn.ModuleList([
            GraphTransformerLayer(embed_dim, transformer_heads, dropout, use_graph_bias)
            for _ in range(transformer_layers)
        ])

        self.sym_prop = nn.ModuleList([
            GraphPropLayer(embed_dim, dropout=dropout) for _ in range(prop_steps)
        ])
        self.herb_prop = nn.ModuleList([
            GraphPropLayer(embed_dim, dropout=dropout) for _ in range(prop_steps)
        ])

        self.sym2herb_attn = CrossGraphAttention(embed_dim, heads=transformer_heads, dropout=dropout)
        self.herb2sym_attn = CrossGraphAttention(embed_dim, heads=transformer_heads, dropout=dropout)

        self.sym_fusion = GatedFusion(embed_dim)
        self.herb_fusion = GatedFusion(embed_dim)

        self.predictor_mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        self.sym_projector = Projector(embed_dim, proj_dim)
        self.herb_projector = Projector(embed_dim, proj_dim)

    def forward(self,
                batch_symptom_ids_list,
                symptom_feat,
                herb_feat,
                sym_adj=None,
                herb_adj=None):

        device = symptom_feat.device
        sym_h = self.sym_input(symptom_feat)
        herb_h = self.herb_input(herb_feat)

        for transformer in self.sym_transformers:
            sym_h = transformer(sym_h, sym_adj)

        for transformer in self.herb_transformers:
            herb_h = transformer(herb_h, herb_adj)

        if sym_adj is not None:
            for p in self.sym_prop:
                sym_h = p(sym_h, sym_adj)
        if herb_adj is not None:
            for p in self.herb_prop:
                herb_h = p(herb_h, herb_adj)

        sym_to_herb = self.sym2herb_attn(sym_h, herb_h)
        herb_to_sym = self.herb2sym_attn(herb_h, sym_h)

        sym_h = self.sym_fusion(sym_h, sym_to_herb)
        herb_h = self.herb_fusion(herb_h, herb_to_sym)

        batch_agg = []
        for s_list in batch_symptom_ids_list:
            if len(s_list) == 0:
                agg = torch.zeros(sym_h.size(1), device=device)
            else:
                idx = torch.tensor(s_list, dtype=torch.long, device=device)
                agg = sym_h[idx].mean(dim=0)
            batch_agg.append(agg)
        batch_agg = torch.stack(batch_agg, dim=0)

        fused = self.predictor_mlp(batch_agg)
        logits = torch.matmul(fused, herb_h.t())

        return logits, sym_h, herb_h