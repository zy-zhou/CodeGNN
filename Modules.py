import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from torch_geometric.nn import MessagePassing, global_max_pool
from torch_geometric.data import Batch
from Utils import sequence_mask, tuple_map
from Data import pad
import math
from collections import namedtuple

word_embed = 512
lstm_units = 512
dropout = 0.3

class KeyQueryAttention(nn.Module):
    ''' Multiplicative key-query attention.
        shape of key: [batch_size, key_length, units]
        shape of query: [batch_size, query_length, units]
        shape of alignments: [batch_size, query_length, key_length]'''
    def __init__(self, units, query_dim=None, key_dim=None):
        super(KeyQueryAttention, self).__init__()
        self.Wq = nn.Linear(query_dim or units, units, bias=False)
        self.Wk = nn.Linear(key_dim or units, units, bias=False)
    
    def score(self, query, key):
        query = self.Wq(query)
        key = self.Wk(key)
        output = torch.bmm(query, key.transpose(1, 2))
        return output
    
    def forward(self, query, key, key_lengths=None):
        score = self.score(query, key) # batch_size * query_length * key_length
        if key_lengths is not None:
            score_mask = sequence_mask(key_lengths, key.shape[1]).unsqueeze(1)
            score.masked_fill_(~score_mask, float('-inf'))
        
        alignments = F.softmax(score, 2)
        contexts = torch.bmm(alignments, key)
        return contexts, alignments

class GlobalAttention(nn.Module):
    ''' Multiplicative and additive global attention.
        shape of query: [batch_size, units]
        shape of key: [batch_size, max_steps, key_dim]
        shape of context: [batch_size, units]
        shape of alignments: [batch_size, max_steps]
        style should be either "add" or "mul"'''
    def __init__(self, units, key_dim=None, style='mul', scale=True):
        super(GlobalAttention, self).__init__()
        self.style = style
        self.scale = scale
        key_dim = key_dim or units
            
        self.Wk = nn.Linear(key_dim, units, bias=False)
        if self.style == 'mul':
            if self.scale:
                self.v = nn.Parameter(torch.tensor(1.))
        elif self.style == 'add':
            self.Wq = nn.Linear(units, units)
            self.v = nn.Parameter(torch.ones(units))
        else:
            raise ValueError(str(style) + ' is not an appropriate attention style.')
            
    def score(self, query, key):
        query = query.unsqueeze(1) # batch_size * 1 * units
        key = self.Wk(key)
        
        if self.style == 'mul':
            output = torch.bmm(query, key.transpose(1, 2))
            output = output.squeeze(1)
            if self.scale:
                output = self.v * output
        else:
            output = torch.sum(self.v * torch.tanh(self.Wq(query) + key), 2)
        return output
    
    def forward(self, query, key, memory_lengths=None, custom_mask=None):
        score = self.score(query, key) # batch_size * max_steps
        if memory_lengths is not None:
            if type(memory_lengths) in {list, tuple}:
                memory_lengths = memory_lengths[0]
            score_mask = sequence_mask(memory_lengths, key.shape[1])
            score.masked_fill_(~score_mask, float('-inf'))
        elif custom_mask is not None:
            score.masked_fill_(~custom_mask, float('-inf'))
        
        alignments = F.softmax(score, 1)
        context = torch.bmm(alignments.unsqueeze(1), key) # batch_size * 1 * units
        context = context.squeeze(1)
        return context, alignments

class RNNEncoder(nn.Module):
    def __init__(self, field=None, embed_dim=word_embed, units=lstm_units, num_layers=1,
                 bidirectional=True, dropout=dropout):
        super(RNNEncoder, self).__init__()
        self.field = field
        if field is not None:
            vocab_size = len(field.vocab)
            pad_id = field.vocab.stoi[pad]
            self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_id)
        
        self.lstm = nn.LSTM(embed_dim, units, num_layers, batch_first=True,
                            bidirectional=bidirectional)
        self.units = units
        self.num_layers = num_layers
        self.directions = int(bidirectional) + 1
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, inputs, src_lengths=None):
        if self.field is not None:
            inputs = self.embedding(inputs)
        x = self.dropout(inputs)
        if src_lengths is not None:
            x = nn.utils.rnn.pack_padded_sequence(
                    x, src_lengths.to('cpu'), batch_first=True, enforce_sorted=False)
        output_seqs, states = self.lstm(x)
        
        if self.directions > 1:
            final_idx = torch.arange(self.num_layers, device=inputs.device) * self.directions + 1
            states = tuple_map(lambda x: x.index_select(0, final_idx), states)
        
        if src_lengths is not None:
            output_seqs, _ = nn.utils.rnn.pad_packed_sequence(output_seqs, batch_first=True)
        return output_seqs, states

class StaticMsgPass(MessagePassing):
    def __init__(self):
        super(StaticMsgPass, self).__init__(aggr='add')

    def forward(self, x, edge_index):
        out = self.propagate(edge_index=edge_index, x=x)
        return out

class DynamicMsgPass(MessagePassing):
    def __init__(self, num_edge_attr, hidden_dim):
        super(DynamicMsgPass, self).__init__(aggr='add')
        self.WV = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.WF = nn.Linear(num_edge_attr, hidden_dim, bias=False)

    def forward(self, x, edge_index, edge_attr, norm):
        out = self.propagate(edge_index, x=x, edge_attr=edge_attr, norm=norm)
        return out

    def message(self, x_j, edge_attr, norm):
        h = self.WV(x_j) # edge_num * hidden_dim
        e = self.WF(edge_attr)
        return norm * (h + e)

class Fusion(nn.Module):
    def __init__(self, hidden_dim):
        super(Fusion, self).__init__()
        self.Wz = nn.Linear(4 * hidden_dim, hidden_dim)

    def forward(self, a, b):
        z = torch.sigmoid(self.Wz(torch.cat([a, b, a * b, a - b], dim=-1)))
        fuse = z * a + (1 - z) * b
        return fuse

class NodeEmbedding(nn.Module):
    def __init__(self, vocab_size, embed_dim, pad_id, dropout=dropout, aggr='max'):
        super(NodeEmbedding, self).__init__()
        self.pad_id = pad_id
        self.embed_dim = embed_dim
        self.aggr = aggr
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_id)
        if aggr not in {'rnn', 'max', 'mean'}:
            raise ValueError(f'Unknown aggr: {aggr}')
        if aggr == 'rnn':
            self.NodeEncoder = RNNEncoder(embed_dim=embed_dim, units=embed_dim // 2, dropout=dropout)
        else:
            self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        batch_size, node_num, token_num = x.shape
        node_lengths = token_num - torch.count_nonzero(x == self.pad_id, dim=-1)
        node_lengths[node_lengths == 0] = 1 # batch_size * node_num
        x = self.embedding(x) # batch_size * node_num * token_num * hidden_dim
        
        if self.aggr == 'rnn':
            node_lengths = node_lengths.view(-1)
            x = x.view(-1, token_num, self.embed_dim)
            x, _ = self.NodeEncoder(x, node_lengths)
            x = x[:,-1,:] # total_node_num * hidden_dim
            x = x.view(batch_size, node_num, self.embed_dim)
        elif self.aggr == 'max':
            x = self.dropout(x)
            x = x.max(-2).values
        else:
            x = self.dropout(x)
            x = x.sum(-2) / node_lengths
        return x

class HGNN(nn.Module):
    def __init__(self, code_field, nl_field, embed_dim=word_embed, num_edge_attr=3,
                 hidden_dim=lstm_units, hops=1, dropout=dropout, node_aggr='max'):
        super(HGNN, self).__init__()
        code_vocab_size = len(code_field.vocab)
        pad_id = code_field.vocab.stoi[pad]
        self.code_field = code_field
        self.nl_field = nl_field
        self.hops = hops
        self.hidden_dim = hidden_dim
        self.CodeEmbedding = NodeEmbedding(code_vocab_size, embed_dim, pad_id, dropout, node_aggr)
        self.ComEncoder = RNNEncoder(nl_field, embed_dim, hidden_dim // 2, dropout=dropout)
        self.KQAttn = KeyQueryAttention(hidden_dim)
        self.WQ = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.WK = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.WR = nn.Linear(num_edge_attr, hidden_dim, bias=False)
        
        self.SMP = StaticMsgPass()
        self.DMP = DynamicMsgPass(num_edge_attr, hidden_dim)
        self.GRU = nn.GRUCell(hidden_dim, hidden_dim)
        self.Fuse = Fusion(hidden_dim)
        self.src_lengths = None

    def ret_aug(self, Hc, Hc_, rev_node_num, rev_coms, rev_com_lens, sims):
        sims = sims.view(-1, 1, 1)
        H1_c, _ = self.KQAttn(Hc, Hc_, rev_node_num) # batch_size * node_num * hidden_dim
        comp = Hc + H1_c * sims # batch_size * node_num * hidden_dim
        out, _ = self.ComEncoder(rev_coms, rev_com_lens)
        h_rev = out * sims # batch_size * com_len * hidden_dim
        rev_final = h_rev[:,-1,:]
        return comp, h_rev, rev_final
    
    def get_A_dyn(self, graphs, comp):
        q = F.relu(self.WQ(comp)) # batch_size * node_num * hidden_dim
        k = F.relu(self.WK(comp))
        
        node_num = torch.tensor([graph.x.shape[0] for graph in graphs], device=comp.device)
        batch_size, max_nodes, hidden_dim = q.shape
        edge_attr = torch.cat([graph.edge_attr for graph in graphs])
        edge_attr = F.relu(self.WR(edge_attr)) # total_edge_num * hidden_dim
        edge_idx0 = torch.cat([torch.full([graph.edge_attr.shape[0]], i, device=q.device) \
                                   for graph, i in zip(graphs, range(batch_size))])
        edge_idx1 = torch.cat([graph.edge_index[0] for graph in graphs])
        edge_idx2 = torch.cat([graph.edge_index[1] for graph in graphs])
        
        k = k.unsqueeze(-2).expand(-1, -1, max_nodes, -1).clone()
        k[edge_idx0, edge_idx1, edge_idx2] += edge_attr  # batch_size * node_num * node_num * hidden_dim
        k = k.view(-1, max_nodes, hidden_dim) # total_node_num * node_num * hidden_dim
        q = q.view(-1, 1, hidden_dim) / math.sqrt(hidden_dim) # total_node_num * 1 * hidden_dim
        A_dyn = torch.bmm(q, k.transpose(1, 2)).view(batch_size, max_nodes, max_nodes)
        mask = sequence_mask(node_num, max_nodes).unsqueeze(1)
        A_dyn.masked_fill_(~mask, float('-inf'))
        A_dyn = F.softmax(A_dyn, dim=2)
        return A_dyn
        
    def forward(self, graphs, batch_nodes, rev_batch_nodes, rev_node_num, rev_coms, rev_com_lens, sims):
        node_num = torch.tensor([graph.x.shape[0] for graph in graphs], device=sims.device)
        # Graph Initialization
        Hc = self.CodeEmbedding(batch_nodes) # batch_size * node_num * hidden_dim
        Hc_ = self.CodeEmbedding(rev_batch_nodes)
        comp, h_rev, rev_final = self.ret_aug(Hc, Hc_, rev_node_num, rev_coms, rev_com_lens, sims)
        A_dyn = self.get_A_dyn(graphs, comp)
        
        # dense to sparse
        A_dyn = [a[graph.edge_index[0], graph.edge_index[1]] for graph, a in zip(graphs, A_dyn)]
        A_dyn = torch.cat(A_dyn).unsqueeze(1) # edge_num * 1
        comp = torch.cat([c[:n,:] for c, n in zip(comp, node_num)], dim=0) # total_node_num * hidden_dim
        graphs = Batch.from_data_list(graphs)
        
        # Hybrid Message Passing
        hv = hv_ = fv = comp
        for _ in range(self.hops):
            hv = self.SMP(hv, graphs.edge_index)
            hv_ = self.DMP(hv_, graphs.edge_index, graphs.edge_attr, A_dyn) # total_node_num * hidden_dim
            fuse = self.Fuse(hv, hv_)
            fv = self.GRU(fuse, fv) # total_node_num * hidden_dim
            
        graph_reps = global_max_pool(fv, batch=graphs.batch) # batch_size * hidden_dim
        init_h = self.Fuse(graph_reps, rev_final).unsqueeze(0)
        dec_init = (init_h, torch.zeros_like(init_h))
        
        # sparse to dense
        enc_outs = []
        for f, h in zip(torch.split(fv, node_num.tolist()), torch.split(h_rev, 1)):
            enc_out = torch.cat([f, h.squeeze(0)], dim=0)
            enc_outs.append(enc_out)
        self.src_lengths = torch.tensor(list(map(len, enc_outs)), device=fv.device)
        enc_outs = pad_sequence(enc_outs, batch_first=True) # batch_size * (node_num + com_len) * hidden_dim
        return enc_outs, dec_init

class DecoderCellState(
        namedtuple('DecoderCellState',
                   ('context', 'state', 'alignments'), defaults=[None])):
    def batch_select(self, indices):
        select = lambda x, dim=0: x.index_select(dim, indices)
        return self._replace(context = select(self.context),
                             state = tuple_map(select, self.state, dim=1),
                             alignments = tuple_map(select, self.alignments))

class DecoderCell(nn.Module):
    def __init__(self, embed_dim, units=lstm_units, num_layers=1, dropout=dropout,
                 glob_attn='mul', memory_dim=None, input_feed=True, use_attn_layer=True):
        super(DecoderCell, self).__init__()
        self.glob_attn = glob_attn
        self.input_feed = glob_attn and input_feed
        self.use_attn_layer = glob_attn and use_attn_layer
        self.dropout = nn.Dropout(dropout)
        
        if memory_dim is None:
            memory_dim = units
        cell_in_dim, context_dim = embed_dim, memory_dim
        if glob_attn is not None:
            self.attention = GlobalAttention(units, memory_dim, glob_attn)
            
            if use_attn_layer:
                self.attn_layer = nn.Linear(context_dim + units, units)
                context_dim = units
            if input_feed:
                cell_in_dim += context_dim
        self.cell = nn.LSTM(cell_in_dim, units, num_layers, batch_first=True,
                            dropout=dropout, bidirectional=False)
            
    def forward(self, tgt_embed, prev_state, memory=None, src_lengths=None):
#        perform one decoding step
        cell_input = self.dropout(tgt_embed) # batch_size * embed_size
        
        if self.glob_attn is not None:
            if self.input_feed:
                cell_input = torch.cat([cell_input, prev_state.context], 1).unsqueeze(1)
                # batch_size * 1 * (embed_size + units)
            output, state = self.cell(cell_input, prev_state.state)
            output = output.squeeze(1) # batch_size * units
            context, alignments = self.attention(output, memory, src_lengths)
            if self.use_attn_layer:
                context = torch.cat([context, output], 1)
                context = torch.tanh(self.attn_layer(context))
            context = self.dropout(context)
            return DecoderCellState(context, state, alignments)
        else:
            output, state = self.cell(cell_input, prev_state.state)
            output = output.squeeze(1)
            return DecoderCellState(output, state)

class BasicDecoder(nn.Module):
    def __init__(self, field, embed_dim=word_embed, units=lstm_units, num_layers=1,
                 dropout=dropout, glob_attn='mul', **kwargs):
        ' If hybrid, memory, memory_dim and src_lengths should be tuples. '
        super(BasicDecoder, self).__init__()
        vocab_size = len(field.vocab)
        pad_id = field.vocab.stoi[pad]
        self.field = field
        self.units = units
        self.glob_attn = glob_attn
        
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_id)
        self.out_layer = nn.Linear(units, vocab_size, bias=False)
        self.cell = DecoderCell(embed_dim, units, num_layers, dropout,
                                glob_attn, **kwargs)
        
    @property
    def attn_history(self):
        if self.cell.hybrid_mode:
            return self.cell.attention.attn_history
        elif self.glob_attn:
            return 'std'
    
    def initialize(self, enc_final):
        init_context = torch.zeros(enc_final[0].shape[1], self.units,
                                   device=enc_final[0].device)
        return DecoderCellState(init_context, enc_final)
    
    def forward(self, tgt_inputs, enc_final, memory=None, src_lengths=None, return_contxt=False):
        tgt_embeds = self.embedding(tgt_inputs) # batch_size * max_steps * units
        prev_state = self.initialize(enc_final)
        output_seqs, attn_history = [], []
        
        for tgt_embed in tgt_embeds.split(1, 1):
            state = self.cell(tgt_embed.squeeze(1), prev_state, memory, src_lengths)
            output_seqs.append(state.context)
            if self.glob_attn is not None:
                attn_history.append(state.alignments)
            prev_state = state
        
        output_seqs = torch.stack(output_seqs, 1)
        logits = self.out_layer(output_seqs)
        if self.glob_attn is not None:
            attn_history = torch.stack(attn_history, 1)
        if return_contxt:
            return logits, output_seqs, attn_history
        else:
            return logits, attn_history