import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from masking import utils as masking_utils
from utils import *
from TreeCRF import ConstituencyTreeCRF
from torch.distributions import Bernoulli


class RNNLM(nn.Module):
    def __init__(self, vocab=10000,
                 w_dim=650,
                 h_dim=650,
                 num_layers=2,
                 dropout=0.5):
        super(RNNLM, self).__init__()
        self.h_dim = h_dim
        self.num_layers = num_layers
        self.word_vecs = nn.Embedding(vocab, w_dim)
        self.dropout = nn.Dropout(dropout)
        self.rnn = nn.LSTM(w_dim, h_dim, num_layers=num_layers,
                           dropout=dropout, batch_first=True)
        self.vocab_linear = nn.Linear(h_dim, vocab)
        self.vocab_linear.weight = self.word_vecs.weight  # weight sharing

    def forward(self, sent):
        word_vecs = self.dropout(self.word_vecs(sent[:, :-1]))
        h, _ = self.rnn(word_vecs)
        log_prob = F.log_softmax(self.vocab_linear(self.dropout(h)), 2)  # b x l x v
        ll = torch.gather(log_prob, 2, sent[:, 1:].unsqueeze(2)).squeeze(2)
        return ll.sum(1)

    def generate(self, bos=2, eos=3, max_len=150):
        x = []
        bos = torch.LongTensor(1, 1).cuda().fill_(bos)
        emb = self.dropout(self.word_vecs(bos))
        prev_h = None
        for l in range(max_len):
            h, prev_h = self.rnn(emb, prev_h)
            prob = F.softmax(self.vocab_linear(self.dropout(h.squeeze(1))), 1)
            sample = torch.multinomial(prob, 1)
            emb = self.dropout(self.word_vecs(sample))
            x.append(sample.item())
            if x[-1] == eos:
                x.pop()
                break
        return x


class SeqLSTM(nn.Module):
    def __init__(self, i_dim=200,
                 h_dim=0,
                 num_layers=1,
                 dropout=0):
        super(SeqLSTM, self).__init__()
        self.i_dim = i_dim
        self.h_dim = h_dim
        self.num_layers = num_layers
        self.linears = nn.ModuleList([nn.Linear(h_dim + i_dim, h_dim * 4) if l == 0 else
                                      nn.Linear(h_dim * 2, h_dim * 4) for l in range(num_layers)])
        self.dropout = dropout
        self.dropout_layer = nn.Dropout(dropout)

    def forward(self, x, prev_h=None):
        if prev_h is None:
            prev_h = [(x.new(x.size(0), self.h_dim).fill_(0),
                       x.new(x.size(0), self.h_dim).fill_(0)) for _ in range(self.num_layers)]
        curr_h = []
        for l in range(self.num_layers):
            input = x if l == 0 else curr_h[l - 1][0]
            if l > 0 and self.dropout > 0:
                input = self.dropout_layer(input)
            concat = torch.cat([input, prev_h[l][0]], 1)
            all_sum = self.linears[l](concat)
            i, f, o, g = all_sum.split(self.h_dim, 1)
            c = F.sigmoid(f) * prev_h[l][1] + F.sigmoid(i) * F.tanh(g)
            h = F.sigmoid(o) * F.tanh(c)
            curr_h.append((h, c))
        return curr_h


class TreeLSTM(nn.Module):
    def __init__(self, dim=200):
        super(TreeLSTM, self).__init__()
        self.dim = dim
        self.linear = nn.Linear(dim * 2, dim * 5)

    def forward(self, x1, x2, e=None):
        if not isinstance(x1, tuple):
            x1 = (x1, None)
        h1, c1 = x1
        if x2 is None:
            x2 = (torch.zeros_like(h1), torch.zeros_like(h1))
        elif not isinstance(x2, tuple):
            x2 = (x2, None)
        h2, c2 = x2
        if c1 is None:
            c1 = torch.zeros_like(h1)
        if c2 is None:
            c2 = torch.zeros_like(h2)
        concat = torch.cat([h1, h2], 1)
        all_sum = self.linear(concat)
        i, f1, f2, o, g = all_sum.split(self.dim, 1)

        c = F.sigmoid(f1) * c1 + F.sigmoid(f2) * c2 + F.sigmoid(i) * F.tanh(g)
        h = F.sigmoid(o) * F.tanh(c)
        return (h, c)


class PositionalEmbedding(nn.Module):
    def __init__(self, demb):
        super(PositionalEmbedding, self).__init__()

        self.demb = demb

        inv_freq = 1 / (10000 ** (torch.arange(0.0, demb, 2.0) / demb))
        self.register_buffer('inv_freq', inv_freq)

    def forward(self, pos_seq, bsz=None):
        sinusoid_inp = torch.ger(pos_seq, self.inv_freq)
        pos_emb = torch.cat([sinusoid_inp.sin(), sinusoid_inp.cos()], dim=-1)

        if bsz is not None:
            return pos_emb[:, None, :].expand(-1, bsz, -1)
        else:
            return pos_emb[:, None, :]  # r * None * d_model


class UTG(nn.Module):
    def __init__(self, vocab=100,
                 w_dim=20,
                 #  h_dim = 20,
                 n_head=10,
                 d_head=10,
                 d_inner=10,
                 num_layers=1,
                 dropout=0.5,
                 dropout_core=0.1,
                 dropatt=0.1,
                 q_dim=20,
                 idx2word={},
                 word2idx={},
                 max_len=250,
                 tgt_len=None,
                 mem_len=None,
                 pre_lnorm=False,
                 same_emb=False):
        super(UTG, self).__init__()
        self.S = 0  # action idx for shift/generate
        self.R = 1  # action idx for reduce
        self.n_token = vocab
        self.d_emb = w_dim
        self.d_model = w_dim
        self.n_head = n_head
        self.d_head = d_head
        self.d_inner = d_inner
        self.tgt_len = tgt_len
        self.dropatt = dropatt
        self.emb = nn.Embedding(vocab, w_dim)
        if same_emb:
            self.emb_att = self.emb
        else:
            self.emb_att = nn.Embedding(vocab, w_dim)
        self.projection = nn.Linear(w_dim, vocab)
        self.projection.weight = self.emb_att.weight
        self.dropout = nn.Dropout(dropout)
        self.dropout_core = nn.Dropout(dropout_core)
        # self.stack_rnn = SeqLSTM(w_dim, h_dim, num_layers = num_layers, dropout = dropout)
        self.tree_rnn = TreeLSTM(w_dim)
        # self.vocab_mlp = nn.Sequential(nn.Dropout(dropout), nn.Linear(h_dim, vocab))
        self.n_layers = num_layers
        self.q_binary = nn.Sequential(nn.Linear(q_dim * 2, q_dim * 2), nn.ReLU(), nn.LayerNorm(q_dim * 2),
                                      nn.Dropout(dropout), nn.Linear(q_dim * 2, 1))
        # self.action_mlp_p = nn.Sequential(nn.Dropout(dropout), nn.Linear(h_dim, 1))
        self.w_dim = w_dim
        # self.h_dim = h_dim
        self.q_dim = q_dim
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            self.layers.append(RelPartialLearnableDecoderLayer(
                self.n_head, self.d_model, self.d_head, self.d_inner, dropout_core,
                tgt_len=tgt_len, ext_len=None, mem_len=None,
                dropatt=dropatt, pre_lnorm=pre_lnorm))

        self.pos_emb = PositionalEmbedding(self.d_model)
        self.r_w_bias = nn.Parameter(torch.Tensor(self.n_head, self.d_head))
        self.r_r_bias = nn.Parameter(torch.Tensor(self.n_head, self.d_head))

        self.q_leaf_rnn = nn.LSTM(w_dim, q_dim, bidirectional=True, batch_first=True)
        self.q_crf = ConstituencyTreeCRF()
        self.pad1 = 1  # idx for <s> token from ptb.dict
        self.pad2 = 2  # idx for </s> token from ptb.dict
        self.q_pos_emb = nn.Embedding(max_len, w_dim)  # position embeddings
        # self.vocab_mlp[-1].weight = self.emb.weight #share embeddings
        self.idx2word = idx2word
        self.word2idx = word2idx
        self.left_arc = self.word2idx['(S']
        self.right_arc = self.word2idx['S)']

    def get_span_scores(self, x):
        # produces the span scores s_ij
        bos = x.new(x.size(0), 1).fill_(self.pad1)
        eos = x.new(x.size(0), 1).fill_(self.pad2)
        x = torch.cat([bos, x, eos], 1)
        x_vec = self.dropout(self.emb(x))
        pos = torch.arange(0, x.size(1)).unsqueeze(0).expand_as(x).long().cuda()
        x_vec = x_vec + self.dropout(self.q_pos_emb(pos))
        q_h, _ = self.q_leaf_rnn(x_vec)
        fwd = q_h[:, 1:, :self.q_dim]
        bwd = q_h[:, :-1, self.q_dim:]
        fwd_diff = fwd[:, 1:].unsqueeze(1) - fwd[:, :-1].unsqueeze(2)
        bwd_diff = bwd[:, :-1].unsqueeze(2) - bwd[:, 1:].unsqueeze(1)
        concat = torch.cat([fwd_diff, bwd_diff], 3)
        scores = self.q_binary(concat).squeeze(3)
        return scores

    def get_action_masks(self, actions, length):
        # this masks out actions so that we don't incur a loss if some actions are deterministic
        # in practice this doesn't really seem to matter
        mask = actions.new(actions.size(0), actions.size(1)).fill_(1)
        for b in range(actions.size(0)):
            num_shift = 0
            stack_len = 0
            for l in range(actions.size(1)):
                if stack_len < 2:
                    mask[b][l].fill_(0)
                if actions[b][l].item() == self.S:
                    num_shift += 1
                    stack_len += 1
                else:
                    stack_len -= 1
        return mask

    def get_tree_str(self, idx, tree_bracket):
        tree_str = ""
        i = 0
        while i < len(tree_bracket):
            c = tree_bracket[i]
            if c == '(':
                tree_str += "(S "
            if c == ')':
                tree_str += "S) "
            if c >= '0' and c <= '9':
                id = 0
                while i < len(tree_bracket) and tree_bracket[i] >= '0' and tree_bracket[i] <= '9':
                    id = id * 10 + int(tree_bracket[i])
                    i += 1
                tree_str += self.idx2word[int(idx[id])] + ' '
                continue
            i += 1
        if tree_str[-1] == ' ':
            tree_str = tree_str[:-1]
        return tree_str

    def get_tree_id(self, idx, tree_bracket):
        tree_id = []
        i = 0
        while i < len(tree_bracket):
            c = tree_bracket[i]
            if c == '(':
                tree_id.append(self.word2idx['(S'])
            if c == ')':
                tree_id.append(self.word2idx['S)'])
            if c >= '0' and c <= '9':
                id = 0
                while i < len(tree_bracket) and tree_bracket[i] >= '0' and tree_bracket[i] <= '9':
                    id = id * 10 + int(tree_bracket[i])
                    i += 1
                tree_id.append(int(idx[id]))
                continue
            i += 1
        return tree_id

    def generate_left_tree(self, idx):
        num = len(idx)
        tree = ''
        for i in range(num - 1):
            tree += '( '
        tree += '0 '
        for i in range(1, num):
            tree += str(i) + ' ) '
        return tree

    def generate_right_tree(self, idx):
        num = len(idx)
        tree = ''
        for i in range(num - 1):
            tree += '( ' + str(i) + ' '
        tree += str(num - 1) + ' '
        for i in range(num - 1):
            tree += ') '
        return tree

    def get_ranges(self, start_token, pad_token, left_arc, right_arc):
        return masking_utils.TokenTypeRanges(start_token,
                                             pad_token,
                                             left_arc,
                                             right_arc
                                             )

    def forward(self, x, samples=1, is_temp=1., has_eos=True, mode='default', kl_pen=1.):
        # For has eos, if </s> exists, then inference network ignores it.
        # Note that </s> is predicted for training since we want the model to know when to stop.
        # However it is ignored for PPL evaluation on the version of the PTB dataset from
        # the original RNNG paper (Dyer et al. 2016)
        # print(x)

        # init_emb = self.dropout(self.emb(x[:, 0]))
        x = x[:, 1:]
        batch, length = x.size(0), x.size(1)
        # print(x)
        # print(batch)
        # print(length)
        ranges = self.get_ranges(1, 0, self.left_arc, self.right_arc)
        maskrules = masking_utils.get_masking_rules(
            "stack_compose_double_closing_nt",
            sequence_length=768,
            memory_length=768,
            transparency_prob=0.0,
            gather_into_new_memory=False,
            transparency_depth_threshold=-1
        )
        # print(1)
        # print(batch, length)
        if has_eos:
            parse_length = length - 1
            parse_x = x[:, :-1]
        else:
            parse_length = length
            parse_x = x
        if mode == 'left':
            # print(parse_x[0])
            tree_brackets = []
            for i in range(batch):
                tree = self.generate_left_tree(parse_x[i])
                tree_brackets.append(tree)
        elif mode == 'right':
            tree_brackets = []
            for i in range(batch):
                tree = self.generate_right_tree(parse_x[i])
                tree_brackets.append(tree)
        else:
            # word_vecs =  self.dropout(self.emb(x))
            scores = self.get_span_scores(parse_x)
            self.scores = scores
            scores = scores / is_temp
            self.q_crf._forward(scores)
            self.q_crf._entropy(scores)
            entropy = self.q_crf.entropy[0][parse_length - 1]
            crf_input = scores.unsqueeze(1).expand(batch, samples, parse_length, parse_length)
            crf_input = crf_input.contiguous().view(batch * samples, parse_length, parse_length)
            for i in range(len(self.q_crf.alpha)):
                for j in range(len(self.q_crf.alpha)):
                    self.q_crf.alpha[i][j] = self.q_crf.alpha[i][j].unsqueeze(1).expand(
                        batch, samples).contiguous().view(batch * samples)
            # print(2)
            ##sample tree
            _, log_probs_action_q, tree_brackets, spans = self.q_crf._sample(crf_input, self.q_crf.alpha)
        # print(tree_brackets)
        # print(3)
        attn_masks = []
        attn_relpos = []
        inputs = []
        labels = []
        actions = []
        max_len_tmp = 0
        for b in range(batch * samples):
            # add NT
            action = get_actions(tree_brackets[b])
            if has_eos:
                actions.append(
                    action + [self.S, self.R])  # we train the model to generate <s> and then do a final reduce
            else:
                actions.append(action)

            if has_eos:
                sent_str = self.get_tree_str(parse_x[b // samples], tree_brackets[b])
                sent_id = np.array(self.get_tree_id(parse_x[b // samples], tree_brackets[b]), dtype=np.int32)
            else:
                sent_str = self.get_tree_str(parse_x[b // samples], tree_brackets[b])
                sent_id = np.array(self.get_tree_id(parse_x[b // samples], tree_brackets[b]), dtype=np.int32)
            # print(sent_str)
            # print(len(sent_id))

            input = np.array([1] + list(sent_id))
            if len(input) > max_len_tmp:
                max_len_tmp = len(input)

            if has_eos:
                label = np.array(list(sent_id) + [2])
            else:
                label = np.array(list(sent_id) + [0])

            tmp = {"inputs": input, "labels": label}
            tmp = masking_utils.compute_token_types(tmp, ranges)
            # generate chunks
            chunks = maskrules.chunks_for_sequence(
                tmp["inputs"],
                tmp["inputs_ttypes"],
                tmp["labels"],
                tmp["labels_ttypes"],
            )
            # print(chunks[0][0])
            # print(chunks[0][1])
            # only consider in one chunk
            len_inp = len(input)

            len_inp_processed = (len_inp * 4 - 2) // 3
            input_processed = np.array(chunks[0][0][0:len_inp_processed])
            inputs.append(input_processed)
            label_processed = np.array(chunks[0][2][0:len_inp_processed])
            labels.append(label_processed)
            attn_mask = np.array(chunks[0][4])
            l_chunk = len(attn_mask[0])
            attn_mask = attn_mask[0:len_inp_processed, 0:len_inp_processed]
            # !!! not L?
            # Prevent attend to paddings
            # pad_mask = np.array([1 if id > 0 else 0 for id in input])
            # # padding_len = len(attn_mask) - len(pad_mask)
            # # pad_mask = np.pad(pad_mask, (0,padding_len))
            # attn_mask = pad_mask[None,:] * attn_mask
            attn_masks.append(attn_mask)
            # print(attn_mask)
            attn_relpos.append(np.array(chunks[0][5])[0: len(attn_mask), \
                               l_chunk: l_chunk + len(attn_mask)])
            # exit()
        # print(max_len_tmp)
        attn_masks = np.array(attn_masks)
        attn_relpos = np.array(attn_relpos)

        actions = torch.Tensor(actions).float().cuda()
        attn_masks = torch.LongTensor(attn_masks).cuda()  # B * l_inp * l_inp
        attn_relpos = torch.LongTensor(attn_relpos).cuda()  # B * l_inp * l_inp

        inputs = np.array(inputs)
        labels = np.array(labels)

        inp = torch.LongTensor(inputs.T).cuda()  # l_inp * B
        tgt = torch.LongTensor(labels.T).cuda()  # l_tgt(=l_inp) * B
        tgt_len = tgt.size(0)
        inp_len = inp.size(0)
        batch_expand = batch * samples

        # print(4)

        hidden = self._forward_TG(batch_size=batch_expand, inp_len=inp_len, inp=inp, attn_masks=attn_masks,
                                  attn_relpos=attn_relpos)
        out_logits = self.projection(hidden)
        weights = torch.ones(self.n_token)
        weights[4] = kl_pen
        weights[5] = kl_pen
        crit = nn.CrossEntropyLoss(reduction='none', ignore_index=0, weight=weights.cuda())
        out_p = out_logits.view(tgt_len, batch_expand, -1)
        out_p = out_p.permute(0, 2, 1)
        log_p = -crit(out_p, tgt)
        log_p = log_p.transpose(0, 1).contiguous()
        log_p = log_p.sum(1)
        log_p = log_p.contiguous().view(batch, samples)
        actions = actions.contiguous().view(batch, samples, -1)
        # out_log_p = F.log_softmax(out_logits, dim=-1)
        # out_log_p = out_log_p.view(tgt_len, batch_expand, -1)
        # log_p = torch.gather(out_log_p, -1, tgt.unsqueeze(-1)).squeeze(-1) # l_tgt * B # !!
        # log_p = log_p.transpose(0, 1).contiguous() # B * l_tgt
        # log_p = log_p.sum(1) # B
        # log_p = log_p.contiguous().view(batch, samples) # b * samples
        if mode not in ['left', 'right']:
            log_probs_action_q = log_probs_action_q.contiguous().view(batch, samples)
            return log_p, log_probs_action_q, actions, entropy
        else:
            return log_p, None, actions, None
        # out_logit processing

    def _forward_TG(self, batch_size, inp_len, inp, attn_masks, attn_relpos):
        # inp: inp_len * B    attn_masks: B * inp_len * inp_len    attn_relpos: B * inp_len * inp_len
        word_emb = self.emb_att(inp)  # inp_len * B * d_emb
        attn_mask = attn_masks.bool().permute(1, 2, 0)  # bool, inp_len * inp_len * B
        # attn_mask = torch.tril(torch.ones(inp_len, inp_len)).bool().cuda()
        hiddens = []

        min_relpos = -inp_len
        max_relpos = inp_len
        pos_seq = np.arange(0, max_relpos, 1.0)
        pos_seq = torch.Tensor(pos_seq).cuda()
        pos_emb = self.pos_emb(pos_seq)  ## L * None * d_model

        core_out = self.dropout_core(word_emb)  # inp_len * B * d_model
        pos_emb = self.dropout_core(pos_emb)

        hiddens.append(core_out)

        for i, layer in enumerate(self.layers):
            core_out = layer(core_out, pos_emb, self.r_w_bias, self.r_r_bias,
                             dec_attn_mask=attn_mask, attn_relpos=attn_relpos, mems=None)
            hiddens.append(core_out)

        core_out = self.dropout_core(core_out)

        return core_out

    def logsumexp(self, x, dim=1):
        d = torch.max(x, dim)[0]
        if x.dim() == 1:
            return torch.log(torch.exp(x - d).sum(dim)) + d
        else:
            return torch.log(torch.exp(x - d.unsqueeze(dim).expand_as(x)).sum(dim)) + d


class PositionwiseFF(nn.Module):
    def __init__(self, d_model, d_inner, dropout, pre_lnorm=False):
        super(PositionwiseFF, self).__init__()

        self.d_model = d_model
        self.d_inner = d_inner
        self.dropout = dropout

        self.CoreNet = nn.Sequential(
            nn.Linear(d_model, d_inner), nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(d_inner, d_model),
            nn.Dropout(dropout),
        )

        self.layer_norm = nn.LayerNorm(d_model)
        # self.layer_norm = nn.Identity()
        self.pre_lnorm = pre_lnorm

    def forward(self, inp):
        if self.pre_lnorm:
            ##### layer normalization + positionwise feed-forward
            core_out = self.CoreNet(self.layer_norm(inp))

            ##### residual connection
            output = core_out + inp
        else:
            ##### positionwise feed-forward
            core_out = self.CoreNet(inp)

            ##### residual connection + layer normalization
            output = self.layer_norm(inp + core_out)

        return output


class RelMultiHeadAttn(nn.Module):
    def __init__(self, n_head, d_model, d_head, dropout, dropatt=0,
                 tgt_len=None, ext_len=None, mem_len=None, pre_lnorm=False):
        super(RelMultiHeadAttn, self).__init__()

        self.n_head = n_head
        self.d_model = d_model
        self.d_head = d_head
        self.dropout = dropout

        self.qkv_net = nn.Linear(d_model, 3 * n_head * d_head, bias=False)

        self.drop = nn.Dropout(dropout)
        self.dropatt = nn.Dropout(dropatt)
        self.o_net = nn.Linear(n_head * d_head, d_model, bias=False)

        self.layer_norm = nn.LayerNorm(d_model)
        # self.layer_norm = nn.Identity()
        self.scale = 1 / (d_head ** 0.5)

        self.pre_lnorm = pre_lnorm

    def _parallelogram_mask(self, h, w, left=False):
        mask = torch.ones((h, w)).byte()
        m = min(h, w)
        mask[:m, :m] = torch.triu(mask[:m, :m])
        mask[-m:, -m:] = torch.tril(mask[-m:, -m:])

        if left:
            return mask
        else:
            return mask.flip(0)

    def _shift(self, x, qlen, klen, mask, left=False):
        if qlen > 1:
            zero_pad = torch.zeros((x.size(0), qlen - 1, x.size(2), x.size(3)),
                                   device=x.device, dtype=x.dtype)
        else:
            zero_pad = torch.zeros(0, device=x.device, dtype=x.dtype)

        if left:
            mask = mask.flip(1)
            x_padded = torch.cat([zero_pad, x], dim=1).expand(qlen, -1, -1, -1)
        else:
            x_padded = torch.cat([x, zero_pad], dim=1).expand(qlen, -1, -1, -1)

        x = x_padded.masked_select(mask[:, :, None, None]) \
            .view(qlen, klen, x.size(2), x.size(3))

        return x

    def _rel_shift(self, x, zero_triu=False):
        zero_pad = torch.zeros((x.size(0), 1, *x.size()[2:]),
                               device=x.device, dtype=x.dtype)
        x_padded = torch.cat([zero_pad, x], dim=1)

        x_padded = x_padded.view(x.size(1) + 1, x.size(0), *x.size()[2:])

        x = x_padded[1:].view_as(x)

        if zero_triu:
            ones = torch.ones((x.size(0), x.size(1)))
            x = x * torch.tril(ones, x.size(1) - x.size(0))[:, :, None, None]

        return x

    def forward(self, w, r, attn_mask=None, mems=None):
        raise NotImplementedError


class RelPartialLearnableMultiHeadAttn(RelMultiHeadAttn):
    def __init__(self, *args, **kwargs):
        super(RelPartialLearnableMultiHeadAttn, self).__init__(*args, **kwargs)

        self.r_net = nn.Linear(self.d_model, self.n_head * self.d_head, bias=False)

    def forward(self, w, r, r_w_bias, r_r_bias, attn_mask=None, attn_relpos=None, mems=None):
        qlen, rlen, bsz = w.size(0), r.size(0), w.size(1)  # L, M-m, B
        # r: M-m * None * d_model
        if mems is not None:
            cat = torch.cat([mems, w], 0)
            if self.pre_lnorm:
                w_heads = self.qkv_net(self.layer_norm(cat))
            else:
                w_heads = self.qkv_net(cat)
            r_head_k = self.r_net(r)  # M-m * None * (n_head * d_head)

            w_head_q, w_head_k, w_head_v = torch.chunk(w_heads, 3, dim=-1)
            w_head_q = w_head_q[-qlen:]
        else:
            if self.pre_lnorm:
                w_heads = self.qkv_net(self.layer_norm(w))
            else:
                w_heads = self.qkv_net(w)
            r_head_k = self.r_net(r)

            w_head_q, w_head_k, w_head_v = torch.chunk(w_heads, 3, dim=-1)

        klen = w_head_k.size(0)

        w_head_q = w_head_q.view(qlen, bsz, self.n_head, self.d_head)  # qlen x bsz x n_head x d_head
        w_head_k = w_head_k.view(klen, bsz, self.n_head, self.d_head)  # qlen x bsz x n_head x d_head
        w_head_v = w_head_v.view(klen, bsz, self.n_head, self.d_head)  # klen x bsz x n_head x d_head

        r_head_k = r_head_k.view(rlen, self.n_head, self.d_head)  # rlen x n_head x d_head

        #### compute attention score
        rw_head_q = w_head_q + r_w_bias  # L * B * n_head * d_head               # qlen x bsz x n_head x d_head
        AC = torch.einsum('ibnd,jbnd->ijbn', (rw_head_q, w_head_k))  # qlen x klen x bsz x n_head

        rr_head_q = w_head_q + r_r_bias
        BD = torch.einsum('ibnd,jnd->ijbn', (rr_head_q, r_head_k))  # qlen x rlen x bsz x n_head
        # # BD = self._rel_shift(BD)
        # attn_relpos = torch.clip(attn_relpos, -qlen, qlen)
        # attn_relpos = (qlen - attn_relpos).long()
        # # print(attn_relpos.size(0), rlen)
        # relpos_one_hot = torch.Tensor(F.one_hot(attn_relpos, num_classes=rlen)).float()               # bsz x qlen x klen x rlen
        # BD = torch.einsum('ijbn,bisj->isbn', BD, relpos_one_hot)                # qlen x klen x bsz x n_head
        attn_score = AC + BD
        attn_score.mul_(self.scale)

        #### compute attention probability
        if attn_mask is not None and attn_mask.any().item():
            if attn_mask.dim() == 2:
                attn_score = attn_score.float().masked_fill(
                    ~attn_mask[:, :, None, None], -float('inf')).type_as(attn_score)
            elif attn_mask.dim() == 3:
                attn_score = attn_score.float().masked_fill(
                    ~attn_mask[:, :, :, None], -float('inf')).type_as(attn_score)

        # [qlen x klen x bsz x n_head]
        attn_prob = F.softmax(attn_score, dim=1)
        attn_prob = self.dropatt(attn_prob)

        #### compute attention vector
        attn_vec = torch.einsum('ijbn,jbnd->ibnd', (attn_prob, w_head_v))

        # [qlen x bsz x n_head x d_head]
        attn_vec = attn_vec.contiguous().view(
            attn_vec.size(0), attn_vec.size(1), self.n_head * self.d_head)

        ##### linear projection
        attn_out = self.o_net(attn_vec)
        attn_out = self.drop(attn_out)

        if self.pre_lnorm:
            ##### residual connection
            output = w + attn_out
        else:
            ##### residual connection + layer normalization
            output = self.layer_norm(w + attn_out)

        return output


class RelPartialLearnableDecoderLayer(nn.Module):
    def __init__(self, n_head, d_model, d_head, d_inner, dropout,
                 **kwargs):
        super(RelPartialLearnableDecoderLayer, self).__init__()

        self.dec_attn = RelPartialLearnableMultiHeadAttn(n_head, d_model,
                                                         d_head, dropout, **kwargs)
        self.pos_ff = PositionwiseFF(d_model, d_inner, dropout,
                                     pre_lnorm=kwargs.get('pre_lnorm'))

    def forward(self, dec_inp, r, r_w_bias, r_r_bias, dec_attn_mask=None, attn_relpos=None, mems=None):
        output = self.dec_attn(dec_inp, r, r_w_bias, r_r_bias,
                               attn_mask=dec_attn_mask, attn_relpos=attn_relpos,
                               mems=mems)
        output = self.pos_ff(output)

        return output
