import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from masking import utils as masking_utils
from masking import masking_types as types
import time


# from helping_utils.logger import configure_logger, get_logger
# logger = get_logger()
class PositionalEmbedding(nn.Module):

    def __init__(self, demb):
        super(PositionalEmbedding, self).__init__()

        self.demb = demb

        inv_freq = 1 / (10000**(torch.arange(0.0, demb, 2.0) / demb))
        self.register_buffer('inv_freq', inv_freq)

    def forward(self, pos_seq, bsz=None):
        sinusoid_inp = torch.ger(pos_seq, self.inv_freq)
        pos_emb = torch.cat([sinusoid_inp.sin(), sinusoid_inp.cos()], dim=-1)

        if bsz is not None:
            return pos_emb[:, None, :].expand(-1, bsz, -1)
        else:
            return pos_emb[:, None, :]  # r * None * d_model


class PositionwiseFF(nn.Module):

    def __init__(self, d_model, d_inner, dropout, pre_lnorm=False):
        super(PositionwiseFF, self).__init__()

        self.d_model = d_model
        self.d_inner = d_inner
        self.dropout = dropout

        self.CoreNet = nn.Sequential(
            nn.Linear(d_model, d_inner),
            nn.ReLU(inplace=True),
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

    def __init__(self,
                 n_head,
                 d_model,
                 d_head,
                 dropout,
                 tgt_len=None,
                 ext_len=None,
                 mem_len=None,
                 pre_lnorm=False):
        super(RelMultiHeadAttn, self).__init__()

        self.n_head = n_head
        self.d_model = d_model
        self.d_head = d_head
        self.dropout = dropout

        self.qkv_net = nn.Sequential(
            nn.Linear(d_model, 3 * n_head * d_head, bias=False),
            nn.Dropout(dropout))

        self.drop = nn.Dropout(dropout)
        self.dropatt = nn.Dropout(dropout)
        self.o_net = nn.Linear(n_head * d_head, d_model,
                               bias=False)  # output net

        self.layer_norm = nn.LayerNorm(d_model)
        # self.layer_norm = nn.Identity()
        self.scale = 1 / (d_head**0.5)

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
                                   device=x.device,
                                   dtype=x.dtype)
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
                               device=x.device,
                               dtype=x.dtype)
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

        self.r_net = nn.Linear(self.d_model,
                               self.n_head * self.d_head,
                               bias=False)

    def forward(self,
                w,
                r,
                r_w_bias,
                r_r_bias,
                attn_mask=None,
                attn_relpos=None,
                min_len=None,
                max_len=None,
                mems=None,
                terminal=False):
        qlen, rlen, bsz = w.size(0), r.size(0), w.size(1)  # L, M-m, B
        # print(qlen, rlen)
        # r: M-m * None * d_model
        if mems is not None:
            cat = torch.cat([mems, w], 0)
            if self.pre_lnorm:
                w_heads = self.qkv_net(self.layer_norm(cat))
            else:
                w_heads = self.qkv_net(cat)
            r_head_k = self.r_net(
                r
            )  # M-m * None * (n_head * d_head) // M-m * B * (n_head * d_head)

            w_head_q, w_head_k, w_head_v = torch.chunk(w_heads, 3, dim=-1)
            w_head_q = w_head_q[-qlen:]
        else:
            if self.pre_lnorm:
                w_heads = self.qkv_net(self.layer_norm(w))
            else:
                # print(w.shape)
                w_heads = self.qkv_net(w)
            r_head_k = self.r_net(r)

            w_head_q, w_head_k, w_head_v = torch.chunk(w_heads, 3, dim=-1)
        # #test
        # r_heads = self.qkv_net(r)
        # r_head_q, r_head_k, r_head_v = torch.chunk(r_heads, 3, dim=-1)
        # r_head_q = r_head_q.view(rlen, self.n_head, self.d_head)
        # r_head_k = r_head_k.view(rlen, self.n_head, self.d_head)
        # #---
        klen = w_head_k.size(0)

        w_head_q = w_head_q.view(qlen, bsz, self.n_head,
                                 self.d_head)  # qlen x bsz x n_head x d_head
        w_head_k = w_head_k.view(klen, bsz, self.n_head,
                                 self.d_head)  # qlen x bsz x n_head x d_head
        w_head_v = w_head_v.view(klen, bsz, self.n_head,
                                 self.d_head)  # klen x bsz x n_head x d_head

        # if composed and rlen == qlen:
        #     r_head_k = r_head_k.view(rlen, bsz, self.n_head, self.d_head)       # rlen x bsz x n_head x d_head
        # else:
        r_head_k = r_head_k.view(rlen, self.n_head,
                                 self.d_head)  # rlen x n_head x d_head
        # #test
        # r_w_bias = r_head_q[-1]
        # r_r_bias = r_head_q[-1]
        # #---
        #### compute attention score
        rw_head_q = w_head_q + r_w_bias  # L * B * n_head * d_head               # qlen x bsz x n_head x d_head
        AC = torch.einsum('ibnd,jbnd->ijbn',
                          (rw_head_q, w_head_k))  # qlen x klen x bsz x n_head

        rr_head_q = w_head_q + r_r_bias
        # if composed and rlen == qlen:
        #     BD = torch.einsum('ibnd,jbnd->ijbn', (rr_head_q, r_head_k))         # qlen x rlen x bsz x n_head
        # else:
        BD = torch.einsum('ibnd,jnd->ijbn',
                          (rr_head_q, r_head_k))  # qlen x rlen x bsz x n_head

        if attn_relpos is None:
            BD = self._rel_shift(BD)
        else:
            # BD = self._rel_shift(BD)
            attn_relpos = torch.clip(attn_relpos, min_len, max_len).long()
            # print(attn_relpos.shape)
            # print(attn_relpos.min(), attn_relpos.max())
            # print(attn_relpos[0])
            attn_relpos = (max_len - attn_relpos).long()
            # print(rlen)
            # print(attn_relpos.size(0), rlen)
            # relpos_one_hot = torch.Tensor(F.one_hot(attn_relpos, num_classes=rlen)).float()               # bsz x qlen x klen x rlen
            # print(relpos_one_hot.shape)
            attn_relpos = attn_relpos.permute(1, 2, 0)

            BD = BD.gather(
                1,
                attn_relpos.unsqueeze(-1).expand(-1, -1, -1, BD.shape[-1]))
            # BD = torch.einsum('ijbn,bisj->isbn', BD, relpos_one_hot)                # qlen x klen x bsz x n_head

        attn_score = AC + BD
        attn_score.mul_(self.scale)

        #### compute attention probability
        if attn_mask is not None and attn_mask.any().item():
            if attn_mask.dim() == 2:
                attn_score = attn_score.float().masked_fill(
                    ~attn_mask[:, :, None, None],
                    -float('inf')).type_as(attn_score)
            elif attn_mask.dim() == 3:
                attn_score = attn_score.float().masked_fill(
                    ~attn_mask[:, :, :, None],
                    -float('inf')).type_as(attn_score)

        # [qlen x klen x bsz x n_head]
        attn_prob = F.softmax(attn_score, dim=1)
        attn_prob = self.drop(attn_prob)

        #### compute attention vector
        attn_vec = torch.einsum('ijbn,jbnd->ibnd', (attn_prob, w_head_v))

        # [qlen x bsz x n_head x d_head]
        attn_vec = attn_vec.contiguous().view(attn_vec.size(0),
                                              attn_vec.size(1),
                                              self.n_head * self.d_head)

        ##### linear projection
        attn_out = self.o_net(attn_vec)
        attn_out = self.dropatt(attn_out)

        if self.pre_lnorm:
            ##### residual connection
            output = w + attn_out
        else:
            ##### residual connection + layer normalization
            output = self.layer_norm(w + attn_out)

        return output


class TransformerGrammarLayer(nn.Module):

    def __init__(self, n_head, d_model, d_head, d_inner, dropoutf, dropouta,
                 **kwargs):
        super(TransformerGrammarLayer, self).__init__()

        self.dec_attn = RelPartialLearnableMultiHeadAttn(
            n_head, d_model, d_head, dropouta, **kwargs)
        self.pos_ff = PositionwiseFF(d_model,
                                     d_inner,
                                     dropoutf,
                                     pre_lnorm=kwargs.get('pre_lnorm'))

    def forward(self,
                dec_inp,
                r,
                r_w_bias,
                r_r_bias,
                attn_mask=None,
                attn_relpos=None,
                min_len=None,
                max_len=None,
                mems=None,
                terminal=False):
        output = self.dec_attn(dec_inp,
                               r,
                               r_w_bias,
                               r_r_bias,
                               attn_mask=attn_mask,
                               attn_relpos=attn_relpos,
                               min_len=min_len,
                               max_len=max_len,
                               mems=mems,
                               terminal=terminal)
        output = self.pos_ff(output)

        return output


class TransformerGrammar(nn.Module):

    def __init__(self,
                 vocab_size=10000,
                 w_dim=380,
                 n_head=10,
                 d_head=38,
                 d_inner=900,
                 num_layers=16,
                 dropout=0.1,
                 dropoutatt=0.0,
                 pad_id=0,
                 bos_id=1,
                 eos_id=2,
                 opening_id=None,
                 closing_id=None,
                 pre_lnorm=False):
        super(TransformerGrammar, self).__init__()
        self.vocab_size = vocab_size
        self.d_model = w_dim
        self.n_head = n_head
        self.d_head = d_head
        self.d_inner = d_inner
        self.num_layers = num_layers

        self.dropout = nn.Dropout(dropout)

        self.emb = nn.Embedding(vocab_size, w_dim)
        self.emb_scale = w_dim**0.5
        self.projection = nn.Linear(w_dim, vocab_size)
        self.projection.weight = self.emb.weight

        self.num_layers = num_layers
        self.w_dim = w_dim

        self.layers = nn.ModuleList()

        for _ in range(num_layers):
            self.layers.append(
                TransformerGrammarLayer(n_head,
                                        w_dim,
                                        d_head,
                                        d_inner,
                                        dropout,
                                        dropoutatt,
                                        tgt_len=None,
                                        ext_len=None,
                                        mem_len=None,
                                        pre_lnorm=pre_lnorm))

        self.pos_emb = PositionalEmbedding(w_dim)
        self.r_w_bias = nn.Parameter(torch.Tensor(self.n_head, self.d_head))
        self.r_r_bias = nn.Parameter(torch.Tensor(self.n_head, self.d_head))

        self.pad_id = pad_id
        self.bos_id = bos_id
        self.eos_id = eos_id
        self.opening_id = opening_id  # tuple (opening_start, opening_end + 1)
        self.closing_id = closing_id  # tuple (closing_start, closing_end + 1)

    def forward(
            self,
            input_batch,
            length,  # 句子的最大长度
            use_mask=True,
            document_level=False,
            return_h=False,
            return_prob=True,
            return_action_score=False,
            max_relative_length=None,
            min_relative_length=None):

        attn_mask = []
        attn_relpos = []
        inputs = []
        targets = []
        batch_size = len(input_batch)
        if use_mask == False:
            # print("Use_mask is False.")
            length_i = max([len(sent) for sent in input_batch])
            for sent in input_batch:
                src_ = sent[:-1]
                tgt_ = sent[1:]
                src_p = src_ + [self.pad_id] * (length_i - len(src_))
                inputs.append(np.array(src_p))
                tgt_p = tgt_ + [self.pad_id] * (length_i - len(tgt_))
                targets.append(np.array(tgt_p))
            inputs = torch.LongTensor(np.array(inputs)).cuda()
            targets = torch.LongTensor(np.array(targets)).cuda()

            attn_mask = torch.tril(
                torch.ones((length_i, length_i),
                           dtype=torch.uint8)).cuda().bool()
            attn_mask = attn_mask.unsqueeze(0).expand(batch_size, -1, -1)
            attn_relpos = None
        else:
            ranges = masking_utils.TokenTypeRanges(self.bos_id, 
                                                   self.pad_id,
                                                   self.eos_id,
                                                   self.opening_id,
                                                   self.closing_id)
            maskrules = masking_utils.get_masking_rules(
                "stack_compose_double_closing_nt",
                sequence_length=512,
                memory_length=512,
                transparency_prob=0.0,
                gather_into_new_memory=True,
                transparency_depth_threshold=-1)

            for sent in input_batch:
                src_ = torch.LongTensor(sent[:-1])
                tgt_ = torch.LongTensor(sent[1:])
                info_tuple = masking_utils.compute_token_types(
                    {
                        "inputs": src_,
                        "labels": tgt_
                    }, ranges)
                chunks = maskrules.chunks_for_sequence(
                    info_tuple['inputs'], info_tuple['inputs_ttypes'],
                    info_tuple['labels'], info_tuple['labels_ttypes'])
                chunks = [types.Chunk(None, *chunk) for chunk in chunks]

                if not document_level:
                    # only consider the first chunk
                    chunk = chunks[0]
                    src_p = chunk.inputs[:length - 1]
                    inputs.append(np.array(src_p))
                    tgt_p = chunk.labels[:length - 1]
                    targets.append(np.array(tgt_p))
                    mask = chunk.attn_mask[:length - 1, :length - 1]
                    for i in range(len(mask)):
                        mask[i, i] = 1
                    attn_mask.append(np.array(mask))
                    chunk_len = len(chunk.attn_mask[0])
                    relpos = chunk.attn_relpos[:len(mask),
                                               chunk_len:chunk_len + len(mask)]
                    attn_relpos.append(np.array(relpos))
                else:
                    pass  # TODO: Document level. Remain to be implemented.
            inputs = torch.LongTensor(np.array(inputs)).cuda()
            targets = torch.LongTensor(np.array(targets)).cuda()
            attn_mask = torch.LongTensor(np.array(attn_mask)).cuda().bool()
            attn_relpos = torch.LongTensor(np.array(attn_relpos)).cuda()

        inputs = inputs.permute(1, 0).contiguous()
        targets = targets.permute(1, 0).contiguous()
        attn_mask = attn_mask.permute(1, 2, 0).contiguous()

        seq_len = inputs.size(0)

        word_emb = self.emb(inputs)

        if use_mask == False:
            pos_emb = self.pos_emb(
                torch.arange(seq_len - 1, -1, -1.0, device=word_emb.device))
        else:
            if max_relative_length is None:
                max_relative_length = seq_len
            if min_relative_length is None:
                min_relative_length = -seq_len
            else:
                min_relative_length = min_relative_length - 1
            pos_emb = self.pos_emb(
                torch.arange(max_relative_length,
                             min_relative_length,
                             -1.0,
                             device=word_emb.device))

        core_out = self.dropout(word_emb)
        pos_emb = self.dropout(pos_emb)
        hiddens = []
        hiddens.append(core_out)
        for i, layer in enumerate(self.layers):
            core_out = layer(core_out,
                             pos_emb,
                             self.r_w_bias,
                             self.r_r_bias,
                             attn_mask=attn_mask,
                             attn_relpos=attn_relpos,
                             min_len=min_relative_length,
                             max_len=max_relative_length)
            hiddens.append(core_out)
            if i < len(self.layers) - 1:
                core_out = self.dropout(core_out)
        core_out = self.dropout(core_out)

        logits = self.projection(core_out)
        crit = nn.CrossEntropyLoss(reduction='none', ignore_index=self.pad_id)
        prob = logits.view(seq_len, batch_size, -1)
        # normalize
        prob = torch.sigmoid(prob).clamp(min=1e-8, max=1 - 1e-8)
        prob = prob.permute(0, 2, 1)
        
        # print("-" * 50)
        # print("prob shape: ", prob.shape)
        # print("targets shape: ", targets.shape)
        loss = crit(prob, targets)
        loss = loss.permute(1, 0).contiguous()
        loss = loss.sum(1)  # given by cross entropy
        # print("loss shape: ", loss.shape)
        if return_h:
            loss = loss.contiguous().view(-1, batch_size)
            return loss, core_out
        elif return_prob:
            loss = loss.contiguous().view(-1, batch_size)
            # prob = prob.contiguous().view(batch_size, -1)
            # print("-" * 50)
            # print("Return Prob = True.")
            # print("prob shape: ", prob.shape)
            # print("loss shape: ", loss.shape)
            # print("attn mask shape: ", attn_mask.shape)
            return loss, prob, attn_mask
        else:
            loss = loss.contiguous().view(-1, batch_size)
            return loss


class TransformerGrammarPlusQNet(nn.Module):

    def __init__(
        self,
        vocab_size=10000,
        w_dim=380,  # word embedding dim
        n_head=10,  # attn head number
        d_head=38,  # attn head dim
        d_inner=900,  # Dimension of Inner Layer in Position-wise Feedforward Net
        num_layers=16,  #
        dropout=0.1,
        dropoutatt=0.0,
        q_dim=20,
        pad_id=0,
        bos_id=1,
        eos_id=2,
        idx2word={},
        word2idx={},
        pos_max_len=250,
        # opening_id=None,
        # closing_id=None,
        # pre_lnorm=False,
    ):
        super(TransformerGrammarPlusQNet, self).__init__()
        self.pad_id = pad_id
        self.bos_id = bos_id
        self.eos_id = eos_id
        self.d_emb = w_dim
        self.word2idx = word2idx
        self.idx2word = idx2word
        self.S = 0  # action idx for shift/generate
        self.R = 1  # action idx for reduce
        self.tg_p_net = TransformerGrammar(vocab_size=vocab_size,
                                           w_dim=w_dim,
                                           n_head=n_head,
                                           d_head=d_head,
                                           d_inner=d_inner,
                                           num_layers=num_layers,
                                           dropout=dropout,
                                           dropoutatt=dropoutatt,
                                           pad_id=self.pad_id,
                                           bos_id=self.bos_id,
                                           eos_id=self.eos_id,
                                           opening_id=None,
                                           closing_id=None,
                                           pre_lnorm=False)

        self.left_arc = self.word2idx['(S']
        self.right_arc = self.word2idx['S)']
        
        from TreeCRF import ConstituencyTreeCRF
        self.q_crf = ConstituencyTreeCRF()
        self.q_pos_emb = nn.Embedding(pos_max_len,
                                      w_dim)  # pos embedding of Q Net
        self.q_dim = q_dim
        self.q_leaf_rnn = nn.LSTM(w_dim,
                                  q_dim,
                                  bidirectional=True,
                                  batch_first=True)
        self.q_binary = nn.Sequential(nn.Linear(q_dim * 2, q_dim * 2),
                                      nn.ReLU(), nn.LayerNorm(q_dim * 2),
                                      nn.Dropout(dropout),
                                      nn.Linear(q_dim * 2, 1))

        self.dropout = nn.Dropout(dropout)
        self.emb = nn.Embedding(vocab_size, w_dim)

    # utils
    def get_ranges(self, start_token, pad_token, end_token, left_arc, right_arc):
        return masking_utils.TokenTypeRanges(start_token, 
                                             pad_token, 
                                             end_token,
                                             left_arc,
                                             right_arc)

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

    def get_span_scores(self, x):
        # produces the span scores s_ij
        bos = x.new(x.size(0), 1).fill_(self.bos_id)
        eos = x.new(x.size(0), 1).fill_(self.eos_id)
        x = torch.cat([bos, x, eos], 1)
        x_vec = self.dropout(self.emb(x))
        pos = torch.arange(0,
                           x.size(1)).unsqueeze(0).expand_as(x).long().cuda()
        x_vec = x_vec + self.dropout(self.q_pos_emb(pos))
        q_h, _ = self.q_leaf_rnn(x_vec)
        fwd = q_h[:, 1:, :self.q_dim]
        bwd = q_h[:, :-1, self.q_dim:]
        fwd_diff = fwd[:, 1:].unsqueeze(1) - fwd[:, :-1].unsqueeze(2)
        bwd_diff = bwd[:, :-1].unsqueeze(2) - bwd[:, 1:].unsqueeze(1)
        concat = torch.cat([fwd_diff, bwd_diff], 3)
        scores = self.q_binary(concat).squeeze(3)
        return scores

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
                while i < len(tree_bracket) and tree_bracket[
                        i] >= '0' and tree_bracket[i] <= '9':
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
                while i < len(tree_bracket) and tree_bracket[
                        i] >= '0' and tree_bracket[i] <= '9':
                    id = id * 10 + int(tree_bracket[i])
                    i += 1
                tree_id.append(int(idx[id]))
                continue
            i += 1
        return tree_id

    # forward propagation
    def _forward_Q_CRF(self, scores, parse_length, batch_size, samples):
        # scores: scores = scores / is_temp
        self.q_crf._forward(scores)
        self.q_crf._entropy(scores)

        crf_input = scores.unsqueeze(1).expand(batch_size, samples,
                                               parse_length, parse_length)
        crf_input = crf_input.contiguous().view(batch_size * samples,
                                                parse_length, parse_length)
        for i in range(len(self.q_crf.alpha)):
            for j in range(len(self.q_crf.alpha)):
                self.q_crf.alpha[i][j] = self.q_crf.alpha[i][j].unsqueeze(
                    1).expand(batch_size,
                              samples).contiguous().view(batch_size * samples)
        return self.q_crf._sample(crf_input, self.q_crf.alpha)

    def _forward_TG(self,
                    input_batch,
                    length,
                    use_mask=True,
                    document_level=False,
                    return_h=False,
                    return_prob=True,
                    return_action_score=False,
                    max_relative_length=None,
                    min_relative_length=None):

        return self.tg_p_net(input_batch, length, use_mask, document_level,
                             return_h, return_prob, return_action_score, max_relative_length,
                             min_relative_length)

    def forward(
        self,
        x,
        samples=1,  # TODO: Figure out what is `samples`
        is_temp=1.,
        has_eos=True,
        mode='default',
    ):

        # prepare for masking and original input
        # print('-' * 50)
        # print("Preparing for Forwarding")
        # print("x shape: ", x.shape)
        x = x[:, 1:]
        # print("x shape after x=x[:,1:]  : ", x.shape)
        batch_size, length = x.size(0), x.size(1)
        # print("Batch Size: ", batch_size)
        # print("Length: ", length)
        ranges = self.get_ranges(self.bos_id, self.pad_id, self.eos_id, self.left_arc, self.right_arc)

        maskrules = masking_utils.get_masking_rules(
            "stack_compose_double_closing_nt",
            sequence_length=768,
            memory_length=768,
            transparency_prob=0.0,
            gather_into_new_memory=False,
            transparency_depth_threshold=-1)

        if has_eos:
            parse_length = length - 1
            parse_x = x[:, :-1]
        else:
            parse_length = length
            parse_x = x

        # q inference net forward. left/right(2 test trees)  /Q.
        if mode == 'left':
            tree_brackets = []
            for i in range(batch_size):
                tree = self.generate_left_tree(parse_x[i])
                tree_brackets.append(tree)
        elif mode == 'right':
            tree_brackets = []
            for i in range(batch_size):
                tree = self.generate_right_tree(parse_x[i])
                tree_brackets.append(tree)
        else:
            # use Q Inference Net to get the label.
            # FIXME: Maybe some problems because 我把QNet的推理单独提进新的函数了
            scores = self.get_span_scores(parse_x)
            self.scores = scores
            scores = scores / is_temp
            # print("Q Inference CRF Net Forwarding")
            _, log_probs_action_q, tree_brackets, spans = self._forward_Q_CRF(
                scores, parse_length, batch_size, samples)
            # print("Tree Brackets: ", tree_brackets)
            entropy = self.q_crf.entropy[0][parse_length - 1]
            

        # prepare for p tg net forward + process q net output
        # attn_masks = []
        # attn_relpos = []
        inputs = []
        labels = []
        actions = []
        max_len_tmp = 0
        # print("Tree brackets: ", tree_brackets)
        # print("Preparing input for TG Net + Processing Q Net output")
        for b in range(batch_size * samples):
            # add NT
            from utils import get_actions
            action = get_actions(tree_brackets[b])
            if has_eos:
                actions.append(
                    action + [self.S, self.R]
                )  # we train the model to generate <s> and then do a final reduce
            else:
                actions.append(action)

            if has_eos:
                sent_str = self.get_tree_str(parse_x[b // samples],
                                             tree_brackets[b])
                sent_id = np.array(self.get_tree_id(parse_x[b // samples],
                                                    tree_brackets[b]),
                                   dtype=np.int32)
            else:
                sent_str = self.get_tree_str(parse_x[b // samples],
                                             tree_brackets[b])
                sent_id = np.array(self.get_tree_id(parse_x[b // samples],
                                                    tree_brackets[b]),
                                   dtype=np.int32)

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
            len_inp = len(input)

            len_inp_processed = (len_inp * 4 - 2) // 3
            input_processed = np.array(chunks[0][0][0:len_inp_processed])
            inputs.append(input_processed)
            label_processed = np.array(chunks[0][2][0:len_inp_processed])
            labels.append(label_processed)
            # attn_mask = np.array(chunks[0][4])
            # l_chunk = len(attn_mask[0])
            # attn_mask = attn_mask[0:len_inp_processed, 0:len_inp_processed]
            # attn_masks.append(attn_mask)
            # attn_relpos.append(np.array(chunks[0][5])[0: len(attn_mask), \
                            #    l_chunk: l_chunk + len(attn_mask)])

        # attn_masks = np.array(attn_masks)
        # attn_relpos = np.array(attn_relpos)

        actions = torch.Tensor(actions).float().cuda()
        # print("Actions in model before return: ", actions)
        # attn_masks = torch.LongTensor(attn_masks).cuda()  # B * l_inp * l_inp
        # attn_relpos = torch.LongTensor(attn_relpos).cuda()  # B * l_inp * l_inp

        inputs = np.array(inputs)
        labels = np.array(labels)

        inp = torch.LongTensor(inputs.T).cuda()  # l_inp * B
        tgt = torch.LongTensor(labels.T).cuda()  # l_tgt(=l_inp) * B
        tgt_len = tgt.size(0)
        inp_len = inp.size(0)
        batch_expand = batch_size * samples

        # p tg net forward
        # 上面的 attn_masks 和 attn_relpos 已经在 _forward_TG 中创建与填充，不需要了
        loss, log_probs_action_p, tg_attn_mask = self._forward_TG(input_batch=inputs, 
                                                    length=inp_len, 
                                                    use_mask=True,
                                                    document_level=False,
                                                    return_h=False,
                                                    return_prob=True,
                                                    return_action_score=False,
                                                    min_relative_length=None, 
                                                    max_relative_length=None
                                                    )
        log_p = -loss
        # print("-" * 50)
        # print("TG Net Forwarding Done")
        # print("log_p shape: ", log_p.shape)
        # print("log_probs_action_p shape: ", log_probs_action_p.shape)
        # return
        if mode not in ['left', 'right']:
            log_probs_action_q = log_probs_action_q.contiguous().view(
                batch_size, samples)
            return log_p, log_probs_action_p, log_probs_action_q, actions, entropy, tg_attn_mask
        else:
            return log_p, log_probs_action_p, None, actions, None, tg_attn_mask
