#!/usr/bin/env python3

import argparse
import time

import torch
from torch import cuda

from data import Dataset
from models import RNNG
from utils import *
from tg_model import TransformerGrammar, TransformerGrammarPlusQNet

parser = argparse.ArgumentParser()

# Data path options
parser.add_argument('--train_file', default='data/ptb-1unk-train.pkl')
parser.add_argument('--val_file', default='data/ptb-1unk-val.pkl')
parser.add_argument('--ckpt_path', default='')
# Model options
parser.add_argument(
    '--w_dim',
    default=380,
    type=int,
    help='Hidden dimension for LM/TG. Word Embedding Dimension')
parser.add_argument('--num_layers',
                    default=16,
                    type=int,
                    help='Number of TG Layers and the stack LSTM (for RNNG)')
parser.add_argument('--dropout',
                    default=0.5,
                    type=float,
                    help='Dropout rate for Embedding, Position Encoding and\
                    TG Decoder Layers.')
parser.add_argument('--n_head',
                    default=10,
                    type=int,
                    help='Number of Attention Heads.')
parser.add_argument('--d_head',
                    default=38,
                    type=int,
                    help='Dimension of Attention Heads.')
parser.add_argument(
    '--d_inner',
    default=900,
    type=int,
    help='Dimension of Inner Layer in Position-wise Feedforward Net.')
parser.add_argument('--dropoutatt',
                    default=0.1,
                    type=float,
                    help='Dropout rate for Attention Layer.')
parser.add_argument('--q_dim',
                    default=20,
                    type=int,
                    help='Hidden dimension for Leaf LSTM in Q Inference Net')

# Optimization options
parser.add_argument('--count_eos_ppl',
                    default=0,
                    type=int,
                    help='whether to count eos in val PPL')
parser.add_argument('--save_path',
                    default='./ckpt/tg.pt',
                    help='where to save the data')
parser.add_argument('--num_epochs',
                    default=18,
                    type=int,
                    help='number of training epochs')
parser.add_argument(
    '--warmup_epochs',
    default=8,
    type=int,
    help='do not decay learning rate for at least this many epochs')
parser.add_argument('--mode',
                    default='unsupervised',
                    type=str,
                    choices=['unsupervised', 'supervised'])
parser.add_argument('--mc_samples',
                    default=5,
                    type=int,
                    help='how many samples for IWAE bound calc for evaluation')
parser.add_argument('--samples',
                    default=8,
                    type=int,
                    help='how many samples for score function gradients')
parser.add_argument('--lr',
                    default=1,
                    type=float,
                    help='starting learning rate')
parser.add_argument('--q_lr',
                    default=0.0001,
                    type=float,
                    help='learning rate for inference network q')
# parser.add_argument('--action_lr',
#                     default=0.1,
#                     type=float,
#                     help='learning rate for action layer')
parser.add_argument(
    '--lr_decay',
    default=0.5,
    type=float,
    help='After warmup_epochs, we have lr decayed by this param.')
parser.add_argument('--kl_warmup',
                    default=2,
                    type=int,
                    help='KL-Annealing Decay.')
parser.add_argument('--train_q_epochs', default=2, type=int, help='')
parser.add_argument('--param_init',
                    default=0.1,
                    type=float,
                    help='parameter initialization (over uniform)')
parser.add_argument('--max_grad_norm',
                    default=5,
                    type=float,
                    help='gradient clipping parameter')
parser.add_argument('--q_max_grad_norm',
                    default=1,
                    type=float,
                    help='gradient clipping parameter for q')
parser.add_argument('--gpu', default=0, type=int, help='which gpu to use')
parser.add_argument('--seed', default=3407, type=int, help='random seed')
parser.add_argument('--print_every',
                    type=int,
                    default=500,
                    help='print stats after this many batches')


def tg_main(args):
    # 0. Preprocessing
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    train_data = Dataset(args.train_file)
    val_data = Dataset(args.val_file)
    idx2word = train_data.idx2word
    word2idx = train_data.word2idx
    vocab_size = int(train_data.vocab_size)
    print('Train: %d sents / %d batches, Val: %d sents / %d batches' %
          (train_data.sents.size(0), len(train_data), val_data.sents.size(0),
           len(val_data)))
    print('Vocab size: %d' % vocab_size)
    if torch.cuda.is_available():
        try:
            cuda.set_device(args.gpu)
        except Exception as e:
            print(f"We caught an exception, but that doesn't matter: {e}")
        device = f'cuda:{args.gpu}'
    else:
        device = 'cpu'
    if args.ckpt_path == '':
        model = TransformerGrammarPlusQNet(
            vocab_size=vocab_size,
            w_dim=args.w_dim,
            n_head=args.n_head,
            d_head=args.d_head,
            d_inner=args.d_inner,
            dropoutatt=args.dropoutatt,
            dropout=args.dropout,
            num_layers=args.num_layers,
            q_dim=args.q_dim,
            idx2word=idx2word,
            word2idx=word2idx,
        )
        if args.param_init > 0:
            for param in model.parameters():
                param.data.uniform_(-args.param_init, args.param_init)
    else:
        print('loading model from ' + args.ckpt_path)
        checkpoint = torch.load(args.ckpt_path)
        model = checkpoint['model']
    # print("model architecture")
    # print(model)

    # 1. 把模型参数分成2部分，分别是：model_params（给主模型）, q_params（给adversarial inference net）
    # action params 被 TransformerGrammarPlusQNet 代替了
    q_params = []
    # action_params = []
    model_params = []
    for name, param in model.named_parameters():
        # if 'action' in name:
            # print(name)
            # action_params.append(param)
        if 'q_' in name:
            # print(name)
            q_params.append(param)
        else:
            model_params.append(param)
    q_lr = args.q_lr
    optimizer = torch.optim.SGD(model_params, lr=args.lr)
    q_optimizer = torch.optim.Adam(q_params, lr=q_lr)
    # action_optimizer = torch.optim.SGD(action_params, lr=args.action_lr)
    model.train()
    # model.cuda(device=device)
    model.to(torch.device(device))

    epoch = 0
    lr_decay = 0
    if args.kl_warmup > 0:
        kl_pen = 0.
        kl_warmup_batch = 1. / (args.kl_warmup * len(train_data))
    else:
        kl_pen = 1.
    best_val_ppl = 5e5
    best_val_f1 = 0
    samples = args.samples
    best_val_ppl, best_val_f1 = tg_eval(val_data,
                                     model,
                                     samples=args.mc_samples,
                                     count_eos_ppl=args.count_eos_ppl)
    all_stats = [[0., 0., 0.]]  # true pos, false pos, false neg for f1 calc
    while epoch < args.num_epochs:
        start_time = time.time()
        epoch += 1
        if epoch > args.train_q_epochs:
            # stop training q after this many epochs
            args.q_lr = 0.
            for param_group in q_optimizer.param_groups:
                param_group['lr'] = args.q_lr
        print('Starting epoch %d' % epoch)
        train_nll_recon = 0.
        train_nll_iwae = 0.
        train_kl = 0.
        train_q_entropy = 0.
        num_sents = 0.
        num_words = 0.
        b = 0

        #
        for i in np.random.permutation(len(train_data)):
            if args.kl_warmup > 0:
                kl_pen = min(1., kl_pen + kl_warmup_batch)
            sents, length, batch_size, gold_actions, gold_spans, gold_binary_trees, other_data = train_data[
                i]
            if length == 1:
                # we ignore length 1 sents during training/eval since we work with binary trees only
                continue
            sents = sents.cuda(device=device)
            b += 1
            q_optimizer.zero_grad()
            optimizer.zero_grad()
            # action_optimizer.zero_grad()
            if args.mode == 'unsupervised':
                ll_word, ll_action_p, ll_action_q, all_actions, q_entropy = model(
                    sents, samples=samples, has_eos=True)
                log_f = ll_word + kl_pen * ll_action_p
                iwae_ll = log_f.mean(1).detach() + kl_pen * q_entropy.detach()
                obj = log_f.mean(1)
                if epoch < args.train_q_epochs:
                    obj += kl_pen * q_entropy
                    baseline = torch.zeros_like(log_f)
                    baseline_k = torch.zeros_like(log_f)
                    for k in range(samples):
                        baseline_k.copy_(log_f)
                        baseline_k[:, k].fill_(0)
                        baseline[:,
                                 k] = baseline_k.detach().sum(1) / (samples -
                                                                    1)
                    obj += ((log_f.detach() - baseline.detach()) *
                            ll_action_q).mean(1)
                kl = (ll_action_q - ll_action_p).mean(1).detach()
                ll_word = ll_word.mean(1)
                train_q_entropy += q_entropy.sum().item()
            else:
                gold_actions = gold_binary_trees
                ll_action_q = model.forward_tree(sents,
                                                 gold_actions,
                                                 has_eos=True)
                ll_word, ll_action_p, all_actions = model.forward_actions(
                    sents, gold_actions)
                obj = ll_word + ll_action_p + ll_action_q
                kl = -ll_action_q
                iwae_ll = ll_word + ll_action_p
            train_nll_iwae += -iwae_ll.sum().item()
            actions = all_actions[:, 0].long().cpu()
            train_nll_recon += -ll_word.sum().item()
            train_kl += kl.sum().item()
            (-obj.mean()).backward()
            if args.max_grad_norm > 0:
                # torch.nn.utils.clip_grad_norm_(model_params + action_params,  # FIXME: OK?
                torch.nn.utils.clip_grad_norm_(model_params,
                                               args.max_grad_norm)
            if args.q_max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(q_params, args.q_max_grad_norm)
            q_optimizer.step()
            optimizer.step()
            # action_optimizer.step()
            num_sents += batch_size
            num_words += batch_size * length
            for bb in range(batch_size):
                action = list(actions[bb].numpy())
                span_b = get_spans(action)
                span_b_set = set(
                    span_b[:-1])  # ignore the sentence-level trivial span
                update_stats(span_b_set, [set(gold_spans[bb][:-1])], all_stats)
            if b % args.print_every == 0:
                all_f1 = get_f1(all_stats)
                param_norm = sum([p.norm()**2
                                  for p in model.parameters()]).item()**0.5
                log_str = 'Epoch: %d, Batch: %d/%d, LR: %.4f, qLR: %.5f, qEntropy: %.4f, Train VAE Perplexity: %.2f, ' + \
                          'Train Reconstrction Perplexity: %.2f, Train KL: %.2f, Train IWAE Perplexity: %.2f, ' + \
                          '|Param| (Norm of the whole param vector): %.2f, Best Validation Perplexity: %.2f, Best Val F1 Score: %.2f, KL Penalty: %.4f, ' + \
                          'Gold Tree F1 Score: %.2f, Throughput: %.2f examples/sec'
                print(
                    log_str %
                    (epoch, b, len(train_data), args.lr, args.q_lr,
                     train_q_entropy / num_sents,
                     np.exp((train_nll_recon + train_kl) / num_words),
                     np.exp(train_nll_recon / num_words), train_kl / num_sents,
                     np.exp(train_nll_iwae / num_words), param_norm,
                     best_val_ppl, best_val_f1, kl_pen, all_f1[0], num_sents /
                     (time.time() - start_time)))
                sent_str = [
                    train_data.idx2word[word_idx]
                    for word_idx in list(sents[-1][1:-1].cpu().numpy())
                ]
                print("PRED:", get_tree(action[:-2], sent_str))
                print("GOLD:", get_tree(gold_binary_trees[-1], sent_str))
        print('--------------------------------')
        print('Checking validation performance...')
        val_ppl, val_f1 = tg_eval(val_data,
                               model,
                               samples=args.mc_samples,
                               count_eos_ppl=args.count_eos_ppl)
        print('--------------------------------')
        if val_ppl < best_val_ppl:
            best_val_ppl = val_ppl
            best_val_f1 = val_f1
            checkpoint = {
                'args': args.__dict__,
                'model': model.cpu(),
                'word2idx': train_data.word2idx,
                'idx2word': train_data.idx2word
            }
            print('Saving checkpoint to %s' % args.save_path)
            torch.save(checkpoint, args.save_path)
            model.cuda(device=device)
        else:  # ppl is not decreasing
            if epoch > args.warmup_epochs:
                lr_decay = 1
        if lr_decay == 1:
            args.lr = args.lr_decay * args.lr
            args.q_lr = args.lr_decay * args.q_lr
            # args.action_lr = args.lr_decay * args.action_lr
            for param_group in optimizer.param_groups:
                param_group['lr'] = args.lr
            for param_group in q_optimizer.param_groups:
                param_group['lr'] = args.q_lr
            # for param_group in action_optimizer.param_groups:
            #     param_group['lr'] = args.action_lr
        if args.lr < 0.03:
            break
    print("Finished training!")

def tg_eval(data, model, samples=0, count_eos_ppl=0):
    model.eval()
    num_sents = 0
    num_words = 0
    total_nll_recon = 0.
    total_kl = 0.
    total_nll_iwae = 0.
    corpus_f1 = [0., 0., 0.]
    sent_f1 = []
    with torch.no_grad():
        for i in list(reversed(range(len(data)))):
            sents, length, batch_size, gold_actions, gold_spans, gold_binary_trees, other_data = data[
                i]
            if length == 1:  # length 1 sents are ignored since URNNG needs at least length 2 sents
                continue
            if args.count_eos_ppl == 1:
                tree_length = length
                length += 1
            else:
                sents = sents[:, :-1]
                tree_length = length
            sents = sents.cuda()
            ll_word_all, ll_action_p_all, ll_action_q_all, actions_all, q_entropy = model(
                sents, samples=samples, has_eos=count_eos_ppl == 1)
            ll_word, ll_action_p, ll_action_q = ll_word_all.mean(
                1), ll_action_p_all.mean(1), ll_action_q_all.mean(1)
            kl = ll_action_q - ll_action_p
            _, binary_matrix, argmax_spans = model.q_crf._viterbi(model.scores)
            actions = []
            for b in range(batch_size):
                tree = get_tree_from_binary_matrix(binary_matrix[b],
                                                   tree_length)
                actions.append(get_actions(tree))
            actions = torch.Tensor(actions).long()
            total_nll_recon += -ll_word.sum().item()
            total_kl += kl.sum().item()
            num_sents += batch_size
            num_words += batch_size * length
            if samples > 0:
                # PPL estimate based on IWAE
                sample_ll = torch.zeros(batch_size, samples)
                for j in range(samples):
                    ll_word_j, ll_action_p_j, ll_action_q_j = ll_word_all[:,
                                                                          j], ll_action_p_all[:,
                                                                                              j], ll_action_q_all[:,
                                                                                                                  j]
                    sample_ll[:, j].copy_(ll_word_j + ll_action_p_j -
                                          ll_action_q_j)
                ll_iwae = model.logsumexp(sample_ll, 1) - np.log(samples)
                total_nll_iwae -= ll_iwae.sum().item()
            for b in range(batch_size):
                action = list(actions[b].numpy())
                span_b = get_spans(action)
                span_b = argmax_spans[b]
                span_b_set = set(span_b[:-1])
                gold_b_set = set(gold_spans[b][:-1])
                tp, fp, fn = get_stats(span_b_set, gold_b_set)
                corpus_f1[0] += tp
                corpus_f1[1] += fp
                corpus_f1[2] += fn

                # sent-level F1 is based on L83-89 from https://github.com/yikangshen/PRPN/test_phrase_grammar.py
                model_out = span_b_set
                std_out = gold_b_set
                overlap = model_out.intersection(std_out)
                prec = float(len(overlap)) / (len(model_out) + 1e-8)
                reca = float(len(overlap)) / (len(std_out) + 1e-8)
                if len(std_out) == 0:
                    reca = 1.
                    if len(model_out) == 0:
                        prec = 1.
                f1 = 2 * prec * reca / (prec + reca + 1e-8)
                sent_f1.append(f1)
    tp, fp, fn = corpus_f1
    prec = tp / (tp + fp)
    recall = tp / (tp + fn)
    corpus_f1 = 2 * prec * recall / (prec +
                                     recall) * 100 if prec + recall > 0 else 0.
    sent_f1 = np.mean(np.array(sent_f1)) * 100

    elbo_ppl = np.exp((total_nll_recon + total_kl) / num_words)
    recon_ppl = np.exp(total_nll_recon / num_words)
    iwae_ppl = np.exp(total_nll_iwae / num_words)
    kl = total_kl / num_sents
    print(
        'ElboPPL: %.2f, ReconPPL: %.2f, KL: %.4f, IwaePPL: %.2f, CorpusF1: %.2f, SentAvgF1: %.2f'
        % (elbo_ppl, recon_ppl, kl, iwae_ppl, corpus_f1, sent_f1))
    # note that corpus F1 printed here is different from what you should get from
    # evalb since we do not ignore any tags (e.g. punctuation), while evalb ignores it
    model.train()
    return iwae_ppl, corpus_f1


if __name__ == '__main__':
    args = parser.parse_args()
    # urnng_main(args)
    tg_main(args)

def urnng_main(args):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    train_data = Dataset(args.train_file)
    val_data = Dataset(args.val_file)
    vocab_size = int(train_data.vocab_size)
    print('Train: %d sents / %d batches, Val: %d sents / %d batches' %
          (train_data.sents.size(0), len(train_data), val_data.sents.size(0),
           len(val_data)))
    print('Vocab size: %d' % vocab_size)
    cuda.set_device(args.gpu)
    if args.ckpt_path == '':
        model = RNNG(vocab=vocab_size,
                     w_dim=args.w_dim,
                     h_dim=args.h_dim,
                     dropout=args.dropout,
                     num_layers=args.num_layers,
                     q_dim=args.q_dim)
        if args.param_init > 0:
            for param in model.parameters():
                param.data.uniform_(-args.param_init, args.param_init)
    else:
        print('loading model from ' + args.ckpt_path)
        checkpoint = torch.load(args.ckpt_path)
        model = checkpoint['model']
    print("model architecture")
    print(model)
    q_params = []
    action_params = []
    model_params = []
    for name, param in model.named_parameters():
        if 'action' in name:
            print(name)
            action_params.append(param)
        elif 'q_' in name:
            print(name)
            q_params.append(param)
        else:
            model_params.append(param)
    q_lr = args.q_lr

    # optimizer to update model parameters, q parameters and action parameters
    optimizer = torch.optim.SGD(model_params, lr=args.lr)
    q_optimizer = torch.optim.Adam(q_params, lr=q_lr)
    action_optimizer = torch.optim.SGD(action_params, lr=args.action_lr)
    model.train()
    model.cuda()

    epoch = 0
    lr_decay = 0
    if args.kl_warmup > 0:
        kl_pen = 0.
        kl_warmup_batch = 1. / (args.kl_warmup * len(train_data))
    else:
        kl_pen = 1.
    best_val_ppl = 5e5
    best_val_f1 = 0
    samples = args.samples
    best_val_ppl, best_val_f1 = eval(val_data,
                                     model,
                                     samples=args.mc_samples,
                                     count_eos_ppl=args.count_eos_ppl)
    all_stats = [[0., 0., 0.]]  # true pos, false pos, false neg for f1 calc

    # start training
    while epoch < args.num_epochs:
        start_time = time.time()
        epoch += 1
        if epoch > args.train_q_epochs:
            # stop training q after this many epochs
            args.q_lr = 0.
            for param_group in q_optimizer.param_groups:
                param_group['lr'] = args.q_lr
        print('Starting epoch %d' % epoch)
        train_nll_recon = 0.
        train_nll_iwae = 0.
        train_kl = 0.
        train_q_entropy = 0.
        num_sents = 0.
        num_words = 0.
        b = 0
        for i in np.random.permutation(len(train_data)):
            if args.kl_warmup > 0:
                kl_pen = min(1., kl_pen + kl_warmup_batch)
            sents, length, batch_size, gold_actions, gold_spans, gold_binary_trees, other_data = train_data[
                i]
            if length == 1:
                # we ignore length 1 sents during training/eval since we work with binary trees only
                continue
            sents = sents.cuda()
            b += 1
            q_optimizer.zero_grad()
            optimizer.zero_grad()
            action_optimizer.zero_grad()
            if args.mode == 'unsupervised':
                print('Doing unsupervised training.')
                ll_word, ll_action_p, ll_action_q, all_actions, q_entropy = model(
                    sents, samples=samples, has_eos=True)
                log_f = ll_word + kl_pen * ll_action_p
                iwae_ll = log_f.mean(1).detach() + kl_pen * q_entropy.detach()
                obj = log_f.mean(1)
                if epoch < args.train_q_epochs:
                    obj += kl_pen * q_entropy
                    baseline = torch.zeros_like(log_f)
                    baseline_k = torch.zeros_like(log_f)
                    for k in range(samples):
                        baseline_k.copy_(log_f)
                        baseline_k[:, k].fill_(0)
                        baseline[:,
                                 k] = baseline_k.detach().sum(1) / (samples -
                                                                    1)
                    obj += ((log_f.detach() - baseline.detach()) *
                            ll_action_q).mean(1)
                kl = (ll_action_q - ll_action_p).mean(1).detach()
                ll_word = ll_word.mean(1)
                train_q_entropy += q_entropy.sum().item()
            else:
                print('Doing plain supervised training.')
                gold_actions = gold_binary_trees
                ll_action_q = model.forward_tree(sents,
                                                 gold_actions,
                                                 has_eos=True)
                ll_word, ll_action_p, all_actions = model.forward_actions(
                    sents, gold_actions)
                obj = ll_word + ll_action_p + ll_action_q
                kl = -ll_action_q
                iwae_ll = ll_word + ll_action_p
            train_nll_iwae += -iwae_ll.sum().item()
            actions = all_actions[:, 0].long().cpu()
            train_nll_recon += -ll_word.sum().item()
            train_kl += kl.sum().item()
            (-obj.mean()).backward()
            if args.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(model_params + action_params,
                                               args.max_grad_norm)
            if args.q_max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(q_params, args.q_max_grad_norm)
            q_optimizer.step()
            optimizer.step()
            action_optimizer.step()
            num_sents += batch_size
            num_words += batch_size * length
            for bb in range(batch_size):
                action = list(actions[bb].numpy())
                span_b = get_spans(action)
                span_b_set = set(
                    span_b[:-1])  # ignore the sentence-level trivial span
                update_stats(span_b_set, [set(gold_spans[bb][:-1])], all_stats)
            if b % args.print_every == 0:
                all_f1 = get_f1(all_stats)
                param_norm = sum([p.norm()**2
                                  for p in model.parameters()]).item()**0.5
                log_str = 'Epoch: %d, Batch: %d/%d, LR: %.4f, qLR: %.5f, qEnt: %.4f, TrainVAEPPL: %.2f, ' + \
                          'TrainReconPPL: %.2f, TrainKL: %.2f, TrainIWAEPPL: %.2f, ' + \
                          '|Param|: %.2f, BestValPerf: %.2f, BestValF1: %.2f, KLPen: %.4f, ' + \
                          'GoldTreeF1: %.2f, Throughput: %.2f examples/sec'
                print(
                    log_str %
                    (epoch, b, len(train_data), args.lr, args.q_lr,
                     train_q_entropy / num_sents,
                     np.exp((train_nll_recon + train_kl) / num_words),
                     np.exp(train_nll_recon / num_words), train_kl / num_sents,
                     np.exp(train_nll_iwae / num_words), param_norm,
                     best_val_ppl, best_val_f1, kl_pen, all_f1[0], num_sents /
                     (time.time() - start_time)))
                sent_str = [
                    train_data.idx2word[word_idx]
                    for word_idx in list(sents[-1][1:-1].cpu().numpy())
                ]
                print("PRED:", get_tree(action[:-2], sent_str))
                print("GOLD:", get_tree(gold_binary_trees[-1], sent_str))
        print('--------------------------------')
        print('Checking validation perf...')
        val_ppl, val_f1 = eval(val_data,
                               model,
                               samples=args.mc_samples,
                               count_eos_ppl=args.count_eos_ppl)
        print('--------------------------------')
        if val_ppl < best_val_ppl:
            best_val_ppl = val_ppl
            best_val_f1 = val_f1
            checkpoint = {
                'args': args.__dict__,
                'model': model.cpu(),
                'word2idx': train_data.word2idx,
                'idx2word': train_data.idx2word
            }
            print('Saving checkpoint to %s' % args.save_path)
            torch.save(checkpoint, args.save_path)
            model.cuda()
        else:
            if epoch > args.warmup_epochs:
                lr_decay = 1
        if lr_decay == 1:
            args.lr = args.lr_decay * args.lr
            args.q_lr = args.lr_decay * args.q_lr
            args.action_lr = args.lr_decay * args.action_lr
            for param_group in optimizer.param_groups:
                param_group['lr'] = args.lr
            for param_group in q_optimizer.param_groups:
                param_group['lr'] = args.q_lr
            for param_group in action_optimizer.param_groups:
                param_group['lr'] = args.action_lr
        if args.lr < 0.03:
            break
    print("Finished training!")


def eval(data, model, samples=0, count_eos_ppl=0):
    model.eval()
    num_sents = 0
    num_words = 0
    total_nll_recon = 0.
    total_kl = 0.
    total_nll_iwae = 0.
    corpus_f1 = [0., 0., 0.]
    sent_f1 = []
    with torch.no_grad():
        for i in list(reversed(range(len(data)))):
            sents, length, batch_size, gold_actions, gold_spans, gold_binary_trees, other_data = data[
                i]
            if length == 1:  # length 1 sents are ignored since URNNG needs at least length 2 sents
                continue
            if args.count_eos_ppl == 1:
                tree_length = length
                length += 1
            else:
                sents = sents[:, :-1]
                tree_length = length
            sents = sents.cuda()
            ll_word_all, ll_action_p_all, ll_action_q_all, actions_all, q_entropy = model(
                sents, samples=samples, has_eos=count_eos_ppl == 1)
            ll_word, ll_action_p, ll_action_q = ll_word_all.mean(
                1), ll_action_p_all.mean(1), ll_action_q_all.mean(1)
            kl = ll_action_q - ll_action_p
            _, binary_matrix, argmax_spans = model.q_crf._viterbi(model.scores)
            actions = []
            for b in range(batch_size):
                tree = get_tree_from_binary_matrix(binary_matrix[b],
                                                   tree_length)
                actions.append(get_actions(tree))
            actions = torch.Tensor(actions).long()
            total_nll_recon += -ll_word.sum().item()
            total_kl += kl.sum().item()
            num_sents += batch_size
            num_words += batch_size * length
            if samples > 0:
                # PPL estimate based on IWAE
                sample_ll = torch.zeros(batch_size, samples)
                for j in range(samples):
                    ll_word_j, ll_action_p_j, ll_action_q_j = ll_word_all[:,
                                                                          j], ll_action_p_all[:,
                                                                                              j], ll_action_q_all[:,
                                                                                                                  j]
                    sample_ll[:, j].copy_(ll_word_j + ll_action_p_j -
                                          ll_action_q_j)
                ll_iwae = model.logsumexp(sample_ll, 1) - np.log(samples)
                total_nll_iwae -= ll_iwae.sum().item()
            for b in range(batch_size):
                action = list(actions[b].numpy())
                span_b = get_spans(action)
                span_b = argmax_spans[b]
                span_b_set = set(span_b[:-1])
                gold_b_set = set(gold_spans[b][:-1])
                tp, fp, fn = get_stats(span_b_set, gold_b_set)
                corpus_f1[0] += tp
                corpus_f1[1] += fp
                corpus_f1[2] += fn

                # sent-level F1 is based on L83-89 from https://github.com/yikangshen/PRPN/test_phrase_grammar.py
                model_out = span_b_set
                std_out = gold_b_set
                overlap = model_out.intersection(std_out)
                prec = float(len(overlap)) / (len(model_out) + 1e-8)
                reca = float(len(overlap)) / (len(std_out) + 1e-8)
                if len(std_out) == 0:
                    reca = 1.
                    if len(model_out) == 0:
                        prec = 1.
                f1 = 2 * prec * reca / (prec + reca + 1e-8)
                sent_f1.append(f1)
    tp, fp, fn = corpus_f1
    prec = tp / (tp + fp)
    recall = tp / (tp + fn)
    corpus_f1 = 2 * prec * recall / (prec +
                                     recall) * 100 if prec + recall > 0 else 0.
    sent_f1 = np.mean(np.array(sent_f1)) * 100

    elbo_ppl = np.exp((total_nll_recon + total_kl) / num_words)
    recon_ppl = np.exp(total_nll_recon / num_words)
    iwae_ppl = np.exp(total_nll_iwae / num_words)
    kl = total_kl / num_sents
    print(
        'ElboPPL: %.2f, ReconPPL: %.2f, KL: %.4f, IwaePPL: %.2f, CorpusF1: %.2f, SentAvgF1: %.2f'
        % (elbo_ppl, recon_ppl, kl, iwae_ppl, corpus_f1, sent_f1))
    # note that corpus F1 printed here is different from what you should get from
    # evalb since we do not ignore any tags (e.g. punctuation), while evalb ignores it
    model.train()
    return iwae_ppl, corpus_f1


