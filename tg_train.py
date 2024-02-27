#!/usr/bin/env python3

import argparse
import time

import torch
from torch import cuda
from tqdm import tqdm

from data import Dataset
from utils import *
from tg_model import TransformerGrammar, TransformerGrammarPlusQNet
import wandb

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
                    default=12,
                    type=int,
                    help='Number of TG Layers')
parser.add_argument('--dropout',
                    default=0.4,
                    type=float,
                    help='Dropout rate for Embedding, Position Encoding and\
                    TG Decoder Layers.')
parser.add_argument('--n_head',
                    default=10,
                    type=int,
                    help='Number of Attention Heads.')
parser.add_argument('--d_head',
                    default=30,
                    type=int,
                    help='Dimension of Attention Heads.')
parser.add_argument(
    '--d_inner',
    default=900,
    type=int,
    help='Dimension of Inner Layer in Position-wise Feedforward Net.')
parser.add_argument('--dropoutatt',
                    default=0.2,
                    type=float,
                    help='Dropout rate for Attention Layer.')
parser.add_argument('--q_dim',
                    default=300,
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
                    default=5,
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
parser.add_argument('--eval_samples',
                    default=3,
                    type=int,
                    help='how many samples for evaluation Monte Carlo Sampling')
parser.add_argument('--samples',
                    default=10,
                    type=int,
                    help='how many samples for training Monte Carlo Sampling')
parser.add_argument('--lr',
                    default=1,
                    type=float,
                    help='starting learning rate')
parser.add_argument('--q_lr',
                    default=1e-4,
                    type=float,
                    help='learning rate for inference network q')
parser.add_argument('--lr_decay',
                    default=0.8,
                    type=float,
                    help='After warmup_epochs, we have lr decayed by this param.')
parser.add_argument('--kl_cost_annealing_warmup',
                    default=1.2,
                    type=int,
                    help='KL Cost Annealing, trying to solve KL Vanishing Problem i.e. Posterior Collapse')
parser.add_argument('--kl_pen_max',
                    default=0.2,
                    type=float,
                    help='maximum KL penalty')
parser.add_argument('--train_q_epochs', default=1, type=int, help='Max number of epoch to train Q-Net. After these epochs, only TG will be trained.')
parser.add_argument('--train_q_steps', default=3000, type=int, help='Max number of epoch to train Q-Net. After these steps, only TG will be trained.')
parser.add_argument('--param_init',
                    default=0.1,
                    type=float,
                    help='parameter initialization (over uniform)')
parser.add_argument('--max_grad_norm',
                    default=0,
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
                    default=100,
                    help='print stats after this many batches, also for wandb log-gradient frequency')
parser.add_argument('--wandb', action='store_true', help='use wandb')
parser.add_argument('--wandb_entity', default='anonymous', type=str, help='wandb entity')
parser.add_argument('--run_name', default='tg114514', type=str, help='wandb run name')
parser.add_argument('--wandb_key', default='', type=str, help='wandb key')


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
        # 随机初始化模型参数
        # 初始化范围为 [-args.param_init, args.param_init]
        # random initialization of model parameters
        # range: [-args.param_init, args.param_init]
        if args.param_init > 0:
            for param in model.parameters():
                param.data.uniform_(-args.param_init, args.param_init)
        # print('-' * 50)
        # print('Model Architecture:')
        # print(model)
        # print('-' * 50)
    else:
        print('loading model from ' + args.ckpt_path)
        checkpoint = torch.load(args.ckpt_path)
        model = checkpoint['model']

    # 1. 把模型参数分成2部分，分别是：model_params（给主模型）, q_params（给adversarial inference net）
    # action params 被 TransformerGrammarPlusQNet 代替了
    q_params = []
    # action_params = []
    model_params = []
    for name, param in model.named_parameters():
        if 'q_' in name:
            q_params.append(param)
        else:
            model_params.append(param)
    # optimizer = torch.optim.SGD(model_params, lr=args.lr)
    optimizer = torch.optim.Adam(model_params, lr=args.lr)
    q_optimizer = torch.optim.Adam(q_params, lr=args.q_lr)
    model.train()
    model.to(torch.device(device))
    epoch = 0
    is_lr_decay = False
    if args.kl_cost_annealing_warmup > 0:
        kl_pen = 0.
        kl_ann_every_batch = 1. / (args.kl_cost_annealing_warmup * len(train_data))
    else:
        kl_pen = 1.
    samples = args.samples
    best_val_ll = tg_eval_only_log_likeli(val_data,
                            model,
                            samples=args.eval_samples,
                            count_eos_ppl=args.count_eos_ppl)
    best_val_ppl = np.exp(-best_val_ll)
    print('-' * 50)
    print('Initial Validation PPL: %.2f, Initial Validation IWAE Log Likelihood: %.2f' % (best_val_ppl, best_val_ll))
    all_stats = [[0, 0, 0]]  # true pos, false pos, false neg for f1 calc
     
    # 2. 开始训练, 一共训练 args.num_epochs 轮
    
    step = 0
    for epoch in range(args.num_epochs):
        start_time = time.time()
        if epoch > args.train_q_epochs or step > args.train_q_steps:
            # stop training q after this many epochs
            args.q_lr = 0.
            for param_group in q_optimizer.param_groups:
                param_group['lr'] = args.q_lr
        train_q_entropy = 0.
        num_sents = 0.
        num_words = 0.
        total_sent_ll = 0.
        total_sent_obj = 0.
        total_sent_iwae_ll = 0.
        b = 0.
        tqdm_pbar = tqdm(total=len(train_data))
        for i in np.random.permutation(len(train_data)): # one step
            step += 1
            tqdm_pbar.update(1)
            if args.kl_cost_annealing_warmup > 0:
                kl_pen = min(args.kl_pen_max, kl_pen + kl_ann_every_batch)
            sents, length, batch_size, gold_actions, gold_spans, gold_binary_trees, other_data = train_data[i]
            if length == 1:
                # we ignore length 1 sents during training/eval since we work with binary trees only
                continue
            sents = sents.cuda(device=device)
            b += 1
            q_optimizer.zero_grad()
            optimizer.zero_grad()
            if args.mode == 'unsupervised':
                log_ll_p, ll_action_q, all_actions, q_entropy = model.forward(sents, samples=samples, has_eos=True)
                
                # q_entropy: shape: (batch_size * samples, )
                # q_entropy = q_entropy.contiguous().view(samples, batch_size) # (samples, batch_size)
                # q_entropy = q_entropy.mean(0) # (batch_size, )
                # ll_action_q: shape: (batch_size, samples)
                
                # ll_p: likelihood of p-net generation, evaluated by q-target
                # obj = likelihood_p.mean(1)
                # if epoch <= args.train_q_epochs:
                #     obj += kl_pen * q_entropy.mean()
                # train_q_entropy += q_entropy.sum().item()
                # log_f = likelihood_p + kl_pen*ll_action_p 
                # iwae_ll = log_f.mean(1).detach() + kl_pen*q_entropy.detach() # shape: (batch_size * samples, )
                log_ll_p = log_ll_p.contiguous().view(batch_size, samples) # (batch_size, samples)
                obj = log_ll_p # (batch_size, samples)
                obj = obj.mean(1)  # shape: (batch_size, )
                obj = obj.mean()
                iwae_ll = log_ll_p.mean(1).detach() + kl_pen*q_entropy.detach() # shape: (batch_size, )
                if epoch <= args.train_q_epochs and step <= args.train_q_steps:
                    obj += kl_pen*q_entropy.mean()
                # baseline = torch.zeros_like(log_f)
                # baseline_k = torch.zeros_like(log_f)
                baseline = torch.zeros_like(log_ll_p) # (batch_size, samples)
                baseline_out_of_k = torch.zeros_like(log_ll_p) # (batch_size, samples)
                for k in range(samples):
                    baseline_out_of_k.copy_(log_ll_p)
                    baseline_out_of_k[:, k].fill_(0)
                    baseline[:, k] = baseline_out_of_k.detach().sum(1) / (samples - 1)

                diff = (log_ll_p - baseline).detach() # (batch_size, samples)
                obj += (diff * ll_action_q).mean()
                # obj += ((likelihood_p.detach() - baseline.detach()) * ll_action_q).mean()
                # kl = (ll_action_q - ll_action_p).mean(1).detach()
                train_q_entropy += q_entropy.sum().item()
                log_ll_p = log_ll_p.mean(1) # shape: (batch_size, )
            else:
                raise NotImplementedError # NOTE: WE DON'T NEED SUPERVISED VERSION
            final_loss = -obj
            final_loss.backward()
            
            # if args.wandb:
            #     wandb.watch(model, log="gradients", log_freq=args.print_every)
            if args.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(model_params,
                                               args.max_grad_norm)
            if args.q_max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(q_params, args.q_max_grad_norm)
            
            q_optimizer.step()
            optimizer.step()
            num_sents += batch_size
            num_words += batch_size * length
            total_sent_ll += log_ll_p.sum().item() # shape: (batch_size, )
            total_sent_obj += obj.item()
            total_sent_iwae_ll += iwae_ll.sum().item()
            for bb in range(batch_size):
                action = list(all_actions[bb].long().cpu().numpy())
                span_b = get_spans(action)
                span_b_set = set(
                    span_b[:-1])  # ignore the sentence-level trivial span
                update_stats(span_b_set, [set(gold_spans[bb][:-1])], all_stats)
            if b % args.print_every == 0:
                log_str = 'Train Info: Epoch: %d, Batch: %d/%d, LR: %.4f, qLR: %.5f, Training Aver qEntropy: %.4f, ' + \
                          'Train Aver PPL for TG LOG LL: %.2f, Train Aver TG Neg LOG LL: %.2f, ' + \
                          'Train Aver IWAE PPL: %.2f, Train Aver IWAE Neg Log Likelihood: %.2f,' + \
                          'Best Validation Perplexity: %.2f, Best Val Log Likelihood: %.2f, KL Penalty: %.4f, ' + \
                          'Throughput: %.2f examples/sec'
                print(
                    log_str %
                    (epoch, b, len(train_data), args.lr, args.q_lr, train_q_entropy / num_sents,
                     np.exp(-total_sent_ll / num_words), -total_sent_ll / num_words,
                     np.exp(-total_sent_iwae_ll / num_words), -total_sent_iwae_ll / num_words,
                     best_val_ppl, best_val_ll, kl_pen, num_sents / (time.time() - start_time)))
                sent_str = [
                    train_data.idx2word[word_idx]
                    for word_idx in list(sents[-1][1:-1].cpu().numpy())
                ]
                print(f"PRED in {b}-th batch: ", get_tree(action[:-2], sent_str))
                print(f"GOLD in {b}-th batch: ", get_tree(gold_binary_trees[-1], sent_str))
            if args.wandb:
                wandb.log({'epoch': epoch})
                wandb.log({'lr': args.lr})
                wandb.log({'q_lr': args.q_lr})
                wandb.log({'Average train_q_entropy': train_q_entropy / num_sents})
                wandb.log({'Train Aver PPL for OBJ': torch.exp(final_loss)})
                wandb.log({'Train Aver OBJ (should be maximized)': obj})
                wandb.log({'Train Aver PPL for TG LOG LL': np.exp(-total_sent_ll / num_words)})
                wandb.log({'Train Aver TG Log Likelihood': -total_sent_ll / num_words})
                wandb.log({'Train Aver IWAE PPL': np.exp(-total_sent_iwae_ll / num_words)})
                wandb.log({'Train Aver IWAE Neg Log Likelihood': -total_sent_iwae_ll / num_words})
                wandb.log({'KL Penalty': kl_pen})
        print('--------------------------------')
        print('Checking validation performance...')
        val_ll = tg_eval_only_log_likeli(val_data,
                        model,
                        samples=args.eval_samples,
                        count_eos_ppl=args.count_eos_ppl)
        val_ppl = np.exp(-val_ll)
        print("Val PPL: ", val_ppl)
        print("Val IWAE Log Likelihood: ", val_ll)
        if args.wandb:
            wandb.log({'Validation Perplexity': val_ppl})
            wandb.log({'Validation IWAE Log Likelihood': val_ll})
            wandb.log({'Best Validation Perplexity': best_val_ppl})
            wandb.log({'Best IWAE Log Likelihood': best_val_ll})
        print('--------------------------------')
        if val_ll > best_val_ll:
            best_val_ppl = val_ppl
            best_val_ll = val_ll
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
                is_lr_decay = True
        if is_lr_decay == True:
            args.lr = args.lr_decay * args.lr
            args.q_lr = args.lr_decay * args.q_lr
            for param_group in optimizer.param_groups:
                param_group['lr'] = args.lr
            for param_group in q_optimizer.param_groups:
                param_group['lr'] = args.q_lr
            print('Learning rate decreased to %.4f' % args.lr)
    tqdm_pbar.close()
    print("Training Finished!")

def tg_eval_only_log_likeli(data, model: TransformerGrammarPlusQNet, samples=5, count_eos_ppl=0):
    # print('-'*50)
    # print("TG EVAL")
    # print("Data length: ", len(data))
    # sample : mc_sample. for iwae calculation
    model.eval()
    num_sents = 0
    num_words = 0
    mean_iwae_ll = 0.
    total_iwae_ll = 0.
    if args.kl_cost_annealing_warmup > 0:
        kl_pen = 0.
        kl_ann_every_batch = 1. / (args.kl_cost_annealing_warmup * len(data))
    else:
        kl_pen = 1.
    # print_data_bool = False
    with torch.no_grad():
        for i in list(reversed(range(len(data)))):
            sents, length, batch_size, gold_actions, gold_spans, gold_binary_trees, other_data = data[i]
            if length == 1:  # length 1 sents are ignored since URNNG needs at least length 2 sents
                continue
            if count_eos_ppl == 1:
                tree_length = length
                length += 1
            else:
                sents = sents[:, :-1]
                tree_length = length
            sents = sents.cuda()
            if args.kl_cost_annealing_warmup > 0:
                kl_pen = min(args.kl_pen_max, kl_pen + kl_ann_every_batch)
            log_likelihood, likeli_action_q_all, all_actions, q_entropy = model.forward(
                sents, samples=samples, has_eos=count_eos_ppl == 1)
            # log likelihood is i.e. ll, shape: (batch_size * samples, )
            # likeli_action_q_all, shape: (batch_size * samples, )
            log_likelihood = log_likelihood.contiguous().view(batch_size, samples)
            iwae_ll = log_likelihood.mean(1).detach() + kl_pen * q_entropy.detach()            
            num_sents += batch_size
            num_words += batch_size * length
            batch_sent_iwae_ll = iwae_ll.sum().item()
            total_iwae_ll += batch_sent_iwae_ll
    mean_iwae_ll = total_iwae_ll / num_words
    model.train()
    return mean_iwae_ll


if __name__ == '__main__':
    args = parser.parse_args()
    if args.wandb == False:
        import os
        os.environ['WANDB_MODE'] = 'dryrun'
        os.environ['WANDB_SILENT'] = 'true'
        os.environ['WANDB_DISABLED'] = 'true'
        os.environ['WANDB_WATCH'] = 'false'
        os.environ['WANDB_CONSOLE'] = 'off'
    else:
        wandb.login(key=args.wandb_key)
        wandb.init(project='transformer_grammar_unsupervised', entity=args.wandb_entity, config=args, name=args.run_name, force=True)
        wandb.config.update(args)
    tg_main(args)
