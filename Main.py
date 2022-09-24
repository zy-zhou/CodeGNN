import torch
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from Modules import HGNN, BasicDecoder
from Models import Model, Translator
from Utils import load, batch_bleu, batch_rouge, batch_meteor
from Train import TeacherForcing
from Data import pad
import argparse
import warnings
warnings.filterwarnings("ignore")

device = 'cuda' if torch.cuda.is_available() else 'cpu'
# device = 'cpu'
batch_size = 32

def collate(raw_batch, nl_field, training=True):
    pad_id = nl_field.vocab.stoi[pad]
    graphs, com_seqs, rev_graphs, rev_coms, sims = zip(*raw_batch)
    batch_nodes = [graph.x for graph in graphs]
    batch_nodes = pad_sequence(batch_nodes, batch_first=True, padding_value=pad_id).to(device)
    # batch_size * node_num * token_num
    for graph in graphs:
        graph.edge_attr = graph.edge_attr.to(torch.float)
    graphs = [graph.to(device) for graph in graphs]
    if training:
        com_seqs = nl_field.process(com_seqs, device=device)
    
    rev_batch_nodes = [graph.x.squeeze(1) for graph in rev_graphs]
    rev_node_num = [x.shape[0] for x in rev_batch_nodes]
    rev_node_num = torch.tensor(rev_node_num, device=device)
    rev_batch_nodes = pad_sequence(rev_batch_nodes, batch_first=True, padding_value=pad_id).to(device)
    rev_com_lens = list(map(len, rev_coms))
    rev_com_lens = torch.tensor(rev_com_lens, device=device)
    rev_coms = nl_field.process(rev_coms, device=device)
    sims = torch.tensor(sims, dtype=torch.float).to(device)
    return (graphs, batch_nodes, rev_batch_nodes, rev_node_num, rev_coms, rev_com_lens, sims), com_seqs

def load_data(lang='java', prefix='train', batch_size=batch_size):
    pre = 'cp.' if prefix[:2] == 'cp' else ''
    node_field = torch.load(f'data/{lang}/preprocessed/{pre}node_field.pkl')
    nl_field = torch.load(f'data/{lang}/preprocessed/{pre}nl_field.pkl')
    graphs = torch.load(f'data/{lang}/preprocessed/{prefix}.graph_dataset.pt')
    com_seqs = load(f'data/{lang}/preprocessed/{prefix}.nl.json', is_json=True)
    
    d_sim = torch.load(f'data/{lang}/preprocessed/{prefix}.rnn.topk.pt')
    # training set to retrieve from
    if 'train' not in prefix:
        if lang == 'java':
            pre = ''
        corpus = torch.load(f'data/{lang}/preprocessed/{pre}train.graph_dataset.pt')
        ref = load(f'data/{lang}/preprocessed/{pre}train.nl.json', is_json=True)
    else:
        corpus = graphs
        ref = com_seqs
    idx = d_sim['idx'][:,0]
    rev_graphs = [corpus[i] for i in idx]
    rev_coms = [ref[i] for i in idx]
    sims = d_sim['sims'][:,0]
    data = list(zip(graphs, com_seqs, rev_graphs, rev_coms, sims))
    data_gen = DataLoader(data, batch_size, shuffle='train' in prefix,
                          collate_fn=lambda x: collate(x, nl_field, 'train' in prefix))
    return data_gen, (node_field, nl_field)

def build_model(fields):
    encoder = HGNN(fields[0], fields[1])
    decoder = BasicDecoder(fields[1])
    model = Model(encoder, decoder)
    model = model.to(device)
    return model

def print_scores(pred, ref):
    hyp = [s.split() for s in pred]
    for n in range(1, 5, 1):
        bleu = batch_bleu(hyp, ref, smooth_method=0, n=n)
        print('BLEU-{} score: {:.2f}'.format(n, bleu * 100))
    bleu_s = batch_bleu(hyp, ref, smooth_method=3)
    print('Smoothed BLEU-4 score: {:.2f}'.format(bleu_s * 100))
    hyp = pred
    ref = [' '.join(s) for s in ref]
    rouge = batch_rouge(hyp, ref)
    print('ROUGE-L score: {:.2f}'.format(rouge['rouge-l']['f'] * 100))
    meteor = batch_meteor(hyp, ref)
    print('METEOR score: {:.2f}'.format(meteor * 100))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Main script for HGNN')
    parser.add_argument('-l', '--lang', default='java', choices=['java', 'python'])
    parser.add_argument('-c', '--cross_proj', action='store_true')
    parser.add_argument('-b', '--batch', type=int, default=batch_size)
    parser.add_argument('-t', '--test', action='store_true')
    args = parser.parse_args()
    
    pre = 'cp.' if args.cross_proj else ''
    post = '_cp' if args.cross_proj else ''
    if not args.test:
        train_gen, fields = load_data(args.lang, f'{pre}train', args.batch)
        val_gen, _ = load_data(args.lang, f'{pre}valid', args.batch)
        model = build_model(fields)
        trainer = TeacherForcing(model, epoches=70, metrics=['bleu'],
                                 smooth=3, save_per_epoch=False,
                                 save_path=f'checkpoints/HGNN_{args.lang}{post}.pt')
        reports = trainer(train_gen, val_gen)
    else:
        test_gen, fields = load_data(args.lang, f'{pre}test', args.batch)
        model = build_model(fields)
        print('Loading model from checkpoint...')
        checkpoints = torch.load(f'checkpoints/HGNN_{args.lang}{post}.pt', map_location=device)
        model.load_state_dict(checkpoints['model'])
        evaluator = Translator(model, metrics=[])
        predicts, reports = evaluator(test_gen, save_path=f'predicts/HGNN_{args.lang}{post}.txt')
        
        ref = [t[1] for t in test_gen.dataset]
        print_scores(predicts, ref)
    
