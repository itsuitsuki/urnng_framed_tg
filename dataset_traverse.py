from data import Dataset
train_file_dir = 'data/ptb_20240226-train.pkl'
train_data = Dataset(train_file_dir)
idx2word = train_data.idx2word
word2idx = train_data.word2idx
vocab_size = int(train_data.vocab_size)
for i in range(min(len(train_data), 1)):
    sents, length, batch_size, gold_actions, gold_spans, gold_binary_trees, other_data = train_data[i]
    print("Sents: ", sents)
    print('-'*50)
    print("Batch Size of ", i, ": ", batch_size) # size of one batch
    print('-'*50)
    print("Gold Actions: ", gold_actions)
    print('-'*50)
    print("Gold Spans: ", gold_spans)
    print('-'*50)
    print("Gold Binary Trees: ", gold_binary_trees)
    print(other_data)
    print('-'*50)
    print('-'*50)
    