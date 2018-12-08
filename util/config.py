
embedding_path = {
    'glove' : "/home/i/Code/python/CRNLP/datasets/embedding/glove.6B.100d.txt",
    # 'glove' : "E:/datasets/embedding/glove.6B.100d.txt",
    'google' : "/home/i/Code/python/CRNLP/datasets/embedding/GoogleNews-vectors-negative300.bin"
}

PAD = "<PAD>"
SOS = "<SOS>"
EOS = "<EOS>"
UNK = "<UNK>"
SVOCAB = {PAD:0, SOS:1, EOS:2, UNK:3}

vocab = {}
vocab['prep'] = ['in', 'at', 'on', 'of', 'behind', 'via', 'under', 'for', 'beyond', 'towards', 'since', 
    'beside', 'inside', 'outside', 'before', 'above', 'with', 'from', 'between', 'upon', 'into', 
    'after', 'as', 'by', 'about', 'without']
vocab['art'] = ['the', 'a', 'an']
vocab['pred'] = ['am', 'is', 'are', 'was', 'were', 'have', "'s", "'ve", 'had', 'has', "'m"]
vocab['punc'] = ['.', ',', ';', ':', '"', "'", '?', '!']

