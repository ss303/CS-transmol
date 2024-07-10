import os
import sys
import torch
from torch.utils.data import DataLoader
import numpy as np

from modelzoo.CS_transmol.torchtext.torchtextDataset import TabularDataset
from modelzoo.CS_transmol.torchtext.torchtextField import Field
from modelzoo.CS_transmol.torchtext.torchtextIterator import Iterator, batch


# Tokenization function for SMILES strings
def tokenize_smiles(text):
    return list(text)

# Define constants
BOS_WORD = '<s>'
EOS_WORD = '</s>'
BLANK_WORD = "<blank>"
MAX_LEN = 100
MIN_FREQ = 1

# Define Fields for source (SRC) and target (TGT) sequences
SRC = Field(tokenize=tokenize_smiles, pad_token=BLANK_WORD)
TGT = Field(tokenize=tokenize_smiles, init_token=BOS_WORD, eos_token=EOS_WORD, pad_token=BLANK_WORD)
train_data = None
val_data = None
test_data = None
pad_idx = None

global max_src_in_batch, max_tgt_in_batch
#local_rank = int(os.environ["LOCAL_RANK"])
#global_rank = torch.distributed.dist.get_rank()
#device = global_rank % torch.cuda.device_count()
#torch.cuda.set_device(device)

# Function to load datasets
def load_datasets(train_file, val_file, test_file):
    global train_data, val_data, test_data, pad_idx
    if train_data == None:
        train_data, val_data, test_data = TabularDataset.splits(
            path='.', 
            train=train_file,
            validation=val_file,
            test=test_file,
            format='csv',
            fields=[('src', SRC), ('trg', TGT)],
            skip_header=False,
            filter_pred=lambda x: len(vars(x)['src']) <= MAX_LEN and len(vars(x)['trg']) <= MAX_LEN
        )
        # Build vocabularies
        SRC.build_vocab(train_data.src, min_freq=MIN_FREQ)
        TGT.build_vocab(train_data.trg, min_freq=MIN_FREQ)
    if pad_idx == None:
        pad_idx = TGT.vocab.stoi["<blank>"]
    
    return train_data, val_data, test_data

'''
class TabularDatasetWrapper(IterableDataset):
    def __init__(self, tabular_dataset, batch_size=1, repeat=False, batch_size_fn=None, sort_key=None, train=True, sort=None):
        """
        Args:
            tabular_dataset (torchtext.data.TabularDataset): The torchtext TabularDataset.
        """
        self.dataset = tabular_dataset
        self.examples = tabular_dataset.examples
        self.indices = tabular_dataset.indices
        self.fields = tabular_dataset.fields
        self.batch_size = batch_size
        self.repeat = repeat
        self.batch_size_fn = batch_size_fn
        self.sort_key = sort_key
        self.train = train
        self.random_shuffler = RandomShuffler()
        self.sort = not train if sort is None else sort
        self.iterations = 0
    
    def __getitem__(self, i):
        return self.examples[i]
    
    
    #def __getitems__(self, indices):
    #    return [self.__getitem__(idx) for idx in indices]
    
    

    def __len__(self):
        try:
            return len(self.examples)
        except TypeError:
            return 2**32
    
    
    
    def __iter__(self):
        self.create_batches()
        self._iterations_this_epoch = 0
        for idx, minibatch in enumerate(self.batches):
                # fast-forward if loaded from state
                if self._iterations_this_epoch > idx:
                    continue
                self.iterations += 1
                self._iterations_this_epoch += 1
                if self.sort:
                    # NOTE: `rnn.pack_padded_sequence` requires that a minibatch
                    # be sorted by decreasing order, which requires reversing
                    # relative to typical sort keys
                    minibatch.reverse()
                else:
                    minibatch.sort(key=self.sort_key, reverse=True)
                yield rebatch(pad_idx, Batch(minibatch, self.dataset, self.device))
                if not self.repeat:
                    return

    def __getattr__(self, attr):
        if attr in self.fields:
            for x in self.examples:
                yield getattr(x, attr)
    
    def data(self):
        """Return the examples in the dataset in order, sorted, or shuffled."""
        if self.sort:
            xs = sorted(self.dataset, key=self.sort_key)
        elif self.shuffle:
            xs = [self.dataset[i] for i in self.random_shuffler(range(len(self.dataset)))]
        else:
            xs = self.dataset
        return xs
    
    def create_batches(self):
        if self.train:
            def pool(d, random_shuffler):
                for p in data.batch(d, self.batch_size * 100):
                    p_batch = data.batch(
                        sorted(p, key=self.sort_key),
                        self.batch_size, self.batch_size_fn)
                    for b in random_shuffler(list(p_batch)):
                        yield b
            self.batches = pool(self.data(), self.random_shuffler)
        else:
            self.batches = []
            for b in data.batch(self.data(), self.batch_size, self.batch_size_fn):
                self.batches.append(sorted(b, key=self.sort_key))
    




#write a wrapper class, super class is dataloader. add create_batchs as the 
class CustomDataLoader(DataLoader):
    def __init__(self, dataset, batch_size=1, shuffle=False, batch_size_fn=None, sort_key=None, train=True, **kwargs):
        self.batch_size_fn = batch_size_fn
        self.sort_key = sort_key
        self.train = train
        self.random_shuffler = RandomShuffler()
        super().__init__(dataset, batch_size, shuffle, **kwargs)
        self._iterator = Iterator(dataset, batch_size=batch_size, #device=device,
                            repeat=False, sort_key=lambda x: (len(x.src), len(x.trg)),
                            batch_size_fn=batch_size_fn, train=train)
        
'''

# Batch size function for padding
def batch_size_fn(new, count, sofar):
    global max_src_in_batch, max_tgt_in_batch
    if count == 1:
        max_src_in_batch = 0
        max_tgt_in_batch = 0
    max_src_in_batch = max(max_src_in_batch, len(new.src))
    max_tgt_in_batch = max(max_tgt_in_batch, len(new.trg) + 2)
    src_elements = count * max_src_in_batch
    tgt_elements = count * max_tgt_in_batch
    return max(src_elements, tgt_elements)

# Function to rebatch the data to match the input format
def rebatch(pad_idx, batch):
    src, trg = batch.src.transpose(0, 1), batch.trg.transpose(0, 1)
    #src, trg = src.cuda(), trg.cuda()
    return ReBatch(src, trg, pad_idx)

class ReBatch:
    def __init__(self, src, trg=None, pad=0):
        self.src = src
        self.src_mask = (src != pad).unsqueeze(-2)
        if trg is not None:
            self.trg = trg[:, :-1]
            self.trg_y = trg[:, 1:]
            self.trg_mask = self.make_std_mask(self.trg, pad)
            self.ntokens = (self.trg_y != pad).data.sum()

    @staticmethod
    def make_std_mask(tgt, pad):
        tgt_mask = (tgt != pad).unsqueeze(-2)
        tgt_mask = tgt_mask & subsequent_mask(tgt.size(-1)).type_as(tgt_mask.data)
        return tgt_mask

def subsequent_mask(size):
    attn_shape = (1, size, size)
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    return torch.from_numpy(subsequent_mask) == 0


# Functions to create data loaders as required by Cerebras
def get_train_dataloader(params):
    train_data, _, _ = load_datasets(params["train_input"]["data_small/train_path"], params["train_input"]["data_small/val_path"], params["eval_input"]['data_small/eval_path'])
    #train_data_wrapper = TabularDatasetWrapper(train_data, batch_size=params["train_input"]["batch_size"], repeat=False, sort_key=lambda x: (len(x.src), len(x.trg)), batch_size_fn=batch_size_fn, train=True)
    #train_iter = CustomDataLoader(train_data_wrapper, batch_size=params["train_input"]["batch_size"], shuffle=params["train_input"]["shuffle"], sort_key=lambda x: (len(x.src), len(x.trg)), batch_size_fn=batch_size_fn, train=True)
    train_iter= Iterator(train_data, batch_size=params["train_input"]["batch_size"], repeat=False, sort_key=lambda x: (len(x.src), len(x.trg)),
                            batch_size_fn=batch_size_fn, train=True)
    num_workers = params.get("num_workers", 0)
    if not num_workers:
        prefetch_factor = None
        persistent_workers = False
    else:
        prefetch_factor = params.get("prefetch_factor", 10)
        persistent_workers = params.get("persistent_workers", True)
    train_dataloader = DataLoader(train_iter, batch_size = None, num_workers = num_workers, prefetch_factor = prefetch_factor, persistent_workers = persistent_workers)
    #train_iter = CustomDataLoader(train_data, batch_size = params["train_input"]["batch_size"], shuffle=params["train_input"]["shuffle"], batch_size_fn=batch_size_fn, train = True)
    params["train_input"]["SRC"] = SRC
    params["train_input"]["TGT"] = TGT
    params["train_input"]["pad_idx"] = pad_idx
    return train_dataloader

def get_eval_dataloader(params):
    _, val_data, _ = load_datasets(params["train_input"]["data_small/train_path"], params["train_input"]["data_small/val_path"], params["eval_input"]['data_small/eval_path'])
    #val_data_wrapper = TabularDatasetWrapper(val_data)
    #val_iter = CustomDataLoader(val_data_wrapper, batch_size=params["train_input"]["batch_size"], shuffle=params["train_input"]["shuffle"], sort_key=lambda x: (len(x.src), len(x.trg)), batch_size_fn=batch_size_fn, train=False)
    return #(rebatch(pad_idx, b) for b in val_iter)


#from cerebras.modelzoo.common.utils.run.cli_pytorch import get_params_from_args
params = {"train_input": {"train_path": "moses_train_small.csv", "val_path": "moses_val_small.csv", "exts": ["src", "trg"], "batch_size": 10, "shuffle": True}, "eval_input": {"eval_path": "moses_val_small.csv"}}
dataloader = get_train_dataloader(params)
print(dataloader)
#print((rebatch(pad_idx, b) for b in dataloader))
items = (rebatch(pad_idx, b) for b in dataloader)
i = 0
for b in items:
    #c = rebatch(pad_idx, b)
    print(b.src)
    print(i)
    i += 1