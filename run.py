import sys
import os
import inspect

sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))
# isort: on
from modelzoo.common.pytorch.utils import get_params_from_args

def main():
    parent = inspect.getouterframes(inspect.currentframe())[1]
    run_dir = os.path.dirname(os.path.abspath(parent.filename))
    print(run_dir)
    params = get_params_from_args(run_dir)
    #from cerebras.modelzoo.fc_mnist.pytorch.utils import set_defaults

    #set_defaults(params)

    #from modelzoo.common.run_utils import main
    from modelzoo.CS_transmol.data import (
        get_train_dataloader,
    )
    dataloader = get_train_dataloader(params)
    print(dataloader)
    #print((rebatch(pad_idx, b) for b in dataloader))
    #items = (rebatch(pad_idx, b) for b in dataloader)
    i = 0
    for b in dataloader:
        #c = rebatch(pad_idx, b)
        print(b.src)
        print(i)
        i += 1


if __name__ == '__main__':
    main()










'''
# Load parameters
with open('configs/params.yaml', 'r') as file:
    params = yaml.safe_load(file)

# Set random seeds
torch.manual_seed(params['runconfig']['seed'])

# Load data
SRC = Field(tokenize=str.split, init_token='<sos>', eos_token='<eos>', lower=True)
TGT = Field(tokenize=str.split, init_token='<sos>', eos_token='<eos>', lower=True)

train_data, val_data = load_data(params['train_input']['train_path'],
                                 params['train_input']['val_path'],
                                 params['train_input']['exts'],
                                 (SRC, TGT))

train_iter, val_iter = create_iterators(train_data, val_data, 64, torch.device('cuda'))

# Initialize model
model = make_model(len(SRC.vocab), len(TGT.vocab), N=6)
model = model.cuda()

# Initialize criterion and optimizer
criterion = LabelSmoothing(size=len(TGT.vocab), padding_idx=0, smoothing=0.1)
criterion = criterion.cuda()
optimizer = NoamOpt(model.src_embed[0].d_model, 1, 2000,
                    torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))

# Training loop
for epoch in range(params['runconfig']['max_steps']):
    model.train()
    run_epoch((rebatch(0, b) for b in train_iter), model,
              MultiGPULossCompute(model.generator, criterion, devices=[0], opt=optimizer))
    model.eval()
    loss = run_epoch((rebatch(0, b) for b in val_iter), model,
                     MultiGPULossCompute(model.generator, criterion, devices=[0], opt=None))
    print(loss)
    if epoch % params['runconfig']['checkpoint_steps'] == 0:
        torch.save(model.state_dict(), f'./model/lc_epoch_{epoch}.pt')
'''