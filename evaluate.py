import argparse

parser = argparse.ArgumentParser(description='')

parser.add_argument('--device', type=int, default=0,
                        help='which gpu to use if any (default: 0)')
parser.add_argument('--batch_size', type=int, default=32,
                        help='input batch size for training (default: 32)')
parser.add_argument('--num_workers', type=int, default=0,
                        help='number of workers (default: 0)')

parser.add_argument('--dataset', type=str, default="ogbg-molhiv",
                        help='dataset name (default: ogbg-molhiv)')
parser.add_argument('--filename', type=str, default="",
                        help='')
args = parser.parse_args()


from ogb.graphproppred import PygGraphPropPredDataset, Evaluator
import torch
from torch_geometric.loader import DataLoader

# setting
device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")

# load a dataset
dataset = PygGraphPropPredDataset(name = args.dataset)
split_idx = dataset.get_idx_split()
test_loader = DataLoader(dataset[split_idx["test"]], batch_size=args.batch_size, shuffle=False, num_workers = args.num_workers)

# evaluation metrics
evaluator = Evaluator(args.dataset)


# load model and its weight
md_weight_param = torch.load(args.filename)
from model import model_selector
model = model_selector(**md_weight_param['param']).to(device)
model.load_state_dict(md_weight_param['weight'])

# $$$$$$ evaluation $$$$$ 
from train import evaluate
test_perf = evaluate(model, device, test_loader, evaluator)


# show and save the result as a file

import os
print(test_perf)
fname = os.path.splitext(args.filename)[0]
with open(f'{fname}.txt','w') as f:
    f.write(str(test_perf))
    print(f'the result is saved in {fname}.txt ')