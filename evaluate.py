import argparse

parser.add_argument('--dataset', type=str, default="ogbg-molhiv",
                        help='dataset name (default: ogbg-molhiv)')
parser.add_argument('--filename', type=str, default="",
                        help='')
args = parser.parse_args()


device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")

from ogb.graphproppred import PygGraphPropPredDataset, Evaluator
import torch
from torch_geometric.loader import DataLoader

dataset = PygGraphPropPredDataset(name = args.dataset)
split_idx = dataset.get_idx_split()

test_loader = DataLoader(dataset[split_idx["test"]], batch_size=args.batch_size, shuffle=False, num_workers = args.num_workers)

md_weight_param = torch.load(args.filename)

# load model and its weight
from model import model_selector
model = model_selector(**md_weight_param['param']).to(device)
model.load_state_dict(md_weight_param['weight'])

# test_perf = eval(model, device, test_loader, evaluator)

