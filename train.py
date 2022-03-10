import torch
from torch_geometric.loader import DataLoader
import torch.optim as optim
import torch.nn.functional as F
#from gnn import GNN

from tqdm import tqdm
import argparse
import time
import numpy as np

### importing OGB
from ogb.graphproppred import PygGraphPropPredDataset, Evaluator

cls_criterion = torch.nn.BCEWithLogitsLoss()
reg_criterion = torch.nn.MSELoss()

def train(model, device, loader, optimizer, task_type):
    model.train()

    for step, batch in enumerate(tqdm(loader, desc="Iteration")):
        batch = batch.to(device)

        if batch.x.shape[0] == 1 or batch.batch[-1] == 0:
            pass
        else:
            pred = model(batch)
            optimizer.zero_grad()
            ## ignore nan targets (unlabeled) when computing training loss.
            is_labeled = batch.y == batch.y
            if "classification" in task_type: 
                loss = cls_criterion(pred.to(torch.float32)[is_labeled], batch.y.to(torch.float32)[is_labeled])
            else:
                loss = reg_criterion(pred.to(torch.float32)[is_labeled], batch.y.to(torch.float32)[is_labeled])
            loss.backward()
            optimizer.step()

def evaluate(model, device, loader, evaluator):
    model.eval()
    y_true = []
    y_pred = []

    for step, batch in enumerate(tqdm(loader, desc="Iteration")):
        batch = batch.to(device)

        if batch.x.shape[0] == 1:
            pass
        else:
            with torch.no_grad():
                pred = model(batch)

            y_true.append(batch.y.view(pred.shape).detach().cpu())
            y_pred.append(pred.detach().cpu())

    y_true = torch.cat(y_true, dim = 0).numpy()
    y_pred = torch.cat(y_pred, dim = 0).numpy()

    input_dict = {"y_true": y_true, "y_pred": y_pred}

    return evaluator.eval(input_dict)


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='GNN baselines on ogbgmol* data with Pytorch Geometrics')
    parser.add_argument('--device', type=int, default=0,
                        help='which gpu to use if any (default: 0)')
    parser.add_argument('--gnn', type=str, default='gin-virtual',
                        help='GNN gin, gin-virtual, or gcn, or gcn-virtual (default: gin-virtual)')
    parser.add_argument('--drop_ratio', type=float, default=0.5,
                        help='dropout ratio (default: 0.5)')
    parser.add_argument('--num_layer', type=int, default=5,
                        help='number of GNN message passing layers (default: 5)')
    parser.add_argument('--emb_dim', type=int, default=300,
                        help='dimensionality of hidden units in GNNs (default: 300)')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='input batch size for training (default: 32)')
    
    parser.add_argument('--patience', type=str, default=7,
                        help='number of patience')
    parser.add_argument('--epochs', type=int, default=100,
                        help='number of epochs to train (default: 100)')
    parser.add_argument('--num_workers', type=int, default=0,
                        help='number of workers (default: 0)')
    parser.add_argument('--dataset', type=str, default="ogbg-molhiv",
                        help='dataset name (default: ogbg-molhiv)')

    parser.add_argument('--feature', type=str, default="full",
                        help='full feature or simple feature')
    parser.add_argument('--filename', type=str, default="",
                        help='filename to output result (default: )')
    
    parser.add_argument('--test', type=str, default=None,
                        help='simply testing to check the feedfoward/backpro.')
    args = parser.parse_args()

    device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")
    
    ### automatic dataloading and splitting
    dataset = PygGraphPropPredDataset(name = args.dataset)
    
    if args.feature == 'full':
        pass 
    elif args.feature == 'simple':
        print('using simple feature')
        # only retain the top two node/edge features
        dataset.data.x = dataset.data.x[:,:2]
        dataset.data.edge_attr = dataset.data.edge_attr[:,:2]
    if args.gnn == 'gcn-virtual':
        args.batch_size = 12
    
    split_idx = dataset.get_idx_split()

    ### automatic evaluator. takes dataset name as input
    evaluator = Evaluator(args.dataset)
    
    if args.test is not None:
        train_loader = DataLoader(dataset[split_idx["train"]][:64], batch_size=args.batch_size, shuffle=True, num_workers = args.num_workers)
        valid_loader = DataLoader(dataset[split_idx["valid"]][:64], batch_size=args.batch_size, shuffle=False, num_workers = args.num_workers)
    else:
        train_loader = DataLoader(dataset[split_idx["train"]], batch_size=args.batch_size, shuffle=True, num_workers = args.num_workers)
        valid_loader = DataLoader(dataset[split_idx["valid"]], batch_size=args.batch_size, shuffle=False, num_workers = args.num_workers)
    # test_loader = DataLoader(dataset[split_idx["test"]], batch_size=args.batch_size, shuffle=False, num_workers = args.num_workers)

    from model import model_selector
    md_params = {'gnn':args.gnn,'num_tasks': dataset.num_tasks, 'num_layer': args.num_layer, 'emb_dim': args.emb_dim, 'drop_ratio': args.drop_ratio}
    model = model_selector(**md_params).to(device)

    optimizer = optim.Adam(model.parameters(), lr=0.001)

    valid_curve = []
    test_curve = []
    train_curve = []
    
    i = 0
    for epoch in range(1, args.epochs + 1):
        print("=====Epoch {}".format(epoch))
        print('Training...')
        train(model, device, train_loader, optimizer, dataset.task_type)

        print('Evaluating...')
        train_perf = evaluate(model, device, train_loader, evaluator)
        valid_perf = evaluate(model, device, valid_loader, evaluator)
        # test_perf = evaluate(model, device, test_loader, evaluator)

        print({'Train': train_perf, 'Validation': valid_perf}) # , 'Test': test_perf})

        import copy
        if len(valid_curve)==0:
            best_model = model
            # best_model = copy.deepcopy(model)
        else:
            if 'classification' in dataset.task_type:
                if max(valid_curve) < valid_perf[dataset.eval_metric]:
                    best_model = model
                    ## best_model = copy.deepcopy(model)
                    i = 0
                else:
                    i += 1
                    if i == args.patience:
                        break
            else:
                if min(valid_curve) > valid_perf[dataset.eval_metric]:
                    best_model = copy.deepcopy(model)


        train_curve.append(train_perf[dataset.eval_metric])
        valid_curve.append(valid_perf[dataset.eval_metric])
        # test_curve.append(test_perf[dataset.eval_metric])

    if 'classification' in dataset.task_type:
        best_val_epoch = np.argmax(np.array(valid_curve))
        best_train = max(train_curve)
        # save the best model
    else:
        best_val_epoch = np.argmin(np.array(valid_curve))
        best_train = min(train_curve)
    best_val = valid_curve[best_val_epoch]


    print('Finished training!')
    print('Best validation score: {}'.format(best_val))
    # print('Test score: {}'.format(test_curve[best_val_epoch]))

    
    # save model
    
    best_model = best_model.to('cpu')
    import os
    if not args.filename == '':
        file_path = args.filename
    else:
        model_dir = './models'
        if not os.path.exists(model_dir):
            os.mkdir(model_dir)
        file_name = args.gnn + '_' + str(best_val_epoch) + '.pt'
        file_path = os.path.join(model_dir, file_name)

    torch.save({'weight':best_model.state_dict(),'param':md_params},file_path)
    
        # torch.save({'Val': valid_curve[best_val_epoch], 
                    # 'Test': test_curve[best_val_epoch], 
                    # 'Train': train_curve[best_val_epoch], 'BestTrain': best_train,'model':best_model}, args.filename)
if __name__ == "__main__":
    main()
