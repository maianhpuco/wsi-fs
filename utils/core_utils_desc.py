import numpy as np
import torch
from utils.utils import *
import os
# from datasets.dataset_generic import save_splits
# from models.model_mil import MIL_fc, MIL_fc_mc
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_auc_score, roc_curve, f1_score
from sklearn.metrics import auc as calc_auc
from utils.loss_utils import FocalLoss


class Accuracy_Logger(object):
    """Accuracy logger"""
    def __init__(self, n_classes):
        super(Accuracy_Logger, self).__init__()
        self.n_classes = n_classes
        self.initialize()

    def initialize(self):
        self.data = [{"count": 0, "correct": 0} for i in range(self.n_classes)]
    
    def log(self, Y_hat, Y):
        Y_hat = int(Y_hat)
        Y = int(Y)
        self.data[Y]["count"] += 1
        self.data[Y]["correct"] += (Y_hat == Y)
    
    def log_batch(self, Y_hat, Y):
        Y_hat = np.array(Y_hat).astype(int)
        Y = np.array(Y).astype(int)
        for label_class in np.unique(Y):
            cls_mask = Y == label_class
            self.data[label_class]["count"] += cls_mask.sum()
            self.data[label_class]["correct"] += (Y_hat[cls_mask] == Y[cls_mask]).sum()
    
    def get_summary(self, c):
        count = self.data[c]["count"] 
        correct = self.data[c]["correct"]
        
        if count == 0: 
            acc = None
        else:
            acc = float(correct) / count
        
        return acc, correct, count

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=20, stop_epoch=50, verbose=False):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 20
            stop_epoch (int): Earliest epoch possible for stopping
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
        """
        self.patience = patience
        self.stop_epoch = stop_epoch
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.inf

    def __call__(self, epoch, val_loss, model, ckpt_name = 'checkpoint.pt'):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, ckpt_name)
        elif score <= self.best_score:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience and epoch > self.stop_epoch:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, ckpt_name)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, ckpt_name):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), ckpt_name)
        self.val_loss_min = val_loss

def train(model, datasets, cur, args):
    # Ensure correct types
    args.reg = float(getattr(args, 'reg', 1e-5))
    args.lr = float(getattr(args, 'lr', 1e-4))
    args.testing = getattr(args, 'testing', False)
    args.mode = getattr(args, 'mode', 'transformer')

    print(f"\n=========== Training Fold {cur} ===========")
    writer_dir = os.path.join(args.results_dir, f"fold_{cur}")
    os.makedirs(writer_dir, exist_ok=True)
    writer = SummaryWriter(writer_dir) if getattr(args, 'log_data', False) else None

    # Unpack datasets
    train_set, val_set, test_set = datasets

    # Define loss and optimizer
    loss_fn = nn.CrossEntropyLoss()
    optimizer = get_optim(model, args)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.1, patience=10)

    # Prepare data loaders
    train_loader = get_split_loader(train_set, training=True, testing=args.testing,
                                    weighted=args.weighted_sample, mode=args.mode)
    val_loader = get_split_loader(val_set, testing=args.testing, mode=args.mode)
    test_loader = get_split_loader(test_set, testing=args.testing, mode=args.mode)
    
    print(f"[INFO] #Train samples: {len(train_set)}, #Batches: {len(train_loader)}")
    print(f"[INFO] #Val samples: {len(val_set)}, #Batches: {len(val_loader)}")
    print(f"[INFO] #Test samples: {len(test_set)}, #Batches: {len(test_loader)}") 
    # Set up early stopping
    early_stopping = EarlyStopping(patience=20, stop_epoch=80) if args.early_stopping else None

    # Training loop
    for epoch in range(args.max_epochs):
        train_loop(args, epoch, model, train_loader, optimizer, args.n_classes, writer, loss_fn)
        stop = validate(cur, epoch, model, val_loader, args.n_classes, early_stopping, writer, loss_fn, args.results_dir)
        if stop:
            print(f"[x] Early stopping at epoch {epoch}")
            break

    # Save or load best model
    ckpt_path = os.path.join(args.results_dir, f"s_{cur}_checkpoint.pt")
    if args.early_stopping:
        model.load_state_dict(torch.load(ckpt_path))
    else:
        torch.save(model.state_dict(), ckpt_path)

    # Final evaluation
    _, val_error, val_auc, _, val_f1 = summary(args.mode, model, val_loader, args.n_classes, args.results_dir)
    results_dict, test_error, test_auc, acc_logger, test_f1 = summary(args.mode, model, test_loader, args.n_classes, args.results_dir)

    # Print results
    print(f"Val AUC:  {val_auc:.4f}, F1: {val_f1:.4f}")
    print(f"Test AUC: {test_auc:.4f}, F1: {test_f1:.4f}")
    for i in range(args.n_classes):
        acc, correct, total = acc_logger.get_summary(i)
        print(f"Class {i}: acc {acc:.4f}, correct {correct}/{total}")
        if writer:
            writer.add_scalar(f"final/test_class_{i}_acc", acc, 0)

    if writer:
        writer.add_scalar("final/val_auc", val_auc, 0)
        writer.add_scalar("final/test_auc", test_auc, 0)
        writer.close()

    return results_dict, test_auc, val_auc, 1 - test_error, 1 - val_error, [acc_logger.get_summary(i)[0] for i in range(args.n_classes)], test_f1  
 
def train_loop(args, epoch, model, loader, optimizer, n_classes, writer = None, loss_fn = None):

    device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.train()
    acc_logger = Accuracy_Logger(n_classes=n_classes)
    train_loss = 0.
    train_error = 0.

    print('\n')
    for batch_idx, batch in enumerate(loader):
        # if batch is None:
        #     print(f"[Warning] Skipping empty batch at index {batch_idx}")
        #     continue

        data_s, coord_s, data_l, coords_l, label = batch 
    # for batch_idx, (data_s, coord_s, data_l, coords_l, label) in enumerate(loader):
        # print("print in train loop")
        # print( data_s, coord_s, data_l, coords_l, label)
        data_s, coord_s, data_l, coords_l, label = data_s.to(device), coord_s.to(device), data_l.to(device), coords_l.to(device), label.to(device)
        _, Y_hat, loss, _ = model(data_s, coord_s, data_l, coords_l, label)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        acc_logger.log(Y_hat, label)
        loss_value = loss.item()
        train_loss += loss_value
        error = calculate_error(Y_hat, label)
        train_error += error

    train_loss /= len(loader)
    train_error /= len(loader)

    print('Epoch: {}, train_loss: {:.4f}, train_error: {:.4f}'.format(epoch, train_loss, train_error))
    for i in range(n_classes):
        acc, correct, count = acc_logger.get_summary(i)
        print('class {}: acc {:.4f}, correct {}/{}'.format(i, acc, correct, count))
        if writer:
            writer.add_scalar('train/class_{}_acc'.format(i), acc, epoch)

    if writer:
        writer.add_scalar('train/loss', train_loss, epoch)
        writer.add_scalar('train/error', train_error, epoch)

import json 

def validate(cur, epoch, model, loader, n_classes, early_stopping = None, writer = None, loss_fn = None, results_dir=None, print_results=True):
    device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.eval()
    acc_logger = Accuracy_Logger(n_classes=n_classes)
    val_loss = 0.
    val_error = 0.
    
    prob = np.zeros((len(loader), n_classes))
    labels = np.zeros(len(loader))

    all_pred = []
    all_label = []
    all_descriptions = [] 
    
    with torch.no_grad():
        for batch_idx, (data_s, coord_s, data_l, coords_l, label) in enumerate(loader):
            data_s, coord_s, data_l, coords_l, label = data_s.to(device, non_blocking=True), coord_s.to(device, non_blocking=True), \
                                                                  data_l.to(device, non_blocking=True), coords_l.to(device, non_blocking=True), \
                                                                  label.to(device, non_blocking=True)
            Y_prob, Y_hat, loss, descriptions = model(data_s, coord_s, data_l, coords_l, label)
            # if print_results:
            #     print(f"Y_hat: {Y_hat.tolist()} | Y_prob: {Y_prob.tolist()} | Descriptions: {descriptions}")

            acc_logger.log(Y_hat, label)
            prob[batch_idx] = Y_prob.cpu().numpy()
            labels[batch_idx] = label.item()
            val_loss += loss.item()
            error = calculate_error(Y_hat, label)
            val_error += error
            all_pred.append(Y_hat.cpu())
            all_label.append(label.cpu())
            all_descriptions.append(descriptions)
            
        # Save descriptions to a file
    
    # desc_dir = os.path.join(results_dir, "desc")
    # os.makedirs(desc_dir, exist_ok=True)
    
    # desc_file = os.path.join(desc_dir, f"descriptions.json")
    # with open(desc_file, 'w') as f:
    #     json.dump(all_descriptions, f, indent=2)
        
    val_error /= len(loader)
    val_loss /= len(loader)
    val_f1 = f1_score(all_label, all_pred, average='macro')

    if n_classes == 2:
        auc = roc_auc_score(labels, prob[:, 1])
    
    else:
        auc = roc_auc_score(labels, prob, multi_class='ovr')
    
    if writer:
        writer.add_scalar('val/loss', val_loss, epoch)
        writer.add_scalar('val/auc', auc, epoch)
        writer.add_scalar('val/error', val_error, epoch)

    print('\nVal Set, val_loss: {:.4f}, val_error: {:.4f}, auc: {:.4f}, f1: {: .4f}'.format(val_loss, val_error, auc, val_f1))
    for i in range(n_classes):
        acc, correct, count = acc_logger.get_summary(i)
        print('class {}: acc {:.4f}, correct {}/{}'.format(i, acc, correct, count))

    if early_stopping:
        assert results_dir
        early_stopping(epoch, val_error, model, ckpt_name = os.path.join(results_dir, "s_{}_checkpoint.pt".format(cur)))
        
        if early_stopping.early_stop:
            print("Early stopping")
            return True

    return False

def summary(mode, model, loader, n_classes, results_dir):
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    acc_logger = Accuracy_Logger(n_classes=n_classes)
    model.eval()
    test_loss = 0.
    test_error = 0.
    test_f1 = 0.
    all_pred = []
    all_label = []

    all_probs = np.zeros((len(loader), n_classes))
    all_labels = np.zeros(len(loader))

    slide_ids = loader.dataset.slide_data['slide_id']
    patient_results = {}

    if(mode == 'transformer'):
        for batch_idx, (data_s, coord_s, data_l, coords_l, label) in enumerate(loader):
            data_s, coord_s, data_l, coords_l, label = data_s.to(device), coord_s.to(device), data_l.to(device), coords_l.to(device), label.to(device)
            slide_id = slide_ids.iloc[batch_idx]
            with torch.no_grad():
                Y_prob, Y_hat, loss , desc = model(data_s, coord_s, data_l, coords_l, label)

            acc_logger.log(Y_hat, label)
            probs = Y_prob.cpu().numpy()
            all_probs[batch_idx] = probs
            all_labels[batch_idx] = label.item()

            # patient_results.update({slide_id: {'slide_id': np.array(slide_id), 'prob': probs, 'label': label.item()}})
            patient_results.update({
                slide_id: {
                    'slide_id': slide_id,
                    'prob': probs.tolist(),
                    'label': label.item(),
                    'pred': Y_hat.item(),
                    'desc': ", ".join(desc) if isinstance(desc, list) else str(desc)
                }
            })
        
            error = calculate_error(Y_hat, label)
            test_error += error

            all_pred.append(Y_hat.cpu())
            all_label.append(label.cpu())

        test_error /= len(loader)
        test_f1 = f1_score(all_label, all_pred, average='macro')
        import pandas as pd

        # Convert results to DataFrame
        df = pd.DataFrame.from_dict(patient_results, orient='index')
        df.to_csv(f"{results_dir}/slide_predictions_with_desc.csv", index=False)
        print(f"Saved slide predictions with descriptions to {results_dir}/slide_predictions_with_desc.csv")
        if n_classes == 2:
            auc = roc_auc_score(all_labels, all_probs[:, 1])
            aucs = []
        else:
            aucs = []
            binary_labels = label_binarize(all_labels, classes=[i for i in range(n_classes)])
            for class_idx in range(n_classes):
                if class_idx in all_labels:
                    fpr, tpr, _ = roc_curve(binary_labels[:, class_idx], all_probs[:, class_idx])
                    aucs.append(calc_auc(fpr, tpr))
                else:
                    aucs.append(float('nan'))

            auc = np.nanmean(np.array(aucs))

        return patient_results, test_error, auc, acc_logger, test_f1

