'''
This train use for all camelyon16, tcga renal and tcga-lung 
'''

import argparse
import numpy as np
import torch
import torch.optim
import torch.utils.data
from tensorboardX import SummaryWriter
import datetime
import os
import sys
import yaml
import pandas as pd
from tqdm import tqdm
from copy import deepcopy
import random
import h5py

# Add TOP external path
sys.path.append('src/externals/TOP')
import util
import utliz
from clip import clip
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer
from models.learnable_prompt import MIL_CLIP, PromptLearner
from train_TCGAFeat_MIL_CLIP import (
    Map_few_shot, 
    Map_Negative_breaker, 
    get_pathological_tissue_level_prompts,
    Optimizer,
    str2bool,
    get_parser as get_original_parser
)

_tokenizer = _Tokenizer()


class TCGA_Renal_H5Feat(torch.utils.data.Dataset):
    def __init__(self, split, data_dir_s, label_dict, split_csv_folder=None, fold=None):
        self.samples = []
        self.slide_label_all = []  # Add this attribute for Map_few_shot compatibility
        used_random_split = False
        # If split CSVs are available, use them
        if split_csv_folder and fold:
            csv_path = os.path.join(split_csv_folder, f"fold_{fold}", f"{split}.csv")
            if os.path.exists(csv_path):
                df = pd.read_csv(csv_path)
                for _, row in df.iterrows():
                    slide = row['slide'] if 'slide' in row else row[0]
                    for label, folder in data_dir_s.items():
                        fpath = os.path.join(folder, slide + ".h5")
                        if os.path.exists(fpath):
                            self.samples.append((fpath, label_dict[label.upper()]))
                            self.slide_label_all.append(label_dict[label.upper()])  # Add slide label
                            break
            else:
                print(f"CSV not found: {csv_path}, using random split instead.")
                used_random_split = True
        else:
            used_random_split = True
        if used_random_split:
            # fallback: use all files, split 80/20
            for label, folder in data_dir_s.items():
                for fname in os.listdir(folder):
                    if fname.endswith('.h5'):
                        self.samples.append((os.path.join(folder, fname), label_dict[label.upper()]))
                        self.slide_label_all.append(label_dict[label.upper()])  # Add slide label
            np.random.seed(42)
            np.random.shuffle(self.samples)
            n = int(0.8 * len(self.samples))
            if split == 'train':
                self.samples = self.samples[:n]
                self.slide_label_all = self.slide_label_all[:n]  # Update slide_label_all accordingly
            else:
                self.samples = self.samples[n:]
                self.slide_label_all = self.slide_label_all[n:]  # Update slide_label_all accordingly
        
        # Convert to numpy array for compatibility
        self.slide_label_all = np.array(self.slide_label_all)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        with h5py.File(path, 'r') as f:
            feats = f['features'][:]
        patch_labels = np.zeros(feats.shape[0], dtype=np.long)
        return feats, [patch_labels, label], idx

    def __len__(self):
        return len(self.samples)


# Create a custom Optimizer class that handles the dimension issue
class CustomOptimizer(Optimizer):
    def train_one_epoch(self, epoch):
        self.model.train()
        loader = self.train_loader

        patch_label_gt = []
        patch_label_pred = []
        patch_label_pred_byMax = []
        bag_label_gt = []
        bag_label_pred = []
        bag_label_pred_byInstance = []
        for iter, (data, label, selected) in enumerate(tqdm(loader, desc='Epoch {} training'.format(epoch))):
            for i, j in enumerate(label):
                if torch.is_tensor(j):
                    label[i] = j.to(self.dev)
            selected = selected.squeeze(0)
            niter = epoch * len(loader) + iter

            data = data.to(self.dev)
            bag_prediction, instance_attn_score = self.model(data.squeeze(0))
            bag_prediction = torch.softmax(bag_prediction, 1)
            loss_D = torch.mean(-1. * (label[1] * torch.log(bag_prediction[:, 1]+1e-5) + (1. - label[1]) * torch.log(1. - bag_prediction[:, 1]+1e-5)))
            instance_attn_score_normed = torch.softmax(instance_attn_score, 0)
            
            # Handle the case where instance_attn_score_normed is 1D (single instance)
            if instance_attn_score_normed.dim() == 1:
                # If only one instance, set loss_A to 0
                loss_A = torch.tensor(0.0, device=instance_attn_score_normed.device)
            else:
                # Ensure we have at least 2 dimensions for matrix multiplication
                if instance_attn_score_normed.shape[0] < 2:
                    loss_A = torch.tensor(0.0, device=instance_attn_score_normed.device)
                else:
                    loss_A = torch.triu(instance_attn_score_normed.T @ instance_attn_score_normed, diagonal=1).mean()
            
            loss = loss_D + self.weight_lossA * loss_A
            if type(self.optimizer) is list:
                for optimizer_i in self.optimizer:
                    optimizer_i.zero_grad()
                loss.backward()
                for optimizer_i in self.optimizer:
                    optimizer_i.step()
            else:
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            patch_label_pred.append(instance_attn_score.mean(-1, keepdim=True).detach().squeeze())
            patch_label_pred_byMax.append(instance_attn_score.max(-1, keepdim=True)[0].detach().squeeze())
            patch_label_gt.append(label[0].squeeze(0))
            bag_label_pred.append(bag_prediction.mean(0, keepdim=True).detach()[0, 1])
            bag_label_pred_byInstance.append(instance_attn_score.mean(-1, keepdim=True).max().detach().squeeze())
            bag_label_gt.append(label[1])
            if niter % self.log_period == 0:
                self.writer.add_scalar('train_loss', loss.item(), niter)
                self.writer.add_scalar('train_loss_A', loss_A.item(), niter)
                self.writer.add_scalar('train_loss_D', loss_D.item(), niter)

        # Handle empty lists and ensure proper tensor dimensions
        if len(patch_label_pred) == 0:
            print("Warning: No training data processed")
            return 0
            
        # Ensure tensors have proper dimensions before concatenation
        patch_label_pred = [p.unsqueeze(0) if p.dim() == 0 else p for p in patch_label_pred]
        patch_label_pred_byMax = [p.unsqueeze(0) if p.dim() == 0 else p for p in patch_label_pred_byMax]
        patch_label_gt = [p.unsqueeze(0) if p.dim() == 0 else p for p in patch_label_gt]
        
        patch_label_pred = torch.cat(patch_label_pred)
        patch_label_pred_byMax = torch.cat(patch_label_pred_byMax)
        patch_label_gt = torch.cat(patch_label_gt)
        bag_label_pred = torch.tensor(bag_label_pred)
        bag_label_pred_byInstance = torch.stack(bag_label_pred_byInstance)
        bag_label_gt = torch.cat(bag_label_gt)

        self.estimated_AttnScore_norm_para_min = patch_label_pred.min()
        self.estimated_AttnScore_norm_para_max = patch_label_pred.max()
        patch_label_pred_normed = self.norm_AttnScore2Prob(patch_label_pred)

        self.estimated_AttnScore_norm_para_min_byMax = patch_label_pred_byMax.min()
        self.estimated_AttnScore_norm_para_max_byMax = patch_label_pred_byMax.max()
        patch_label_pred_byMax_normed = self.norm_AttnScore2Prob(patch_label_pred_byMax)

        bag_auc = utliz.cal_auc(bag_label_gt.reshape(-1), bag_label_pred.reshape(-1))
        bag_label_pred_byInstance_normed = self.norm_AttnScore2Prob(bag_label_pred_byInstance)
        bag_auc_byInstance = utliz.cal_auc(bag_label_gt.reshape(-1), bag_label_pred_byInstance_normed.reshape(-1))
        self.writer.add_scalar('train_bag_AUC', bag_auc, epoch)
        self.writer.add_scalar('train_bag_AUC_byInstance', bag_auc_byInstance, epoch)

        bag_pred_metrics, _, _ = utliz.cal_TPR_TNR_FPR_FNR(bag_label_gt.reshape(-1), bag_label_pred.reshape(-1))
        self.writer.add_scalar('train_bag_TPR', bag_pred_metrics[0], epoch)
        self.writer.add_scalar('train_bag_TNR', bag_pred_metrics[1], epoch)
        self.writer.add_scalar('train_bag_FPR', bag_pred_metrics[2], epoch)
        self.writer.add_scalar('train_bag_FNR', bag_pred_metrics[3], epoch)

        return 0

    def test(self, epoch):
        self.model.eval()
        loader = self.test_loader

        patch_label_gt = []
        patch_label_pred = []
        patch_label_pred_byMax = []
        bag_label_gt = []
        bag_label_pred = []
        bag_label_pred_byInstance = []
        for iter, (data, label, selected) in enumerate(tqdm(loader, desc='Epoch {} testing'.format(epoch))):
            for i, j in enumerate(label):
                if torch.is_tensor(j):
                    label[i] = j.to(self.dev)
            selected = selected.squeeze(0)
            niter = epoch * len(loader) + iter

            data = data.to(self.dev)
            with torch.no_grad():
                bag_prediction, instance_attn_score = self.model(data.squeeze(0))
                bag_prediction = torch.softmax(bag_prediction, 1)

            patch_label_pred.append(instance_attn_score.mean(-1, keepdim=True).detach().squeeze())
            patch_label_pred_byMax.append(instance_attn_score.max(-1, keepdim=True)[0].detach().squeeze())
            patch_label_gt.append(label[0].squeeze(0))
            bag_label_pred.append(bag_prediction.mean(0, keepdim=True).detach()[0, 1])
            bag_label_pred_byInstance.append(instance_attn_score.mean(-1, keepdim=True).max().detach().squeeze())
            bag_label_gt.append(label[1])

        # Handle empty lists and ensure proper tensor dimensions
        if len(patch_label_pred) == 0:
            print("Warning: No test data processed")
            return 0
            
        # Ensure tensors have proper dimensions before concatenation
        patch_label_pred = [p.unsqueeze(0) if p.dim() == 0 else p for p in patch_label_pred]
        patch_label_pred_byMax = [p.unsqueeze(0) if p.dim() == 0 else p for p in patch_label_pred_byMax]
        patch_label_gt = [p.unsqueeze(0) if p.dim() == 0 else p for p in patch_label_gt]
        
        patch_label_pred = torch.cat(patch_label_pred)
        patch_label_pred_byMax = torch.cat(patch_label_pred_byMax)
        patch_label_gt = torch.cat(patch_label_gt)
        bag_label_prediction = torch.tensor(bag_label_pred)
        bag_label_pred_byInstance = torch.stack(bag_label_pred_byInstance)
        bag_label_gt = torch.cat(bag_label_gt)

        patch_label_pred_normed = (patch_label_pred - patch_label_pred.min()) / (patch_label_pred.max() - patch_label_pred.min())
        patch_label_pred_byMax_normed = (patch_label_pred_byMax - patch_label_pred_byMax.min()) / (patch_label_pred_byMax.max() - patch_label_pred.min())

        bag_auc = utliz.cal_auc(bag_label_gt.reshape(-1), bag_label_prediction.reshape(-1))
        bag_label_pred_byInstance_normed = self.norm_AttnScore2Prob(bag_label_pred_byInstance)
        bag_auc_byInstance = utliz.cal_auc(bag_label_gt.reshape(-1), bag_label_pred_byInstance_normed.reshape(-1))
        self.writer.add_scalar('test_bag_AUC', bag_auc, epoch)
        self.writer.add_scalar('test_bag_AUC_byInstance', bag_auc_byInstance, epoch)

        bag_pred_metrics, _, _ = utliz.cal_TPR_TNR_FPR_FNR(bag_label_gt.reshape(-1), bag_label_prediction.reshape(-1))
        self.writer.add_scalar('test_bag_TPR', bag_pred_metrics[0], epoch)
        self.writer.add_scalar('test_bag_TNR', bag_pred_metrics[1], epoch)
        self.writer.add_scalar('test_bag_FPR', bag_pred_metrics[2], epoch)
        self.writer.add_scalar('test_bag_FNR', bag_pred_metrics[3], epoch)

        return 0


def seed_torch(seed=7):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def main(args):
    seed_torch(args.seed)
    for fold in range(args.k_start, args.k_end + 1):
        print(f"\n{'='*50}")
        print(f"Training fold {fold}")
        print(f"{'='*50}")
        try:
            train_ds = TCGA_Renal_H5Feat('train', args.paths['data_dir_s'], args.label_dict, args.paths.get('split_folder'), fold)
            val_ds = TCGA_Renal_H5Feat('val', args.paths['data_dir_s'], args.label_dict, args.paths.get('split_folder'), fold)
            if len(train_ds) < 2:
                print(f"Warning: Only {len(train_ds)} training slides available for fold {fold}. Skipping this fold.")
                continue
            train_ds = Map_few_shot(train_ds, num_shot=getattr(args, 'num_shot', -1))
            train_ds = Map_Negative_breaker(train_ds, break_p=getattr(args, 'NegBagBreakProb', 0.0), break_proportion=getattr(args, 'NegBagBreakProP', 1.0))
            train_loader = torch.utils.data.DataLoader(train_ds, batch_size=1, shuffle=True, num_workers=getattr(args, 'workers', 0), drop_last=False)
            val_loader = torch.utils.data.DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=getattr(args, 'workers', 0), drop_last=False)
        except Exception as e:
            print(f"Error setting up data loaders for fold {fold}: {e}")
            print("Skipping this fold and continuing...")
            continue
        class_names = list(args.label_dict.keys())
        bagPrompt_ctx_init = [
            f"Examine the renal tissue image, looking for papillary structures and clear cell patterns ({class_names[0]}). * * * * * * * * * *",
            f"Examine the renal tissue image, looking for clear cell carcinoma patterns ({class_names[1]}). * * * * * * * * * *",
            f"Examine the renal tissue image, looking for chromophobe cell patterns ({class_names[2]}). * * * * * * * * * *",
        ]
        bag_prompt_learner = PromptLearner(
            n_ctx=args.bagLevel_n_ctx,
            ctx_init=bagPrompt_ctx_init,
            all_ctx_trainable=args.all_ctx_trainable,
            csc=args.csc,
            classnames=class_names,
            clip_model='RN50', 
            p_drop_out=args.p_bag_drop_out
        )
        _, _, prompts_pathology_template_withDescription = get_pathological_tissue_level_prompts(multi_templates=False)
        instancePrompt_ctx_init = [i + '* * * * * * * * * *' for i in prompts_pathology_template_withDescription]
        instance_prompt_learner = PromptLearner(
            n_ctx=args.instanceLevel_n_ctx,
            ctx_init=instancePrompt_ctx_init,
            all_ctx_trainable=args.all_ctx_trainable,
            csc=args.csc,
            classnames=[f"Prototype {i}" for i in range(len(instancePrompt_ctx_init))],
            clip_model='RN50', 
            p_drop_out=args.p_drop_out
        )
        model = MIL_CLIP(bag_prompt_learner, instance_prompt_learner, clip_model="RN50", pooling_strategy=args.pooling_strategy).to('cuda:0')
        for param in model.text_encoder.parameters():
            param.requires_grad = False
        optimizer_text_branch = torch.optim.SGD(model.prompt_learner_bagLevel.parameters(), lr=args.lr_TB)
        optimizer_image_branch = torch.optim.SGD(list(model.prompt_learner_instanceLevel.parameters()) +
                                                 list(model.pooling.parameters()) +
                                                 list(model.coord_trans.parameters()) +
                                                 list(model.bag_pred_head.parameters()), lr=args.lr_IB)
        results_dir = args.paths['results_dir']
        os.makedirs(results_dir, exist_ok=True)
        name = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")+"_%s" % args.comment.replace('/', '_') + \
               f"_Fold{fold}_Seed{args.seed}_Bs{args.batch_size}_lrTB{args.lr_TB}_lrIB{args.lr_IB}_{args.num_shot}Shot_bagLevelNCTX{args.bagLevel_n_ctx}_instLevelNCTX{args.instanceLevel_n_ctx}_AllCTXtrainable{args.all_ctx_trainable}_CSC{args.csc}_poolingStrtegy{args.pooling_strategy}_NegBagProb{args.NegBagBreakProb}_NegBagProP{args.NegBagBreakProP}_pDropOut{args.p_drop_out}_pDropOutBag{args.p_bag_drop_out}_weightLossA{args.weight_lossA}"
        writer = SummaryWriter(f'{results_dir}/runs_TOP/{name}')
        writer.add_text('args', " \n".join(['%s %s' % (arg, getattr(args, arg)) for arg in vars(args)]))
        optimizer = CustomOptimizer(
            model=model, 
            train_loader=train_loader, 
            test_loader=val_loader,
            optimizer=[optimizer_text_branch, optimizer_image_branch],
            writer=writer, 
            num_epoch=args.epochs,
            dev=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
            weight_lossA=args.weight_lossA
        )
        optimizer.optimize()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--k_start', type=int, required=True)
    parser.add_argument('--k_end', type=int, required=True)
    args = parser.parse_args()
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    for k, v in config.items():
        setattr(args, k, v)
    main(args)


