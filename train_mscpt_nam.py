import os
import pytorch_lightning as pl
from argparse import ArgumentParser
from pytorch_lightning import Trainer
from pytorch_lightning.loggers.wandb import WandbLogger
from pytorch_lightning.callbacks import TQDMProgressBar
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from src.externals.MSCPT.model import MInterface
from src.externals.MSCPT.data import DInterface
from src.externals.MSCPT.utils import load_model_path_by_args
import warnings
import yaml

warnings.filterwarnings("ignore", category=UserWarning)
os.environ["WANDB_MODE"] = "offline"
import wandb

wandb.init(mode="offline")
label_dicts = {
    "NSCLC_subtyping": {"LUAD": 0, "LUSC": 1},
    "BRCA_recurrence": {"Low": 0, "High": 1},
    "RCC_subtyping": {"CHRCC": 0, "CCRCC": 1, "PRCC": 2},
    "UBC-OCEAN": {"LGSC": 0, "HGSC": 1, "EC": 2, "CC": 3, "MC": 4},
    "PANDA": {"LowRisk": 0, "IntermediateRisk": 1, "HighRisk": 2},
}


class MeterlessProgressBar(TQDMProgressBar):

    def init_train_tqdm(self):
        bar = super().init_train_tqdm()
        bar.dynamic_ncols = False
        bar.ncols = 0
        return bar

    def init_validation_tqdm(self):
        bar = super().init_validation_tqdm()
        bar.dynamic_ncols = False
        bar.ncols = 0
        return bar


def main(args):

    pl.seed_everything(args.seed)

    # Change working directory if specified
    if hasattr(args, "working_dir") and args.working_dir:
        import os

        original_cwd = os.getcwd()
        working_dir_path = os.path.join(original_cwd, args.working_dir)
        print(f"Changing working directory from {original_cwd} to {working_dir_path}")
        os.chdir(working_dir_path)

    load_path = load_model_path_by_args(args)
    data_module = DInterface(**vars(args))

    if args.model_name == "linear":
        name = f"{args.model_name}_epochs_{args.max_epochs}_seed_{args.seed}_numshots_{args.num_shots}_npro_{args.n_vpro}_aggregation_{args.linear_aggregation}_scale_{args.coop_scale}"
    elif args.model_name in ["coop", "cocoop", "maple", "metaprompt"]:
        name = f"{args.model_name}_epochs_{args.max_epochs}_seed_{args.seed}_numshots_{args.num_shots}_npro_{args.n_vpro}_scale_{args.coop_scale}"
    elif "ablation" in args.model_name:
        name = f"{args.model_name}_epochs_{args.max_epochs}_seed_{args.seed}_numshots_{args.num_shots}_ISGPT_{str(args.ISGPT)}_CGP_{str(args.CGP)}"
        print("GPT:", args.ISGPT, "CGP:", args.CGP)
    elif "aggregation" in args.model_name:
        name = f"{args.model_name}_epochs_{args.max_epochs}_seed_{args.seed}_numshots_{args.num_shots}_aggregation_{args.aggregation}"
    else:
        name = f"{args.model_name}_seed_{args.seed}_numshots_{args.num_shots}"
    logger = WandbLogger(project=args.logger_name, name=name)
    if args.model_name == "linear":
        args.txt_result_path = os.path.join(
            "./result", args.base_model, args.dataset_name, args.linear_aggregation
        )
    else:
        args.txt_result_path = os.path.join(
            "./result", args.base_model, args.dataset_name
        )
    args.heatmap_path = os.path.join(
        "./heatmap", args.base_model, args.dataset_name, args.model_name
    )
    args.wandb_name = name
    if not os.path.exists(args.txt_result_path):
        os.makedirs(args.txt_result_path)
    if not os.path.exists(args.heatmap_path):
        os.makedirs(args.heatmap_path)
    args.callbacks = []
    bar = MeterlessProgressBar()
    args.callbacks.append(bar)
    early_stop_callback = EarlyStopping(
        monitor="val_best_score",
        min_delta=0.00,
        patience=args.patience,
        verbose=True,
        mode="max",
    )
    args.callbacks.append(early_stop_callback)

    args.logger = logger
    if load_path is None:
        model = MInterface(**vars(args))
    else:
        model = MInterface(**vars(args))
        args.ckpt_path = load_path
    trainer = Trainer.from_argparse_args(args)
    trainer.fit(model, data_module)


if __name__ == "__main__":
    parser = ArgumentParser()

    # Add config file argument
    parser.add_argument("--config", type=str, help="Path to YAML config file")

    # Basic Training Control
    parser.add_argument("--batch_size", default=1, type=int)
    parser.add_argument("--num_workers", default=4, type=int)
    parser.add_argument("--seed", default=11, type=int)
    parser.add_argument("--lr", default=1e-4, type=float)
    parser.add_argument("--use_gpu", default=True, type=bool, help="use cpu or gpu")
    parser.add_argument("--device_ids", default=[0])
    parser.add_argument("--num_shots", default=16, type=int, help="num of few-shot")
    parser.add_argument("--total_epochs", default=30, type=int, help="num of epochs")

    # LR Scheduler
    parser.add_argument("--lr_scheduler", choices=["step", "cosine"], type=str)
    parser.add_argument("--lr_decay_steps", default=20, type=int)
    parser.add_argument("--lr_decay_rate", default=0.5, type=float)
    parser.add_argument("--lr_decay_min_lr", default=1e-5, type=float)

    # Restart Control
    parser.add_argument("--load_best", action="store_true")
    parser.add_argument("--load_dir", default=None, type=str)
    parser.add_argument("--load_ver", default=None, type=str)
    parser.add_argument("--load_v_num", default=None, type=int)

    # Training Info
    parser.add_argument("--dataset", default="my_data", type=str)  # my_data
    parser.add_argument(
        "--dataset_name",
        default="RCC",
        type=str,
        choices=["RCC", "Lung", "BRCA", "UBC-OCEAN", "PANDA"],
    )
    parser.add_argument("--data_dir", default="path/to/root/dir", type=str)
    parser.add_argument("--feat_data_dir", default="path/to/pt/files", type=str)
    parser.add_argument("--gpt_dir", default="./train_data/gpt")
    parser.add_argument("--csv_dir", default="./tcga_rcc.csv")
    parser.add_argument("--model_name", default="path_gnn", type=str)
    parser.add_argument("--loss", default="ce", type=str)
    parser.add_argument("--weight_decay", default=1e-5, type=float)
    parser.add_argument("--no_augment", action="store_true")
    parser.add_argument("--log_dir", default="lightning_logs", type=str)
    parser.add_argument("--task", default="RCC_subtyping", type=str)
    parser.add_argument("--target_size", default=224, type=int)
    parser.add_argument("--gc", default=1, type=int, help="Gradient Accumulation")
    parser.add_argument("--val", action="store_true")

    # Model Hyperparameters
    parser.add_argument("--hid", default=64, type=int)
    parser.add_argument("--block_num", default=8, type=int)
    parser.add_argument("--in_channel", default=3, type=int)
    parser.add_argument("--layer_num", default=5, type=int)

    # Other
    parser.add_argument("--aug_prob", default=0.5, type=float)

    # Model parameters
    parser.add_argument(
        "--base_model", default="plip", type=str, help="Base Model Name"
    )
    parser.add_argument(
        "--trainer_perc", default=32, type=int, help="双精，单精度， 混合精度"
    )
    parser.add_argument("--n_set", default=5, type=int, help="length of des")
    parser.add_argument("--n_tpro", default=5, type=int, help="length of text prompt")
    parser.add_argument("--n_vpro", default=5, type=int, help="length of vision prompt")
    parser.add_argument("--n_high", default=10, type=int)
    parser.add_argument("--n_topk", default=5, type=int, help="types of imgs")
    parser.add_argument(
        "--linear_aggregation", default="mean", type=str, help="mean, max, attention"
    )
    parser.add_argument("--coop_scale", default=-1, type=int, help="scale of coop")

    # Ablation Study
    parser.add_argument("--ISGPT", default=False, action="store_true")
    parser.add_argument("--CGP", default=False, action="store_true")
    parser.add_argument("--aggregation", default="mean", type=str, help="mean, max")
    parser.add_argument("--num_k", default=None, type=int)

    # Working Directory
    parser.add_argument("--working_dir", type=str, help="Working directory")

    args = parser.parse_args()

    # Load config from YAML file if provided
    if args.config:
        with open(args.config, "r") as f:
            config = yaml.safe_load(f)

        # Update args with config values
        for key, value in config.items():
            if hasattr(args, key):
                setattr(args, key, value)
            else:
                print(f"Warning: Unknown config key '{key}' will be ignored")

        # Ensure numeric parameters are properly typed
        if hasattr(args, "lr"):
            args.lr = float(args.lr)
        if hasattr(args, "weight_decay"):
            args.weight_decay = float(args.weight_decay)
        if hasattr(args, "batch_size"):
            args.batch_size = int(args.batch_size)
        if hasattr(args, "num_workers"):
            args.num_workers = int(args.num_workers)
        if hasattr(args, "seed"):
            args.seed = int(args.seed)
        if hasattr(args, "total_epochs"):
            args.total_epochs = int(args.total_epochs)
        if hasattr(args, "num_shots"):
            args.num_shots = int(args.num_shots)
        if hasattr(args, "n_set"):
            args.n_set = int(args.n_set)
        if hasattr(args, "n_tpro"):
            args.n_tpro = int(args.n_tpro)
        if hasattr(args, "n_vpro"):
            args.n_vpro = int(args.n_vpro)
        if hasattr(args, "n_high"):
            args.n_high = int(args.n_high)
        if hasattr(args, "n_topk"):
            args.n_topk = int(args.n_topk)
        if hasattr(args, "gc"):
            args.gc = int(args.gc)
        if hasattr(args, "patience"):
            args.patience = int(args.patience)
        if hasattr(args, "aug_prob"):
            args.aug_prob = float(args.aug_prob)
        if hasattr(args, "target_size"):
            args.target_size = int(args.target_size)
        if hasattr(args, "trainer_perc"):
            args.trainer_perc = int(args.trainer_perc)
        if hasattr(args, "coop_scale"):
            args.coop_scale = int(args.coop_scale)

    # Add pytorch lightning's args to parser as a group.
    parser = Trainer.add_argparse_args(parser)
    if args.dataset_name == "RCC":
        args.csv_dir = "./tcga_rcc.csv"
        args.task = "RCC_subtyping"
        args.logger_name = "mscpt_rcc"
    elif args.dataset_name == "Lung":
        args.csv_dir = "./tcga_lung.csv"
        args.task = "NSCLC_subtyping"
        args.logger_name = "mscpt_lung"
    elif args.dataset_name == "BRCA":
        args.csv_dir = "./tcga_brca.csv"
        args.task = "BRCA_recurrence"
        args.logger_name = "mscpt_brca"
    elif args.dataset_name == "UBC-OCEAN":
        args.csv_dir = "./ubc_ocean.csv"
        args.task = "UBC-OCEAN"
        args.logger_name = "mscpt_ubc_ocean"
    elif args.dataset_name == "PANDA":
        args.csv_dir = "./panda.csv"
        args.task = "PANDA"
        args.logger_name = "mscpt_panda"
    if args.base_model == "clip":
        args.patience = 20
    elif args.base_model == "plip" or args.base_model == "conch":
        args.patience = 10
    args.max_epochs = args.total_epochs
    if args.use_gpu:
        args.accelerator = "gpu"
        args.devices = args.device_ids
    else:
        args.accelerator = "cpu"
    if args.trainer_perc != 32:
        args.precision = args.trainer_perc
    #  去除trainer.fit时的验证步骤
    # args.num_sanity_val_steps = 0
    # List Arguments
    args.mean_sen = [0.485, 0.456, 0.406]
    args.std_sen = [0.229, 0.224, 0.225]
    args.num_sanity_val_steps = 0
    args.label_dicts = label_dicts[args.task]

    main(args)

# python train_mscpt_nam.py --config configs_local_nam/mscpt_conch_rcc.yaml
