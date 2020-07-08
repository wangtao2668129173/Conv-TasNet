import torch
import argparse
import sys
from trainer import main_worker
from utils import parse
from utils import get_logger
import torch.multiprocessing as mp
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--world_size', type=int, default=1)
    parser.add_argument('--rank', type=int, default=0)
    parser.add_argument('--seed',type=int,default=0)
    args = parser.parse_args()
    opt = parse('./config/train.yml', is_tain=True)
    ngpus_per_node = torch.cuda.device_count()
    args.world_size = ngpus_per_node * args.world_size
    mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node,opt,args))
if __name__ == "__main__":
    main()



