import torch
import torch.nn as nn
import time
import os
import sys
import random
from utils.logger import get_logger
from model import check_parameters
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.nn.parallel import data_parallel
import torch.utils.data.distributed
from torch.nn.utils import clip_grad_norm_
from loss import si_snr_loss
import tqdm
import torch.distributed as dist
import torch.backends.cudnn as cudnn
import warnings
from data import DataLoaders,wav_dataset
from model import ConvTasNet,Dual_RNN_model

def main_worker(gpu, ngpus_per_node,opt,args):
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')
    # --------------------------------- distributed init -------------------------------------- #
    args.gpu = gpu
    args.rank = args.rank*ngpus_per_node + args.gpu   # local 
    args.batch_size = int(opt["datasets"]["batch_size"]/ngpus_per_node)
    args.num_workers = int((opt["datasets"]["num_workers"]+ngpus_per_node-1)/ngpus_per_node)
    dist.init_process_group(backend='nccl',init_method='tcp://127.0.0.1:23456',world_size=args.world_size,rank=args.rank)
    torch.cuda.set_device(args.gpu)   
    torch.backends.cudnn.benchmark = True
    # -----------------------------------  data init -------------------------------------------- #
    train_dataset = wav_dataset(**opt['datasets']['train'])
    dataset_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)  
    train_loader = DataLoaders(train_dataset, is_train=True, chunk_size=opt['datasets']['chunk_size'], 
                        batch_size=args.batch_size, num_workers=args.num_workers ,sampler=dataset_sampler)

    val_dataset = wav_dataset(**opt['datasets']['val'])
    val_loader = DataLoaders(val_dataset, is_train=False, chunk_size=opt['datasets']['chunk_size'], 
                             batch_size=args.batch_size, num_workers=args.num_workers)
    #---------------------------------- model init --------------------------------------------------#
    current_epoch = 0
    logger = get_logger(__name__)
    logger.info('Building the model')
    model = ConvTasNet(**opt['conv-Tasnet']['net_conf'])
    #model = Dual_RNN_model(**opt['dual-path-RNN']['net_conf'])


    #---------------------------------- resume -----------------------------------------------------#

    if opt['resume']['resume_state']:     # True or False
        cpt = torch.load(os.path.join(opt["resume"]["path"],"best.pt"), map_location="cpu")
        current_epoch = cpt["epoch"]
        # load nnet
        model.load_state_dict(cpt["model_state_dict"])
        model = model.cuda(args.gpu)
        model = nn.parallel.DistributedDataParallel(model,device_ids=[args.gpu],output_device=args.gpu,find_unused_parameters=True)
        optimizer = torch.optim.Adam(model.parameters(),lr=1e-3,betas=(0.9,0.98),eps=1e-08,weight_decay= 1e-5)
        optimizer.load_state_dict(cpt["optim_state_dict"])
    else:
        model = model.cuda(args.gpu)
        model = nn.parallel.DistributedDataParallel(model,device_ids=[args.gpu],output_device=args.gpu,find_unused_parameters=True)
        optimizer = torch.optim.Adam(model.parameters(),lr=1e-3,betas=(0.9,0.98),eps=1e-08,weight_decay= 1e-5)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,mode='min',factor=0.5,patience=2,
                                                            verbose=True,threshold=0.001, threshold_mode='abs',
                                                            cooldown=0,min_lr=1e-6,eps=1e-8)
    best_loss = 10000
    no_impr = 0
    scheduler.best = best_loss
    while current_epoch < opt['epochs']:
        logger.info("Starting epoch from {:d}, loss = {:.4f}".format(current_epoch,best_loss))
        current_epoch += 1
        trainer = Trainer(args=args,net=model,optimizer=optimizer,scheduler=scheduler,checkpoint=opt["model_path"],clip_norm=opt["clip_norm"],
                    logging_period=opt["print_freq"],start_epoch=current_epoch)
        train_loss = trainer.train(train_loader)
        val_loss = trainer.val(val_loader)
        if val_loss > best_loss:
            no_impr += 1
            logger.info('no improvement, best loss: {:.4f}'.format(scheduler.best))
        else:
            best_loss = val_loss
            no_impr = 0
            trainer.save_checkpoint(best=True)
            logger.info('Epoch: {:d}, now best loss change: {:.4f}'.format(current_epoch,best_loss))
        # schedule here
        scheduler.step(val_loss)
        # save last checkpoint
        trainer.save_checkpoint(best=False)

        if no_impr == opt["early_stop"]:
            logger.info("Stop training cause no impr for {:d} epochs".format(no_impr))
            break

    logger.info("Training for {:d}/{:d} epoches done!".format(current_epoch,num_epochs))

    

class Trainer(object):
    def __init__(self,args,net,optimizer,scheduler,checkpoint="checkpoint",clip_norm=None,logging_period=100,start_epoch=0):
        # if the cuda is available and if the gpus' type is tuple
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA device unavailable...exist")
       
        # mkdir the file of Experiment path
        if checkpoint and not os.path.exists(checkpoint):
            os.makedirs(checkpoint)
        self.checkpoint = checkpoint

        # build the logger object
        self.logger = get_logger(os.path.join(self.checkpoint, "trainer.log"), file=False)
        self.clip_norm = clip_norm
        self.logging_period = logging_period
        self.current_epoch = start_epoch
        self.net = net
        self.optimizer = optimizer
        self.gpu = args.gpu
        self.scheduler = scheduler
      
        # check model parameters
        self.param = check_parameters(self.net)
        # logging
        self.logger.info("Starting preparing model ............")
        self.logger.info("Loading model to GPUs:{}, #param: {:.2f}M".format(args.gpu, self.param))
        # clip norm
        if self.clip_norm:
            self.logger.info("Gradient clipping by {}, default L2".format(self.clip_norm))
        
    def save_checkpoint(self, best=True):
        '''
            save model
            best: the best model
        '''
        to_save ={"epoch": self.current_epoch,
                "model_state_dict": self.net.state_dict(),
                "optim_state_dict": self.optimizer.state_dict()}
        torch.save(to_save,os.path.join(self.checkpoint, "{0}.pt".format("best" if best else "last")))
        torch.cuda.empty_cache() 

    def to_device(self,dicts):
        if isinstance(dicts,dict):
            return {key:self.to_cuda(dicts[key]) for key in dicts}
        else:
            raise RuntimeError('input egs\'s type is not dict')

    def to_cuda(self,datas):
        if isinstance(datas,torch.Tensor):
            return datas.cuda(self.gpu,non_blocking=True)
        elif isinstance(datas,list):
            return [data.cuda(self.gpu,non_blocking=True) for data in datas]
        else:
            raise RuntimeError("datas is not torch.Tensor.")

    def train(self,train_dataloader):
        '''
           training model
        '''
        self.logger.info('Training model ......')
        self.net.train()
        losses = []
        start = time.time()
        current_step = 0
        for egs in train_dataloader:
            current_step += 1
            egs = self.to_device(egs)
            self.optimizer.zero_grad()
            ests = self.net(egs['mix'])
            loss = si_snr_loss(ests, egs)
            loss.backward()
        
            if self.clip_norm:
                clip_grad_norm_(self.net.parameters(), self.clip_norm)
            self.optimizer.step()
            losses.append(loss.item())

            if len(losses) % self.logging_period == 0:
                avg_loss = sum(losses[-self.logging_period:])/self.logging_period
                self.logger.info('<epoch:{:3d}, iter:{:d}, lr:{:.3e}, loss:{:.3f}, batch:{:d} utterances> '.format(
                    self.current_epoch, current_step, self.optimizer.param_groups[0]['lr'], avg_loss, len(losses)))

        end = time.time()
        total_loss_avg = sum(losses)/len(losses)
        self.logger.info('<epoch:{:3d}, lr:{:.3e}, loss:{:.3f}, Total time:{:.3f} min> '.format(
            self.current_epoch, self.optimizer.param_groups[0]['lr'], total_loss_avg, (end-start)/60))

        return total_loss_avg

    def val(self,val_dataloader):
        '''
           validation model
        '''
        self.logger.info('Validation model ......')
        self.net.eval()
        losses = []
        current_step = 0
        start = time.time()
        with torch.no_grad():
            for egs in val_dataloader:
                current_step += 1
                egs = self.to_device(egs)
                ests = self.net(egs['mix'])
                loss = si_snr_loss(ests, egs)
                losses.append(loss.item())
                if len(losses) % self.logging_period == 0:
                    avg_loss = sum(losses[-self.logging_period:])/self.logging_period
                    self.logger.info('<epoch:{:3d}, iter:{:d}, lr:{:.3e}, loss:{:.3f}, batch:{:d} utterances> '.format(
                        self.current_epoch, current_step, self.optimizer.param_groups[0]['lr'], avg_loss, len(losses)))
        end = time.time()
        total_loss_avg = sum(losses)/len(losses)
        self.logger.info('<epoch:{:3d}, lr:{:.3e}, loss:{:.3f}, Total time:{:.3f} min> '.format(
            self.current_epoch, self.optimizer.param_groups[0]['lr'], total_loss_avg, (end-start)/60))
        return total_loss_avg