import random
import numpy as np
# import ipdb
import math
import torch
import torch.nn as nn
from model.amphibian_base import *
import pdb

class ReluStraightThrough(torch.autograd.Function):
    def __init__(self):
        super(ReluStraightThrough, self).__init__()

    @staticmethod
    def forward(ctx, input):
        return torch.relu(input)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output

class Net(BaseNet):

    def __init__(self,
                 n_inputs,
                 n_outputs,
                 n_tasks,           
                 args):
        super(Net, self).__init__(n_inputs,
                                  n_outputs,
                                  n_tasks,           
                                  args)
        self.nc_per_task = n_outputs / n_tasks
        self.fs_updates  = args.num_fs_updates
        self.regularization_layers = args.regularization_layers # For Amphibian-beta

        if args.mask_type == "relu_STE":
            self.mask = ReluStraightThrough.apply
        else:
            self.mask = nn.functional.relu

        if self.args.sync_update:
            print('-'*50)
            print ('>> No Outer Projection/Scaling in Amphibian ----<<')
            print ('Outer Lr: {}'.format(self.args.outer_lr))
            print('-'*50)
        else:
            print ('>> Amphibian Learner ---- <<')
        
        print('='*100)
        print('Arguments =')
        for arg in vars(args):
            print('\t'+arg+':',getattr(args,arg))
        print('='*100)

    def take_loss(self, t, logits, y):
        # compute loss on data from a single task
        offset1, offset2 = self.compute_offsets(t)
        loss = self.loss(logits[:, offset1:offset2], y-offset1)

        return loss

    def take_multitask_loss(self, bt, t, logits, y):
        # compute loss on data from a multiple tasks
        # separate from take_loss() since the output positions for each task's
        # logit vector are different and we only want to compute loss on the relevant positions
        # since this is a task incremental setting

        loss = 0.0

        for i, ti in enumerate(bt):
            offset1, offset2 = self.compute_offsets(ti)
            loss += self.loss(logits[i, offset1:offset2].unsqueeze(0), y[i].unsqueeze(0)-offset1)
        return loss/len(bt)


    def forward(self, x, t):
        output = self.net.forward(x)
        # make sure we predict classes within the current task
        offset1, offset2 = self.compute_offsets(t)
        if offset1 > 0:
            output[:, :offset1].data.fill_(-10e10)
        if offset2 < self.n_outputs:
            output[:, int(offset2):self.n_outputs].data.fill_(-10e10)
        return output

    def meta_loss(self, x, fast_weights, y, bt, t):
        """
        differentiate the loss through the network updates wrt alpha
        """

        offset1, offset2 = self.compute_offsets(t)

        logits = self.net.forward(x, fast_weights)[:, :offset2]
        loss_q = self.take_multitask_loss(bt, t, logits, y)

        return loss_q, logits

    def inner_update(self, x, fast_weights, y, t, fs_update=False):
        """
        Update the fast weights using the current samples and return the updated fast
        """
        gpm_cnt = 0
        other_cnt = 0 
        offset1, offset2 = self.compute_offsets(t)            

        logits = self.net.forward(x, fast_weights)[:, :offset2]
        loss = self.take_loss(t, logits, y)

        if fast_weights is None:
            fast_weights = self.net.parameters() # network weights/bias - neeed during forward Prop 

        # Compute Gradients : NOTE if we want higher order grads to be allowed, change create_graph=False to True
        if fs_update:
            grads = list(torch.autograd.grad(loss, fast_weights, create_graph=False, retain_graph=False))
        else:
            graph_required = self.args.second_order
            grads = list(torch.autograd.grad(loss, fast_weights, create_graph=graph_required, retain_graph=graph_required))
        # clip gradients 
        for i in range(len(grads)):
            grads[i] = torch.clamp(grads[i], min = -self.args.grad_clip_norm, max = self.args.grad_clip_norm)

        # update weights/biases/BN's    
        new_weights = []
        for idx, (params, g) in enumerate(zip(fast_weights, grads)):
            g = grads[idx]
            if params.gpm:                
                sz = g.data.size(0)
                temp_weight = params -  torch.mm(g.data.view(sz,-1), torch.diag(self.mask(self.net.alpha_lr[gpm_cnt]))).view(params.size())
                temp_weight.gpm = params.gpm
                new_weights.append(temp_weight)
                gpm_cnt += 1
            else:
                temp_weight = params -  g * self.mask(self.net.alpha_other_lr[other_cnt])
                temp_weight.gpm = params.gpm
                new_weights.append(temp_weight)
                other_cnt += 1
   
        return new_weights


    def observe_fs(self, x, y, t, test_tasks, seq_tasks=False):
        """
        perform few-shot model update -> "fast_weights"
        then evaluate few shot performance on the given (t) task 
        """
        self.net.train() 
        perm = torch.randperm(x.size(0))
        x = x[perm]
        y = y[perm]

        self.zero_grads()

        batch_sz = x.shape[0]
        fast_weights = None

        for i in range(self.fs_updates):
            batch_x = x
            batch_y = y
            # take fs steps 
            fast_weights = self.inner_update(batch_x, fast_weights, batch_y, t, True)   

        self.zero_grads()
        
        ## test Few-Shot learning 
        self.net.eval()
        result = 0
        offset1, offset2 = self.compute_offsets(t)

        ## train ACC
        rt = 0
        output = self.net.forward(x, fast_weights)
        if offset1 > 0:
            output[:, :offset1].data.fill_(-10e10)
        if offset2 < self.n_outputs:
            output[:, int(offset2):self.n_outputs].data.fill_(-10e10)

        _, p = torch.max(output.data.cpu(), 1, keepdim=False)
        rt += (p == y.cpu()).float().sum()
        train_acc = rt/batch_sz
        # print('training accuracy: {}', train_acc)

        ## test ACC
        if seq_tasks:
            # for seqential 5 dataset 
            for idx, tasks in enumerate(test_tasks):
                for t_id, task_loader in enumerate(tasks):
                    task_id = t_id + idx*self.args.num_datasets 
                    if task_id == t:
                        rt = 0
                        for (i, (x, y)) in enumerate(task_loader):
                            if self.args.cuda:
                                x = x.cuda()
                            y = y+idx*self.args.class_per_dataset
                            
                            output = self.net.forward(x, fast_weights)
                            if offset1 > 0:
                                output[:, :offset1].data.fill_(-10e10)
                            if offset2 < self.n_outputs:
                                output[:, int(offset2):self.n_outputs].data.fill_(-10e10)
                            
                            _, p = torch.max(output.data.cpu(), 1, keepdim=False)
                            rt += (p == y).float().sum()

                        result = rt / len(task_loader.dataset)
        
        else:
            # for cifar/tinyimagenet dataset             
            for task_id, task_loader in enumerate(test_tasks):
                if task_id == t:
                    rt = 0
                    for (i, (x, y)) in enumerate(task_loader):
                        if self.args.cuda:
                            x = x.cuda()
                        output = self.net.forward(x, fast_weights)
                        if offset1 > 0:
                            output[:, :offset1].data.fill_(-10e10)
                        if offset2 < self.n_outputs:
                            output[:, int(offset2):self.n_outputs].data.fill_(-10e10)

                        _, p = torch.max(output.data.cpu(), 1, keepdim=False)
                        rt += (p == y).float().sum()

                    result = rt / len(task_loader.dataset)
        
        return train_acc.item(), result.item()


    def observe(self, x, y, t):
        self.net.train() 
        for pass_itr in range(self.glances):
            self.pass_itr = pass_itr
            perm = torch.randperm(x.size(0))
            x = x[perm]
            y = y[perm]

            self.epoch += 1
            self.zero_grads()

            if t != self.current_task:
                self.current_task = t

            # selfC: original data batch is partitioned into 'n_batches' -each having 'rough_sz' samples. 
            # selfC: e.g., if original batch size was 10, for "cifar_batches =5", 
            # we will have 5-minibatch inside the minibatch and each small minibatch will have 2 samples
            batch_sz = x.shape[0] 
            n_batches = self.args.inner_batches 
            rough_sz = math.ceil(batch_sz/n_batches)
            fast_weights = None
            meta_losses = [0 for _ in range(n_batches)]

            # get a batch of given data for computing meta-loss
            bx, by, bt = self.getMetaBatch(x.cpu().numpy(), y.cpu().numpy(), t)  # returns only current batch data            

            for i in range(n_batches):

                batch_x = x[i*rough_sz : (i+1)*rough_sz]
                batch_y = y[i*rough_sz : (i+1)*rough_sz]

                # do inner loop update - updated parameters:  
                fast_weights = self.inner_update(batch_x, fast_weights, batch_y, t)   
                # record meta-loss 
                meta_loss, logits = self.meta_loss(bx, fast_weights, by, bt, t)                 
                meta_losses[i] += meta_loss

            # Taking the meta gradient step (will update the learning rates)
            self.zero_grads()
            meta_loss = sum(meta_losses)/len(meta_losses)            
            meta_loss.backward()

            # clip gradients 
            torch.nn.utils.clip_grad_norm_(self.net.alpha_lr.parameters(), self.args.grad_clip_norm)
            torch.nn.utils.clip_grad_norm_(self.net.alpha_other_lr.parameters(), self.args.grad_clip_norm)
            torch.nn.utils.clip_grad_norm_(self.net.parameters(), self.args.grad_clip_norm)
            
            # First update the learning rate parameters 
            if self.args.learn_lr:
                self.opt_lr.step()
                self.opt_other_lr.step()

            if self.args.sync_update:
                for i,p in enumerate(self.net.parameters()): 
                    p.data = p.data - self.args.outer_lr * p.grad
            else:
                # Next update the weights with sgd using updated LRs as step sizes
                gpm_cnt = 0   #counts the number of scaled projection 
                other_cnt = 0 #counts the number of bias lr scaling              
                for i,p in enumerate(self.net.parameters()):          
                    if p.gpm:
                        sz = p.grad.data.size(0)
                        # Gradient Scaling 
                        p.grad.data = torch.mm(p.grad.data.view(sz,-1),torch.diag(self.mask(self.net.alpha_lr[gpm_cnt]))).view(p.size())           
                        # model update 
                        if gpm_cnt<self.regularization_layers and self.current_task>0:
                            p.data = p.data -   p.grad - self.weight_reg *(p.data - self.old_net_params[i])
                        else:
                            p.data = p.data -   p.grad
                        gpm_cnt +=1

                    else:   
                        if other_cnt<self.regularization_layers and self.current_task>0:
                            p.data = p.data -  p.grad * self.mask(self.net.alpha_other_lr[other_cnt]) - self.weight_reg*(p.data - self.old_net_params[i])
                        else:
                            p.data = p.data -  p.grad * self.mask(self.net.alpha_other_lr[other_cnt])
                        other_cnt +=1

            self.net.zero_grad() # model_wt
            self.net.alpha_lr.zero_grad() # Scale_diag
            self.net.alpha_other_lr.zero_grad() # bias/bn params

        return meta_loss.item()

    