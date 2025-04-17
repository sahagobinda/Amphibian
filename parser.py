# coding=utf-8
import os
import argparse

def get_parser():
    parser = argparse.ArgumentParser(description='Online Continual learning')
    parser.add_argument('--run_description', type=str, default='Online Continual learning in Amphibian',
                        help='user comment')
    parser.add_argument('--expt_name', type=str, default='test_amphibian',
                    help='name of the experiment')
    
    # model details
    parser.add_argument('--model', type=str, default='single',
                        help='algo to train')
    parser.add_argument('--arch', type=str, default='linear', 
                        help='arch to use for training', 
                        choices = ['linear', 'pc_cnn','pc_cnn_gpm', 'pc_cnn_gpm_large', 
                        'pc_lenet_gpm' ,'pc_cnn_gpm_wide', 'pc_tnet_cnn', 'pc_cnn_omniglot', 'pc_cnn_minimg'])
    parser.add_argument('--n_hiddens', type=int, default=100,
                        help='number of hidden neurons at each layer')
    parser.add_argument('--n_layers', type=int, default=2,
                        help='number of hidden layers')
    parser.add_argument('--xav_init', default=False , action='store_true',
                        help='Use xavier initialization')


    # optimizer parameters influencing all models
    parser.add_argument("--glances", default=1, type=int,
                        help="Number of times the model is allowed to train over a set of samples in the single pass setting") 
    parser.add_argument('--n_epochs', type=int, default=1,
                        help='Number of epochs per task')
    parser.add_argument('--batch_size', type=int, default=1,
                        help='the amount of items received by the algorithm at one time (set to 1 across all ' +
                        'experiments). Variable name is from GEM project.')
    parser.add_argument('--replay_batch_size', type=float, default=10,
                        help='The batch size for experience replay - for rehearsal-based methods - not in Amphibian.')
    parser.add_argument('--memories', type=int, default=400, 
                        help='number of total memories stored in a reservoir sampling based buffer - for rehearsal-based methods - not in Amphibian')
    # parser.add_argument('--lr', type=float, default=1e-3,
    #                     help='learning rate (For baselines)')
    
    # experiment parameters
    parser.add_argument('--cuda', default=False , action='store_true',
                        help='Use GPU')
    parser.add_argument('--seed', type=int, default=0,
                        help='random seed of model')
    parser.add_argument('--log_every', type=int, default=1000,
                        help='frequency of checking the validation accuracy, in minibatches')
    parser.add_argument('--log_dir', type=str, default='logs/',
                        help='the directory where the logs will be saved')
    parser.add_argument('--tf_dir', type=str, default='',
                        help='(not set by user)')
    parser.add_argument('--calc_test_accuracy', default=False , action='store_true',
                        help='Calculate test accuracy along with val accuracy')

    # data parameters
    parser.add_argument('--data_path', default='data/',
                        help='path where data is located')
    parser.add_argument('--loader', type=str, default='task_incremental_loader',
                        help='data loader to use')
    parser.add_argument('--samples_per_task', type=int, default=-1,
                        help='training samples per task (all if negative)')
    parser.add_argument('--shuffle_tasks', default=False, action='store_true',
                        help='present tasks in order')
    parser.add_argument('--classes_per_it', type=int, default=4,
                        help='number of classes in every batch')
    parser.add_argument('--iterations', type=int, default=5000,
                        help='number of classes in every batch')
    parser.add_argument("--dataset", default="mnist_rotations", type=str,
                    help="Dataset to train and test on.")
    parser.add_argument("--dataset_list", default="mnist-cifar10", type=str,
                    help="Dataset sequence to train and test on.")
    parser.add_argument("--workers", default=3, type=int,
                        help="Number of workers preprocessing the data.")
    parser.add_argument("--validation", default=0., type=float,
                        help="Validation split (0. <= x <= 1.).")
    parser.add_argument("-order", "--class_order", default="old", type=str,
                        help="define classes order of increment ",
                        choices = ["random", "chrono", "old", "super"])
    parser.add_argument("-inc", "--increment", default=5, type=int,
                        help="number of classes to increment by in class incremental loader")
    parser.add_argument('--test_batch_size', type=int, default=100000 ,
                        help='batch size to use during testing.')
    parser.add_argument("-dpc", "--data_per_class", default=500, type=int,
                        help="number of training datapoints per class")
    parser.add_argument("--num_fs_updates", default=20, type=int,
                        help="number of few shot meta test train updates")

 
    # Amphibian parameters
    parser.add_argument('--opt_lr', type=float, default=1e-1,
                        help='learning rate for scales')
    parser.add_argument('--alpha_init', type=float, default=1e-3,
                        help='initialization for the scales')
    parser.add_argument('--weight_reg', type=float, default=5e-1,
                        help='reg coff for the weights- Amphibian-beta')
    parser.add_argument('--learn_lr', default=False, action='store_true',
                        help='model should update the LRs during learning')
    # parser.add_argument('--rand_ortho', default=False , action='store_true',
    #                     help='if random orthogonal basis is used as gradient memory (GPM)')
    parser.add_argument('--momentum', default=0.0, type=float, required=False, help='(default=%(default)f)')
    parser.add_argument('--unique_id', type=str, required=False, default='Gx-Ly-Az-Rp',
                        help='unique identifier: G=glances, L=opt_lr, A=alpha_init, R=weight_reg')
    parser.add_argument('--regularization_layers', type=int, default=5,
                        help='number of layers weight regularization is used for - Amphibian-beta')
    parser.add_argument('--save_stats', default=False , action='store_true',
                        help='if stats are saved')
    parser.add_argument('--mask_type', type=str, default='relu', 
                        help='masks type to use for training', 
                        choices = ['relu', 'relu_STE'])


    parser.add_argument('--grad_clip_norm', type=float, default=2.0,
                        help='Clip the gradients by this value')
    parser.add_argument("--inner_batches", default=3, type=int,
                        help="Number of batches in inner trajectory") 
    parser.add_argument('--second_order', default=False , action='store_true',
                        help='use second order MAML updates')
    parser.add_argument('--n_tasks', type=int, default=100,
                        help='number of tasks')

    # contour plot specific 
    parser.add_argument('--save_opt_path', default=False , action='store_true',
                        help='If True skip gradient projection')
    parser.add_argument('--test_after', default=15, type=int,
                        help='save model after this many iterations in an epoch')

    parser.add_argument('--sync_update', default=False , action='store_true',
                        help='If True outer projection in Amphibian')
    parser.add_argument('--outer_lr', type=float, default=1e-1,
                        help='learning rate for outer projection')

  
    return parser