# import ipdb

class ModelFactory():
    def __init__(self):
        pass

    @staticmethod
    def get_model(model_type, sizes, dataset='mnist', args=None):

        net_list = []
        if "mnist" in dataset:
            if model_type=="linear":
                for i in range(0, len(sizes) - 1):
                    net_list.append(('linear', [sizes[i+1], sizes[i]], ''))
                    if i < (len(sizes) - 2):
                        net_list.append(('relu', [True], ''))
                    if i == (len(sizes) - 2):
                        net_list.append(('rep', [], ''))
                return net_list

        elif dataset == "tinyimagenet" or dataset == "miniImageNet" :

            if model_type == 'pc_cnn':
                channels = 160
                return [
                    ('conv2d', [channels, 3, 3, 3, 2, 1], ''),
                    ('relu', [True], ''),

                    ('conv2d', [channels, channels, 3, 3, 2, 1], ''),
                    ('relu', [True], ''),

                    ('conv2d', [channels, channels, 3, 3, 2, 1], ''),
                    ('relu', [True], ''),

                    ('conv2d', [channels, channels, 3, 3, 2, 1], ''),
                    ('relu', [True], ''),

                    ('flatten', [], ''),
                    ('rep', [], ''),

                    ('linear', [640, 16 * channels], ''),
                    ('relu', [True], ''),

                    ('linear', [640, 640], ''),
                    ('relu', [True], ''),
                    ('linear', [sizes[-1], 640], '')
                ]

            elif model_type == 'pc_cnn_gpm':
                # will be used in AMPHIBIAN - tinyimagenet experiment 
                channels = 160
                return [
                    ('conv2d', [channels, 3, 3, 3, 2, 1], 'gpm'),
                    ('relu', [True], ''),

                    ('conv2d', [channels, channels, 3, 3, 2, 1], 'gpm'),
                    ('relu', [True], ''),

                    ('conv2d', [channels, channels, 3, 3, 2, 1], 'gpm'),
                    ('relu', [True], ''),

                    ('conv2d', [channels, channels, 3, 3, 2, 1], 'gpm'),
                    ('relu', [True], ''),

                    ('flatten', [], ''),
                    ('rep', [], ''),

                    ('linear', [640, 16 * channels], 'gpm'),
                    ('relu', [True], ''),

                    ('linear', [640, 640], 'gpm'),
                    ('relu', [True], ''),
                    ('linear', [sizes[-1], 640], '')
                ]

            elif model_type == 'pc_cnn_minimg':
                channels = 160
                return [
                    ('conv2d', [channels, 3, 3, 3, 2, 1], 'gpm'),
                    ('relu', [True], ''),

                    ('conv2d', [channels, channels, 3, 3, 2, 1], 'gpm'),
                    ('relu', [True], ''),

                    ('conv2d', [channels, channels, 3, 3, 2, 1], 'gpm'),
                    ('relu', [True], ''),

                    ('conv2d', [channels, channels, 3, 3, 2, 1], 'gpm'),
                    ('relu', [True], ''),

                    ('conv2d', [channels, channels, 3, 3, 1, 0], 'gpm'),
                    # ('conv2d', [channels, channels, 2, 2, 2, 1], 'gpm'),
                    ('relu', [True], ''),

                    ('flatten', [], ''),
                    ('rep', [], ''),

                    ('linear', [640, 16 * channels], 'gpm'),
                    ('relu', [True], ''),

                    ('linear', [640, 640], 'gpm'),
                    ('relu', [True], ''),
                    ('linear', [sizes[-1], 640], '')
                ]


        elif dataset == "cifar100":


            if model_type == 'pc_cnn':
                channels = 160
                return [
                    ('conv2d', [channels, 3, 3, 3, 2, 1], ''),
                    ('relu', [True], ''),
                    
                    ('conv2d', [channels, channels, 3, 3, 2, 1], ''),
                    ('relu', [True], ''),

                    ('conv2d', [channels, channels, 3, 3, 2, 1], ''),
                    ('relu', [True], ''),

                    ('flatten', [], ''),
                    # ('rep', [], ''),

                    ('linear', [320, 16 * channels], ''),
                    ('relu', [True], ''),

                    ('linear', [320, 320], ''),
                    ('relu', [True], ''),
                    ('rep', [], ''),
                    ('linear', [sizes[-1], 320], '')
                ]

            elif model_type == 'pc_cnn_gpm':
                # will be used in AMPHIBIAN - CIFAR100/5 dataset experiment 
                channels = 160
                return [
                    ('conv2d', [channels, 3, 3, 3, 2, 1], 'gpm'),
                    ('relu', [True], ''),
                    
                    ('conv2d', [channels, channels, 3, 3, 2, 1], 'gpm'),
                    ('relu', [True], ''),

                    ('conv2d', [channels, channels, 3, 3, 2, 1], 'gpm'),
                    ('relu', [True], ''),

                    ('flatten', [], ''),
                    ('rep', [], ''),

                    ('linear', [320, 16 * channels], 'gpm'),
                    ('relu', [True], ''),

                    ('linear', [320, 320], 'gpm'),
                    ('relu', [True], ''),
                    # ('rep', [], ''),
                    ('linear', [sizes[-1], 320], '')
                ]

            elif model_type == 'pc_lenet_gpm':
                # will be used in AMPHIBIAN - CIFAR100/5 dataset experiment 
                channels = 64
                return [
                    ('conv2d', [channels, 3, 5, 5, 2, 0], 'gpm'),
                    ('relu', [True], ''),
                    
                    ('conv2d', [channels, channels, 5, 5, 2, 0], 'gpm'),
                    ('relu', [True], ''),

                    ('flatten', [], ''),
                    ('rep', [], ''),

                    ('linear', [384, 25 * channels], 'gpm'),
                    ('relu', [True], ''),

                    ('linear', [192, 384], 'gpm'),
                    ('relu', [True], ''),
                    # ('rep', [], ''),
                    ('linear', [sizes[-1], 192], '')
                ]

            elif model_type == 'pc_cnn_gpm_wide':
                # will be used in AMPHIBIAN - CIFAR100/5 dataset experiment 
                print ('>> using 1.25x wider CNN model for 5DExp...')
                channels = 200 #1.25x model
                return [
                    ('conv2d', [channels, 3, 3, 3, 2, 1], 'gpm'),
                    ('relu', [True], ''),
                    
                    ('conv2d', [channels, channels, 3, 3, 2, 1], 'gpm'),
                    ('relu', [True], ''),

                    ('conv2d', [channels, channels, 3, 3, 2, 1], 'gpm'),
                    ('relu', [True], ''),

                    ('flatten', [], ''),
                    ('rep', [], ''),

                    ('linear', [400, 16 * channels], 'gpm'),
                    ('relu', [True], ''),

                    ('linear', [400, 400], 'gpm'),
                    ('relu', [True], ''),
                    # ('rep', [], ''),
                    ('linear', [sizes[-1], 400], '')
                ]

            elif model_type == 'pc_cnn_gpm_large':
                # will be used in AMPHIBIAN - tinyimagenet experiment 
                print ('>> using larger CNN model for 5DExp...')
                channels = 160
                return [
                    ('conv2d', [channels, 3, 3, 3, 2, 1], 'gpm'),
                    ('relu', [True], ''),

                    ('conv2d', [channels, channels, 3, 3, 2, 1], 'gpm'),
                    ('relu', [True], ''),

                    ('conv2d', [channels, channels, 3, 3, 2, 1], 'gpm'),
                    ('relu', [True], ''),

                    ('conv2d', [channels, channels, 3, 3, 2, 1], 'gpm'),
                    ('relu', [True], ''),

                    ('flatten', [], ''),
                    ('rep', [], ''),

                    ('linear', [640, 4 * channels], 'gpm'),
                    ('relu', [True], ''),

                    ('linear', [640, 640], 'gpm'),
                    ('relu', [True], ''),
                    ('linear', [sizes[-1], 640], '')
                ]

            elif model_type == 'pc_cnn_omniglot':
                channels = 160
                return [
                    ('conv2d', [channels, 1, 3, 3, 2, 1], 'gpm'),
                    ('relu', [True], ''),
                    
                    ('conv2d', [channels, channels, 3, 3, 2, 1], 'gpm'),
                    ('relu', [True], ''),

                    ('conv2d', [channels, channels, 3, 3, 2, 1], 'gpm'),
                    ('relu', [True], ''),

                    ('flatten', [], ''),
                    ('rep', [], ''),

                    ('linear', [320, 16 * channels], 'gpm'),
                    ('relu', [True], ''),

                    ('linear', [320, 320], 'gpm'),
                    ('relu', [True], ''),
                    # ('rep', [], ''),
                    ('linear', [sizes[-1], 320], '')
                ]

            elif model_type == 'pc_tnet_cnn':
                channels = 160
                return [
                    ('conv2d', [channels, 3, 3, 3, 2, 1], 'adaptation'),
                    ('tnet', [channels, channels, 1, 1, 1, 0], ''),
                    ('mnet', [channels], ''),
                    ('relu', [True], ''),
                    
                    ('conv2d', [channels, channels, 3, 3, 2, 1], 'adaptation'),
                    ('tnet', [channels, channels, 1, 1, 1, 0], ''),
                    ('mnet', [channels], ''),
                    ('relu', [True], ''),

                    ('conv2d', [channels, channels, 3, 3, 2, 1], 'adaptation'),
                    ('tnet', [channels, channels, 1, 1, 1, 0], ''),
                    ('mnet', [channels], ''),
                    ('relu', [True], ''),

                    ('flatten', [], ''),
                    ('rep', [], ''),

                    ('linear', [320, 16 * channels], 'adaptation'),
                    ('tnet_fc', [320, 320], ''),
                    ('mnet', [320], ''),
                    ('relu', [True], ''),

                    ('linear', [320, 320], 'adaptation'),
                    ('tnet_fc', [320, 320], ''),
                    ('mnet', [320], ''),
                    ('relu', [True], ''),
                    ('linear', [sizes[-1], 320], 'adaptation')
                ]

        else:
            print("Unsupported model; either implement the model in model/ModelFactory or choose a different model")
            assert (False)



 