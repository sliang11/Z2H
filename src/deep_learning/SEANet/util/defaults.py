# coding = utf-8

defaults = {
    'resnet': {
        'model': 'resnet',
            
        'device': 'cuda',

        'checkpoint_filename': 'model.pickle',
        'checkpoint_filepath': None,
        'checkpoint_folderpath': None, # TODO

        'log_filename': 'train.log',
        'log_filepath': None, # TODO

        'default_conf_filename': 'conf.json',
        'conf_filepath': None, # TODO

        'num_epoch': 100,

        'train_positive_samples': None,
        'train_negative_samples': None,
        'valid_positive_samples': None,
        'valid_negative_samples': None,

        # 'size_train': 200000,
        # 'size_val': 10000,
        'batch_size': 256,

        'dim_series': 768,
        'num_input_channels': 5,
        'num_class': 2,
        'size_kernel': 3,
        'num_resblock': 7,
        'dim_latent': 768,
        'num_latent_channels': 256,
        'num_output_channels': 256,

        'resblock_pre_activation': True,

        'dilation_type': 'exponential',
        'dilation_cons': 1,
        'dilation_base': 1,
        'dilation_slope': 1,

        'activation_conv': 'relu',
        'relu_slope': 1e-2,
        'activation_linear': 'lecuntanh',

        'model_init': 'lsuv',
        'lsuv_size': 'default', # -1: to decide according to train size, default: batch_size
        'lsuv_mean': 0,
        'lsuv_std': 1.,
        'lsuv_std_tol': 0.1,
        'lsuv_maxiter': 10,
        'lsuv_ortho': True,

        'layernorm_type': 'layernorm',
        'layernorm_elementwise_affine': True,

        'optim_type': 'sgd',
        'momentum': 0.9,
        'lr_mode': 'linear', # tune
        'lr_cons': 1e-3,
        'lr_max': 1e-3, # tune
        'lr_min': 1e-5,
        'lr_everyk': 2,
        'lr_ebase': 0.9,

        'weightnorm_type': 'none',
        'weightnorm_dim': 0,

        'wd_mode': 'fix', 
        'wd_cons': 1e-4, # tune
        'wd_max': 1e-4,
        'wd_min': 1e-8,

        'adanorm_k': 1 / 10,
        'adanorm_scale': 2.,

        'loss_function': 'ce', # setting
        'ce_weights': 'cvpr19', # setting
        'f_beta': 1,
        'f_alpha': 1,
        'num_training_negative_samples': -1, # setting

        'threshold': 0.5,
        'float32_epsilon': 1e-5,

        'refine_loss_function': 'f1', # setting
        'refine_extra_iterations': 1, # setting
        # 'refine_bpfilter_iterations': {1}, # starting from 1, i.e., {1, 2, 3}, or None
        'refine_bpfilter_iterations': None,

        'warmup': True, # setting
        'warmup_epochs': 7,

        'normalize': False, # setting
        'mu': 0,
        'sigma': 1,

        'sampling_period': 1,
    },
    'res1d18': {
        'model': 'res1d18',
            
        'device': 'cuda',

        'checkpoint_filename': 'model.pickle',
        'checkpoint_filepath': None,
        'checkpoint_folderpath': None, # TODO

        'log_filename': 'train.log',
        'log_filepath': None, # TODO

        'default_conf_filename': 'conf.json',
        'conf_filepath': None, # TODO

        'num_epoch': 100,

        'train_positive_samples': None,
        'train_negative_samples': None,
        'valid_positive_samples': None,
        'valid_negative_samples': None,

        # 'size_train': 200000,
        # 'size_val': 10000,
        'batch_size': 256,

        'dim_series': 768,
        'num_input_channels': 5,
        'num_class': 2,
        # 'size_kernel': 3,
        # 'num_resblock': 7,
        # 'dim_latent': 768,
        # 'num_latent_channels': 256,
        # 'num_output_channels': 256,

        # 'resblock_pre_activation': True,

        # 'dilation_type': 'exponential',
        # 'dilation_cons': 1,
        # 'dilation_base': 1,
        # 'dilation_slope': 1,

        # 'activation_conv': 'relu',
        # 'relu_slope': 1e-2,
        # 'activation_linear': 'lecuntanh',

        'model_init': 'default',
        # 'lsuv_size': 'default', # -1: to decide according to train size, default: batch_size
        # 'lsuv_mean': 0,
        # 'lsuv_std': 1.,
        # 'lsuv_std_tol': 0.1,
        # 'lsuv_maxiter': 10,
        # 'lsuv_ortho': True,

        # 'layernorm_type': 'layernorm',
        # 'layernorm_elementwise_affine': True,

        'optim_type': 'sgd',
        'momentum': 0.9,
        'lr_mode': 'linear', # tune
        'lr_cons': 1e-3,
        'lr_max': 1e-3, # tune
        'lr_min': 1e-5,
        'lr_everyk': 2,
        'lr_ebase': 0.9,

        # 'weightnorm_type': 'none',
        # 'weightnorm_dim': 0,

        # 'wd_mode': 'fix', 
        # 'wd_cons': 1e-4, # tune
        # 'wd_max': 1e-4,
        # 'wd_min': 1e-8,

        # 'adanorm_k': 1 / 10,
        # 'adanorm_scale': 2.,

        'loss_function': 'ce', # setting
        'ce_weights': 'cvpr19', # setting
        'f_beta': 1,
        'f_alpha': 1,
        'num_training_negative_samples': -1, # setting

        'threshold': 0.5,
        'float32_epsilon': 1e-5,

        # 'refine_loss_function': 'f1', # setting
        # 'refine_extra_iterations': 1, # setting
        # # 'refine_bpfilter_iterations': {1}, # starting from 1, i.e., {1, 2, 3}, or None
        # 'refine_bpfilter_iterations': None,

        # 'warmup': True, # setting
        # 'warmup_epochs': 7,

        'normalize': False, # setting
        # 'mu': 0,
        # 'sigma': 1,

        'sampling_period': 1,
    },
    'incept': {
        'model': 'incept',
            
        'device': 'cuda',

        'checkpoint_filename': 'model.pickle',
        'checkpoint_filepath': None,
        'checkpoint_folderpath': None, # TODO

        'log_filename': 'train.log',
        'log_filepath': None, # TODO

        'default_conf_filename': 'conf.json',
        'conf_filepath': None, # TODO

        'num_epoch': 100,

        'train_positive_samples': None,
        'train_negative_samples': None,
        'valid_positive_samples': None,
        'valid_negative_samples': None,

        # 'size_train': 200000,
        # 'size_val': 10000,
        'batch_size': 256,

        'dim_series': 768,
        'num_input_channels': 5,
        'num_class': 2,
        'size_kernel': 3,
        'num_resblock': 7,
        'dim_latent': 768,
        'num_latent_channels': 256,
        'num_output_channels': 256,

        # 'resblock_pre_activation': True,

        # 'dilation_type': 'exponential',
        # 'dilation_cons': 1,
        # 'dilation_base': 1,
        # 'dilation_slope': 1,

        # 'activation_conv': 'relu',
        # 'relu_slope': 1e-2,
        # 'activation_linear': 'lecuntanh',

        'model_init': 'default',
        # 'lsuv_size': 'default', # -1: to decide according to train size, default: batch_size
        # 'lsuv_mean': 0,
        # 'lsuv_std': 1.,
        # 'lsuv_std_tol': 0.1,
        # 'lsuv_maxiter': 10,
        # 'lsuv_ortho': True,
    
        'layernorm_type': 'none',
        # 'layernorm_type': 'layernorm',    
        # 'layernorm_elementwise_affine': True,

        'optim_type': 'sgd',
        'momentum': 0.9,
        'lr_mode': 'linear', # tune
        'lr_cons': 1e-3,
        'lr_max': 1e-3, # tune
        'lr_min': 1e-5,
        'lr_everyk': 2,
        'lr_ebase': 0.9,

        'weightnorm_type': 'none',
        # 'weightnorm_dim': 0,

        # 'wd_mode': 'fix', 
        # 'wd_cons': 1e-4, # tune
        # 'wd_max': 1e-4,
        # 'wd_min': 1e-8,

        # 'adanorm_k': 1 / 10,
        # 'adanorm_scale': 2.,

        'loss_function': 'ce', # setting
        'ce_weights': 'cvpr19', # setting
        'f_beta': 1,
        'f_alpha': 1,
        'num_training_negative_samples': -1, # setting

        'threshold': 0.5,
        'float32_epsilon': 1e-5,

        # 'refine_loss_function': 'f1', # setting
        # 'refine_extra_iterations': 1, # setting
        # # 'refine_bpfilter_iterations': {1}, # starting from 1, i.e., {1, 2, 3}, or None
        # 'refine_bpfilter_iterations': None,

        # 'warmup': True, # setting
        # 'warmup_epochs': 7,

        'normalize': False, # setting
        # 'mu': 0,
        # 'sigma': 1,

        'sampling_period': 1,
    },
    'rnn': {
        'model': 'rnn',
        'rnn_type': 'lstm',

        'if_rnn_bidirectional': True,
        'rnn_dropout': 0.5,
        'num_rnn_layers': 3,
            
        'device': 'cuda',

        'checkpoint_filename': 'model.pickle',
        'checkpoint_filepath': None,
        'checkpoint_folderpath': None, # TODO

        'log_filename': 'train.log',
        'log_filepath': None, # TODO

        'default_conf_filename': 'conf.json',
        'conf_filepath': None, # TODO

        'num_epoch': 100,

        'train_positive_samples': None,
        'train_negative_samples': None,
        'valid_positive_samples': None,
        'valid_negative_samples': None,

        # 'size_train': 200000,
        # 'size_val': 10000,
        'batch_size': 64,

        'dim_series': 768,
        'num_input_channels': 5,
        'num_class': 2,
        'size_kernel': 3,
        'num_resblock': 7,
        'dim_latent': 768,
        'num_latent_channels': 256,
        'num_output_channels': 256,

        'resblock_pre_activation': True,

        'dilation_type': 'exponential',
        'dilation_cons': 1,
        'dilation_base': 1,
        'dilation_slope': 1,

        'activation_conv': 'relu',
        'relu_slope': 1e-2,
        'activation_linear': 'lecuntanh',

        'model_init': 'default',
        'lsuv_size': 'default', # -1: to decide according to train size, default: batch_size
        'lsuv_mean': 0,
        'lsuv_std': 1.,
        'lsuv_std_tol': 0.1,
        'lsuv_maxiter': 10,
        'lsuv_ortho': True,

        'layernorm_type': 'none',
        'layernorm_elementwise_affine': True,

        'optim_type': 'sgd',
        'momentum': 0.9,
        'lr_mode': 'linear', # tune
        'lr_cons': 1e-3,
        'lr_max': 1e-3, # tune
        'lr_min': 1e-5,
        'lr_everyk': 2,
        'lr_ebase': 0.9,

        'weightnorm_type': 'none',
        'weightnorm_dim': 0,

        'wd_mode': 'fix', 
        'wd_cons': 1e-4, # tune
        'wd_max': 1e-4,
        'wd_min': 1e-8,

        'adanorm_k': 1 / 10,
        'adanorm_scale': 2.,

        'loss_function': 'ce', # setting
        'ce_weights': 'cvpr19', # setting
        'f_beta': 1,
        'f_alpha': 1,
        'num_training_negative_samples': -1, # setting

        'threshold': 0.5,
        'float32_epsilon': 1e-5,

        # 'refine_loss_function': 'f1', # setting
        # 'refine_extra_iterations': 1, # setting
        # # 'refine_bpfilter_iterations': {1}, # starting from 1, i.e., {1, 2, 3}, or None
        # 'refine_bpfilter_iterations': None,

        'warmup': False, # setting
        'warmup_epochs': 7,

        'normalize': False, # setting
        'mu': 0,
        'sigma': 1,

        'sampling_period': 1,
    },
    'transformer': {
        'model': 'transformer',
            
        'device': 'cuda',

        'checkpoint_filename': 'model.pickle',
        'checkpoint_filepath': None,
        'checkpoint_folderpath': None, # TODO

        'log_filename': 'train.log',
        'log_filepath': None, # TODO

        'default_conf_filename': 'conf.json',
        'conf_filepath': None, # TODO

        'num_epoch': 100,

        'train_positive_samples': None,
        'train_negative_samples': None,
        'valid_positive_samples': None,
        'valid_negative_samples': None,

        # 'size_train': 200000,
        # 'size_val': 10000,
        # 'batch_size': 256,
        # 'batch_size': 64,
        'batch_size': 32,

        'dim_series': 768,
        'num_input_channels': 5,
        'num_class': 2,
        'size_kernel': 3,
        'num_resblock': 7,
        'dim_latent': 768,
        'num_latent_channels': 256,
        'num_output_channels': 256,

        # 'resblock_pre_activation': True,

        # 'dilation_type': 'exponential',
        # 'dilation_cons': 1,
        # 'dilation_base': 1,
        # 'dilation_slope': 1,

        # 'activation_conv': 'relu',
        # 'relu_slope': 1e-2,
        # 'activation_linear': 'lecuntanh',

        # 'model_init': 'lsuv',
        # 'lsuv_size': 'default', # -1: to decide according to train size, default: batch_size
        # 'lsuv_mean': 0,
        # 'lsuv_std': 1.,
        # 'lsuv_std_tol': 0.1,
        # 'lsuv_maxiter': 10,
        # 'lsuv_ortho': True,

        # 'layernorm_type': 'layernorm',
        # 'layernorm_elementwise_affine': True,

        'optim_type': 'sgd',
        'momentum': 0.9,
        'lr_mode': 'linear', # tune
        'lr_cons': 1e-3,
        'lr_max': 1e-3, # tune
        'lr_min': 1e-5,
        'lr_everyk': 2,
        'lr_ebase': 0.9,

        # 'weightnorm_type': 'none',
        # 'weightnorm_dim': 0,

        'wd_mode': 'fix', 
        'wd_cons': 1e-4, # tune
        'wd_max': 1e-4,
        'wd_min': 1e-8,

        # 'adanorm_k': 1 / 10,
        # 'adanorm_scale': 2.,

        'loss_function': 'ce', # setting
        'ce_weights': 'cvpr19', # setting
        'f_beta': 1,
        'f_alpha': 1,
        'num_training_negative_samples': -1, # setting

        'threshold': 0.5,
        'float32_epsilon': 1e-5,

        # 'refine_loss_function': 'f1', # setting
        # 'refine_extra_iterations': 1, # setting
        # # 'refine_bpfilter_iterations': {1}, # starting from 1, i.e., {1, 2, 3}, or None
        # 'refine_bpfilter_iterations': None,

        # 'warmup': True, # setting
        # 'warmup_epochs': 7,

        'normalize': False, # setting
        'mu': 0,
        'sigma': 1,

        # 'sampling_period': 1,

        'transformer_d_model': 64,
        'transformer_n_heads': 8,
        'transformer_n_layers': 3,
        'transformer_d_feedforward': 256,
        'transformer_dropout': 0.1,
        'transformer_pos_encoding': 'fixed',
        'transformer_activation': 'gelu',
        'transformer_t_norm': 'LayerNorm',
        'transformer_freeze': False,
    },
}
