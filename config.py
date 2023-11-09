import ml_collections

def get_config():
    """Returns the ViT-B/16 configuration."""
    config = ml_collections.ConfigDict()
    config.hidden_size = 768
    config.input_length = 101
    config.num_classes = 3
    config.in_channels = 24
    
    config.train_batch_size = 40
    config.eval_batch_size = 40
    config.learning_rate = 1e-4 
    config.num_steps = 50000
    config.eval_every = 500
    config.max_grad_norm = 1.0
    config.gradient_steps = 1
    config.local_rank  = -1 
    config.decay_type = "cosine" 
    config.warmup_steps = 500 
    config.split = True   # True to split the training dataset into train and validation 
    config.split_ratio = 0.1     # The ratio of the split 
    config.shuffle = True        # Shuffle the training and validation samples. 
    return config