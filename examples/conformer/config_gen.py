from tensorflow_asr.utils import file_util
from pprint import pprint

def get_config(
    path: str,
    conv_filters: int,
    hdim_model: int,
    conf_blocks: int,
    mhsa_heads: int,
    epochs: int,
    batch_size: int,
    reduced_train: bool,
    reduced_eval: bool
) -> dict:

    config = file_util.load_yaml(file_util.preprocess_paths(path))

    # pprint(config)
    # print('-'*40)

    if hdim_model:
        config['model_config']['encoder_dmodel'] = hdim_model

    if conf_blocks:
        config['model_config']['encoder_num_blocks'] = conf_blocks
        
    if mhsa_heads:
        config['model_config']['encoder_num_heads'] = mhsa_heads
        
    if conv_filters:
        config['model_config']['encoder_subsampling']['filters'] = conv_filters

    if batch_size:
        config['learning_config']['running_config']['batch_size'] = batch_size
        
    if epochs:
        config['learning_config']['running_config']['num_epochs'] = epochs

    file_path = '_'.join(map(str, [
        config['model_config']['encoder_subsampling']['filters'], 
        config['model_config']['encoder_dmodel'], 
        config['model_config']['encoder_num_blocks'], 
        config['model_config']['encoder_num_heads']
    ]))
    
    config['learning_config']['running_config']['checkpoint']['filepath'] = set_path(config['learning_config']['running_config']['checkpoint']['filepath'], \
                                                                                    file_path, -2)
    config['learning_config']['running_config']['tensorboard']['log_dir'] = set_path(config['learning_config']['running_config']['tensorboard']['log_dir'], \
                                                                                    file_path, -1)
    config['learning_config']['running_config']['states_dir'] = set_path(config['learning_config']['running_config']['states_dir'], \
                                                                                     file_path, -1)

    if reduced_train:
        train_path = config['learning_config']['train_dataset_config']['data_paths'][0]
        train_path = train_path.split('/')
        train_path[-1] = 'transcripts_reduced.tsv'
        config['learning_config']['train_dataset_config']['data_paths'] = ['/'.join(train_path)]

        decoder_corpus_path = config['decoder_config']['corpus_files'][0]
        decoder_corpus_path = decoder_corpus_path.split('/')
        decoder_corpus_path[-1] = 'transcripts_reduced.tsv'
        config['decoder_config']['corpus_files'] = ['/'.join(decoder_corpus_path)]
    
    if reduced_eval:
        eval_path = config['learning_config']['eval_dataset_config']['data_paths'][0]
        eval_path = eval_path.split('/')
        eval_path[-1] = 'transcripts_reduced.tsv'
        config['learning_config']['eval_dataset_config']['data_paths'] = ['/'.join(eval_path)]

    return config

def set_path(
    path_dir, 
    path_pfx, 
    pos
) -> str:

    path_dir = path_dir.split('/')
    path_dir[pos] = path_dir[pos] + '_' + path_pfx

    return '/'.join(path_dir)
    
# if __name__ == '__main__':
#     pprint(get_config('config.yml', 128, 540, 20, 6, 10, 5, True, True))