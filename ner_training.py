import time
from mitie import *
from ner_utils_func import BsfInfo, parse_bsf, read_train_test_split
import os
from tqdm import tqdm



def prepare_mitie_training_data(dev_files):
    # convert char offset in ner-uk markup to token based MITIE markup
    base_path= "/Applications/CS_diploma_code/train_data/" #folder with all training data
    samples = []
    for f_name in dev_files:
        # read ann
        with open(base_path + f_name + '.ann', 'r') as f:
            annotations = parse_bsf(f.read())
        # read tokens
        with open(base_path + f_name + '.txt', 'r') as f:
            tok_txt = f.read()

        tokens = tok_txt.split()

        # convert char offset to token offset
        tok_ann = []
        tok_idx = 0

        ann: BsfInfo
        for ann in annotations:
            tok_start = 0
            in_token = False
            tok_end = 0
            for i in range(tok_idx, len(tokens)):
                tok_idx = i + 1
                if not in_token and ann.token.startswith(tokens[i]):
                    tok_start = i
                    tok_end = i + 1
                    in_token = (len(ann.token) != len(tokens[i]))
                    if len(ann.token) == len(tokens[i]):
                        break
                elif in_token and ann.token.endswith(tokens[i]):
                    tok_end = i + 1
                    in_token = False
                    break
            tok_ann.append(BsfInfo(ann.id, ann.tag, tok_start, tok_end, ann.token))

        # Create MITIE sample
        sample = ner_training_instance(tokens)
        for t_ann in tok_ann:
            sample.add_entity(xrange(t_ann.start_idx, t_ann.end_idx), t_ann.tag)
        samples.append(sample)

    print(f'Converted to MITIE format. Sample documents {len(samples)}')
    return samples


def run_training(cpu_threads, config_path, feature_extractor_path):
    dev_files, test_files = read_train_test_split(config_path)
    print(f'Loaded corpus file split configuration (documents): DEV={len(dev_files)}, TEST={len(test_files)}')

    samples = prepare_mitie_training_data(dev_files)

    # check for workspace folder existence
    workspace_folder = os.path.join('workspace', 'mitie')
    
    """
    if not os.path.exists(workspace_folder):
        os.makedirs(workspace_folder)
    """

    # Training
    trainer = ner_trainer(feature_extractor_path)

    for s in samples:
        trainer.add(s)

    trainer.num_threads = cpu_threads

    print("Launching training process...")
    # takes long here
    ner = trainer.train()

    model_path = os.path.join(workspace_folder, "mitie_ner_model_ver1.dat")
    ner.save_to_disk(model_path)
    print(f'Training finished. Model saved to "{model_path}"')


if __name__ == '__main__':
    t1=time.perf_counter()
    if not os.path.exists('/Applications/CS_diploma_code/train_data'):
        print("Error: data folder not found")
    else:
        run_training(8, 'dev-test-split.txt', 'total_word_feature_extractor.dat')
    t2=time.perf_counter()
    print(f'Finished in {t2-t1} seconds')

