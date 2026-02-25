import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import tensorflow as tf
import tensorflow_datasets as tfds
import torch
import shutil
import json
from transformers import T5Tokenizer, T5EncoderModel
from tqdm import tqdm
import numpy as np
from PIL import Image
import random

DATASET_VERSION = '0.0.1'
DATASET_NAME = 'language_table'
OUTPUT_DIR = '/data/lang_table'  # Directory to save the preprocessed data
MAX_INSTRUCTION_LENGTH = 32  # Maximum length for padded instructions
T5_MODEL_NAME = 't5-large'  # You can choose a different T5 model
RESIZE_RESOLUTION = (256, 256)  # Set to None to save original size
VALID_RATIO = 0.1  # Ratio of data to use for the validation set
TEST_RATIO = 0.1   # Ratio of data to use for the test set
MIN_EP_LEN = 15
MAX_EP_LEN = 50
RANDOM_SEED = 42  # For reproducibility

dataset_directories = {
    'language_table': 'gs://gresearch/robotics/language_table',
    'language_table_sim': 'gs://gresearch/robotics/language_table_sim',
    'language_table_blocktoblock_sim': 'gs://gresearch/robotics/language_table_blocktoblock_sim',
    'language_table_blocktoblock_4block_sim': 'gs://gresearch/robotics/language_table_blocktoblock_4block_sim',
    'language_table_blocktoblock_oracle_sim': 'gs://gresearch/robotics/language_table_blocktoblock_oracle_sim',
    'language_table_blocktoblockrelative_oracle_sim': 'gs://gresearch/robotics/language_table_blocktoblockrelative_oracle_sim',
    'language_table_blocktoabsolute_oracle_sim': 'gs://gresearch/robotics/language_table_blocktoabsolute_oracle_sim',
    'language_table_blocktorelative_oracle_sim': 'gs://gresearch/robotics/language_table_blocktorelative_oracle_sim',
    'language_table_separate_oracle_sim': 'gs://gresearch/robotics/language_table_separate_oracle_sim',
}

def decode_inst(inst):
  """Utility to decode encoded language instruction"""
  return bytes(inst[np.where(inst != 0)].tolist()).decode("utf-8") 

def preprocess_instruction(instruction_str, tokenizer, max_length):
    """Tokenizes and pads the instruction string using the T5 tokenizer and returns a PyTorch tensor."""
    encoded_instruction = tokenizer(instruction_str, max_length=max_length, padding='max_length', truncation=True, return_tensors='pt')
    return encoded_instruction['input_ids'].squeeze(0)  # Return a 1D tensor

def save_episode(episode, split, episode_id, output_dir, tokenizer, max_instruction_length, resize_resolution, t5_encoder):
    """Saves the frames, metadata (as .pt), and processed instruction (as .pt)."""
    episode_dir = os.path.join(output_dir, split, episode_id.decode())
    os.makedirs(episode_dir, exist_ok=True)

    # Directly convert the 'steps' dataset to a list of NumPy arrays
    steps_dataset = episode['steps']
    # steps = [step.numpy() for step in tf.data.Dataset.from_tensor_slices(steps_dataset).as_numpy_iterator()]
    steps = steps_dataset.as_numpy_iterator()

    for i, step in tqdm(enumerate(steps), total=len(steps_dataset), desc=f"Saving data for {episode_id.decode()}"):
        # Save image
        img = Image.fromarray(step['observation']['rgb'])
        if resize_resolution is not None:
            img = img.resize(resize_resolution)
        img_path = os.path.join(episode_dir, f'frame_{i:04d}.png')
        img.save(img_path)

        # Save metadata as .pt
        metadata = {
            'action': torch.tensor(step['action'], dtype=torch.float32),
            'effector_target_translation': torch.tensor(step['observation']['effector_target_translation'], dtype=torch.float32),
            'effector_translation': torch.tensor(step['observation']['effector_translation'], dtype=torch.float32),
            'reward': torch.tensor(step['reward'], dtype=torch.float32),
            'is_first': torch.tensor(step['is_first'], dtype=torch.bool),
            'is_last': torch.tensor(step['is_last'], dtype=torch.bool),
            'is_terminal': torch.tensor(step['is_terminal'], dtype=torch.bool)
        }
        metadata_path = os.path.join(episode_dir, f'metadata_{i:04d}.pt')
        torch.save(metadata, metadata_path)

        if i == 0:
            instruction_bytes = step['observation']['instruction']
            instruction_str = decode_inst(instruction_bytes)
            processed_instruction = preprocess_instruction(instruction_str, tokenizer, max_instruction_length)
            instruction_embedding = t5_encoder(processed_instruction.unsqueeze(0)).last_hidden_state.squeeze(0)
            instruction_data = {'raw_instruction': instruction_str, 'processed_instruction': processed_instruction, 'instruction_embedding': instruction_embedding}
            instruction_path = os.path.join(episode_dir, 'instruction.pt')
            torch.save(instruction_data, instruction_path)

def process_and_split(dataset, output_dir, tokenizer, max_instruction_length, resize_resolution, valid_ratio, test_ratio, random_seed, t5_encoder):
    """Processes the dataset and splits it into train, valid, and test sets."""
    os.makedirs(os.path.join(output_dir, 'train'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'valid'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'test'), exist_ok=True)

    n_ep = 0
    all_episode_ids = []
    for i, ep in tqdm(enumerate(dataset)):
        ep_len = len(list(ep['steps'].as_numpy_iterator()))
        if ep_len < MIN_EP_LEN or ep_len > MAX_EP_LEN:
            continue
        ep_id = ep['episode_id'].numpy().decode()
        all_episode_ids.append(ep_id)
        n_ep += 1
        save_episode(ep, 'train', ep_id.encode(), output_dir, tokenizer, max_instruction_length, resize_resolution, t5_encoder)

    random.seed(random_seed)
    random.shuffle(all_episode_ids)

    num_episodes = len(all_episode_ids)
    num_valid = int(valid_ratio * num_episodes)
    num_test = int(test_ratio * num_episodes)

    valid_episode_ids = all_episode_ids[:num_valid]
    test_episode_ids = all_episode_ids[num_valid:num_valid + num_test]
    train_episode_ids = all_episode_ids[num_valid + num_test:]
    move_episodes(output_dir, valid_episode_ids, test_episode_ids)
        
def move_episodes(output_dir, valid_episode_ids, test_episode_ids):
    """Moves episode directories from the train directory to valid and test directories."""
    train_dir = os.path.join(output_dir, 'train')
    valid_dir = os.path.join(output_dir, 'valid')
    test_dir = os.path.join(output_dir, 'test')

    os.makedirs(valid_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    print("Moving episodes to validation directory:")
    for episode_id in tqdm(valid_episode_ids):
        src_path = os.path.join(train_dir, episode_id)
        dst_path = os.path.join(valid_dir, episode_id)
        if os.path.exists(src_path):
            shutil.move(src_path, dst_path)
        else:
            print(f"Warning: Episode ID '{episode_id}' not found in the train directory.")

    print("\nMoving episodes to test directory:")
    for episode_id in tqdm(test_episode_ids):
        src_path = os.path.join(train_dir, episode_id)
        dst_path = os.path.join(test_dir, episode_id)
        if os.path.exists(src_path):
            shutil.move(src_path, dst_path)
        else:
            print(f"Warning: Episode ID '{episode_id}' not found in the train directory.")

    print("\nFinished moving episodes.")

if __name__ == '__main__':
    # Create the output directory if it doesn't exist
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Load the T5 tokenizer and encoder
    tokenizer = T5Tokenizer.from_pretrained(T5_MODEL_NAME)
    t5_encoder = T5EncoderModel.from_pretrained(T5_MODEL_NAME)
    t5_encoder.eval()  # Set to evaluation mode
    
    # Define dataset_path
    dataset_path = os.path.join(dataset_directories[DATASET_NAME], DATASET_VERSION)


    # Load the dataset
    builder = tfds.builder_from_directory(dataset_path)
    # builder.download_and_prepare(download_dir='/data/lang_table')
    try:
        full_dataset = builder.as_dataset(split='train')
    except tf.errors.InvalidArgumentError:
        print(f"Warning: 'train' split not found. Trying to load the entire dataset without a specific split.")
        full_dataset = builder.as_dataset()

    # Process and split the dataset
    process_and_split(full_dataset, OUTPUT_DIR, tokenizer, MAX_INSTRUCTION_LENGTH, RESIZE_RESOLUTION, VALID_RATIO, TEST_RATIO, RANDOM_SEED, t5_encoder)

    print("Preprocessing and splitting complete!")
    print(f"Data saved to: {OUTPUT_DIR}")