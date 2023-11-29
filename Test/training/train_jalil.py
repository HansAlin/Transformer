import torch
import torch.nn as nn

from torch.utils.data import DataLoader, Dataset, random_split


from os.path import dirname, abspath, join
import sys
# Find code directory relative to our directory
THIS_DIR = dirname(__file__)
CODE_DIR = abspath(join(THIS_DIR, '..', ''))
sys.path.append(CODE_DIR)

type(sys.path)
for path in sys.path:
   print(path)

from config.config import get_config, getweights_file_path
from data.dataset import BilingualDataset #, causal_mask
from models.models_1 import build_transformer
from datasets import load_dataset
from torch.utils.tensorboard import SummaryWriter
from tokenizers import Tokenizer
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace

import tqdm as tqdm
import warnings

from pathlib import Path

def get_all_sentences(ds,lang):
  for item in ds:
    yield item['translation'][lang] 

def get_or_build_tokenizer(config, ds,lang):
  # config['tokenizer_file'] = 'tokenizer_{}.json'
  tokenizer_path = Path(config['tokenizer_file'].format(lang))
  if not Path.exists(tokenizer_path):
    tokenizer = Tokenizer(WordLevelTrainer(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()
    trainer = WordLevelTrainer(special_tokens=["[UNK]", "[PAD]", "[SOS]", "[EOS]"], min_freq=2)

    tokenizer.train_from_iterator(get_all_sentences(ds,lang), trainer)  

    tokenizer.save(str(tokenizer_path))

  else:
    tokenizer = Tokenizer.from_file(str(tokenizer_path))  

def get_ds(config):
  ds_raw = load_dataset('opus_books', f'{config["lang_src"]}-{config["lang_tgt"]}', split='train')

  # Build tokenizer
  tokenizer_src = get_or_build_tokenizer(config, ds_raw, config['lang_src'])
  tokenizer_tgt = get_or_build_tokenizer(config, ds_raw, config['lang_tgt'])

  # Keep 90 % for training and 10 % for validation
  train_ds_size = int(0.9 * len(ds_raw))
  val_ds_size = len(ds_raw) - train_ds_size
  train_ds_raw, val_ds_raw = random_split(ds_raw, [train_ds_size, val_ds_size])

  train_ds    = BilingualDataset(train_ds_raw, tokenizer_src, tokenizer_tgt, config['lang_src'], config['lang_tgt'], config['seq_len']) 
  val_ds      = BilingualDataset(val_ds_raw, tokenizer_src, tokenizer_tgt, config['lang_src'], config['lang_tgt'], config['seq_len'])

  max_len_src = 0
  max_len_tgt = 0

  for item in ds_raw:
    src_ids = tokenizer_src.encode(item['translation'][config['lang_src']]).ids
    tgt_ids = tokenizer_tgt.encode(item['translation'][config['lang_tgt']]).ids
    max_len_src = max(max_len_src, len(src_ids))  
    max_len_tgt = max(max_len_tgt, len(tgt_ids))

  print(f"Max length of source language: {max_len_src}")
  print(f"Max length of target language: {max_len_tgt}")

  train_dataloader = DataLoader(train_ds, batch_size=config['batch_size'], shuffle=True)
  val_dataloader = DataLoader(val_ds, batch_size=1, shuffle=True)  # batch_size=1 for test sentence by sentence

  return train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt

def get_model(config, vocab_src_len, vocab_tgt_len):
  model = build_transformer(src_vocab_size = vocab_src_len, 
                            tgt_vocab_size = vocab_tgt_len,
                            src_seq_len = config['seq_len'],
                            tgt_seq_len = config['seq_len'],
                            d_model=config['d_model'],)
  return model

def train_model(config):
  # define device
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  print(f"Device: {device}")

  Path(config['model_folder']).mkdir(parents=True, exist_ok=True)

  train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt = get_ds(config)
  model = get_model(config, tokenizer_src.get_vocab_size(), tokenizer_tgt.get_vocab_size()).to(device)

  # Tensorborad
  writer = SummaryWriter(config['experiment_name'])

  optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'], eps=1e-09)

  initial_epoch = 0
  global_step = 0
  if config['preload']:
    model_filename = getweights_file_path(config, config['preload'])
    print(f"Loading model from {model_filename}")
    state = torch.load(model_filename)
    initial_epoch = state['epoch'] + 1
    optimizer.load_state_dict(state['optimizer_state_dict'])
    global_step = state['global_step']

  # label_smoothing=0.1 means that the model gets less sure about
  # the ground truth labels. Give some values to values that not are the highest
  loss_fn = nn.CrossEntropyLoss(ignore_index=tokenizer_src.token_to_id("[PAD]"), label_smoothing=0.1).to(device)

  for epoch in range(initial_epoch, config['num_epochs']):
    model.train()
    # What does this do
    batch_iterator = tqdm.tqdm(train_dataloader, desc=f"Processing epoch {epoch}")
    for batch in batch_iterator:

      encoder_input = batch['encoder_input'].to(device) # (B, seq_len)
      decoder_input = batch['decoder_input'].to(device) # (B, seq_len)
      encoder_mask = batch['encoder_mask'].to(device) # (B, 1, 1 ,seq_len) hide padding tokens
      decoder_mask = batch['decoder_mask'].to(device) # (B, 1, seq_len, seq_len) hide padding tokens and future tokens

      # Run teh tensor througe the transformer
      encoder_output = model.encoder(encoder_input, encoder_mask) # (B, seq_len, d_model)
      decoder_output = model.decoder(decoder_input, encoder_output, decoder_mask) # (B, seq_len, d_model)
      projection = model.projection(decoder_output) # (B, seq_len, tgt_vocab_size)

      label = batch['label'].to(device) # (B, seq_len)

      loss = loss_fn(projection.view(-1, tokenizer_tgt.get_vocab_size()), label.view(-1))
      batch_iterator.set_postfix({'loss': f"{loss.item():6.3f}"})

      # Log the loss
      writer.add_scalar('Train loss', loss.item(), global_step)
      writer.flush()

      # Backpropagation
      loss.backward()

      # Update the weights
      optimizer.step()
      optimizer.zero_grad()

      global_step += 1

    # Save the model
    model_filename = getweights_file_path(config, epoch)
    torch.save({
      'epoch': epoch,
      'model_state_dict': model.state_dict(),
      'optimizer_state_dict': optimizer.state_dict(),
      'global_step': global_step,
      'loss': loss,
      }, model_filename)

if __name__ == "__main__":
  warnings.filterwarnings("ignore")
  config = get_config()
  train_model(config)      



