import os

from transformers import AutoTokenizer, DataCollatorWithPadding # split text; match sequence length
import pandas as pd
from datasets import Dataset # facilitates data manipulation
import torch
from torch.utils.data import DataLoader, RandomSampler # random sampling during data training
from torch.utils.data.distributed import DistributedSampler # distributed across GPUs
import copy # I want to use deep copy later, as I want the two objects to be independent
import random # generates random numbers

from transformers import AdamW, get_scheduler # optimiser; because I want my learning rate to be inconsistent
from tqdm.auto import tqdm # showing progress bar
import math # I'll use it to calculate loss
from BertForMasked import BertForMaskedLM
from src.config import Config


def get_dataset(config):
  tokenizer = AutoTokenizer.from_pretrained(config.initial_pretrain_tokenizer) # the output of tokenizer: 'input_ids', 'attention_mask'
  train_dataloader = data_process('../data/train2.txt', tokenizer, config)
  eval_dataloader = data_process('../data/testh.txt', tokenizer, config)
  return train_dataloader, eval_dataloader

def data_process(file_name, tokenizer, config):
  text = open_file(config.path_datasets + file_name)
  dataset = pd.DataFrame({'src':text, 'labels':text}) # two columns dataframe, for subsequent format transformation
  raw_datasets = Dataset.from_pandas(dataset) # facilitates subsequent data manipulation (e.g., tokenization)
  tokenized_datasets = raw_datasets.map(lambda x: tokenization(x, tokenizer, config), batched=True) # tokenize batch by batch -> improve efficiency; remove redundant column to free up resources
  data_collator = DataCollatorWithPadding(tokenizer=tokenizer) # use our default tokenizer, which converts text into computer-readable numbers; padding -> to the same length
  tokenized_datasets = tokenized_datasets.remove_columns(["src"])
  tokenized_datasets.set_format("torch") # for deep learning
  sampler = RandomSampler(tokenized_datasets) if not torch.cuda.device_count() > 1 else DistributedSampler(tokenized_datasets) # check if there are multiple GPUs available -> apply diff types of sampler accordingly
  dataloader = DataLoader(dataset=tokenized_datasets, sampler=sampler, batch_size=config.batch_size, collate_fn=data_collator) # splits the dataset into batches
  return dataloader

def tokenization(example, tokenizer, config):
  token = tokenizer(example['src'], truncation=True, max_length=config.sen_max_length, padding='max_length') # the output should contain 'input_ids', 'token_type_ids', and 'attention_mask'
  label = copy.deepcopy(token.data['input_ids']) # every token has its id, a sequence of tokens has a list of ids
  token.data['labels'] = label # if 'input_ids' is modified, 'labels' won't be influenced -> still keeping the numerical representation for each sequence
  token_mask = tokenizer.mask_token # 'mask_token', 'pad_token', 'cls_token', and 'sep_token' create four special types of token, which shall be used later
  token_pad = tokenizer.pad_token # serves for subsequent 'DataCollatorWithPadding'
  token_cls = tokenizer.cls_token # marks sequence start
  token_sep = tokenizer.sep_token # marks symbols (e.g., ',')
  ids_mask = tokenizer.convert_tokens_to_ids(token_mask) # masked tokens' ids
  token_ex = [token_mask, token_pad, token_cls, token_sep]
  ids_ex = [tokenizer.convert_tokens_to_ids(token) for token in token_ex] # special tokens' ids, including masked, as we'll use them later
  vocab = tokenizer.vocab # a dictionary - maps tokens and their ids
  vocab_int2str = {vocab:id for id, vocab in vocab.items()} # these are total tokens' ids
  if config.whole_words_mask: # if the WWM setting is true, which is our case
    mask_token = [op_mask_wwm(line, ids_mask, ids_ex, vocab_int2str) for line in token.data['input_ids']]
  token.data['input_ids'] = mask_token # we have 'labels' to represent ids
  return token

def op_mask_wwm(tokens, ids_mask, ids_ex, vocab_int2str):
  if len(tokens) <= 5: # we keep short sequences (subword units, each sequence is a sentence) from being masked, because otherwise it doesn't make sense
    return tokens
  line = tokens
  for index, token in enumerate(tokens): # while iterating over the token lists, index represent the position of each token, and token is the token value (ids)
    if token in ids_ex:
      line[index] = token # don't mask special tokens, they're not whole words
      continue
    if random.random() <= 0.15: # less than or equal to 15% of the time
      x = random.random()
      if x <= 0.80: # less than or equal to 80% of the time, mask pending
        token_str = vocab_int2str[token] # retrieve the string based on id
        if '##' not in token_str: # '##' represents prefix and suffix -> whole word -> mask pending
          line[index] = ids_mask # execute masking
          curr_index = index + 1 # search forward
          flag = True # easy to check - every time we mask a token, check its following token
          while flag:
            token_id = tokens[curr_index]
            token_id_str = vocab_int2str[token_id]
            if '##' not in token_id_str: # in case the subsequent subword is part of the previous whole word (e.g., 'play' + '##ing')
              flag = False
            else:
              line[curr_index] = ids_mask # otherwise they should be masked together, as they represent a whole word
            curr_index += 1 # continue checking
        if x > 0.80 and x < 0.9: # 10% of probability - replacing a token with a random token
          while True:
            token = random.randint(0, len(vocab_int2str)-1)
            if token not in ids_ex: # otherwise continue randomly assigning a token
              break
    return line


def open_file(path):
  text = []
  with open(path, 'r', encoding='utf8') as file: # utf8 - represents each character using 1 to 4 bytes; here, I use it to encode the file
    for line in file.readlines(): # readlines serves as a pointer
      line = line.strip() # strips whitespaces
      text.append(line)
  return text # computer-readable text data



def train(config):
  print('training start')
  device = torch.device(config.device) # as pre-defined, we're using CPU
  print('data loading')
  train_dl, eval_dl = get_dataset(config)
  print('model loading')
  model = BertForMaskedLM.from_pretrained(config.initial_pretrain_model)
  optimizer = AdamW(model.parameters(), lr=config.learning_rate)
  num_training_steps = config.num_epochs * len(train_dl)
  lr_scheduler = get_scheduler("linear", optimizer=optimizer, num_warmup_steps=config.num_warmup_steps, num_training_steps=num_training_steps)
  model.to(device) # load the model to CPU
  if torch.cuda.device_count() > 1:
    model = torch.nn.parallel.DistributedDataParallel(model, find_unused_parameters=True, broadcast_buffers=True) # check if any parameters are missed; apply non-parameter buffers to every device
  print('start to train')
  model.train()

  progress_bar = tqdm(range(num_training_steps))


  loss_best = math.inf # loss parameter is consistently updated


  for epoch in range(config.num_epochs):
    for index, batch in enumerate(train_dl):
      batch.data = {key:value.to(device) for key, value in batch.data.items()}
      print('11111111111')

      outputs = model(**batch) # use the model to predict masked tokens
      print('22222')

      loss = outputs.loss # model performance on data fitting
      print('3333333')

      loss = loss.mean() # prevent the model from overfitting
      print('44444444')

      loss.backward() # calculates loss gradients
      print('55555555')
      optimizer.step() # updates parameters accordingly, gradient descent optimisation
      print('666666666')

      lr_scheduler.step() # updates learning rate of the optimiser
      print('7777777777')

      optimizer.zero_grad() # eliminates current batch's gradients, prapares for the next batch
      print('888888888888888')

      progress_bar.update(1) # shows the number of batches processed
      if index % 500 == 0:
        print('epoch:{0} iter:{1}/{2} loss:{3}'.format(epoch, index, len(train_dl), loss.item()))
    current_loss = eval(eval_dl, model, epoch, device)
    if current_loss < loss_best:
      loss_best = current_loss
      print('saving model')
      path = config.path_model_save + 'epoch_{}/'.format(epoch)
      if not os.path.exists(path):
        os.makedirs(path)
      model_save = model.module if torch.cuda.device_count() > 1 else model
      model_save.save_pretrained(path)

if __name__ == '__main__':
  config = Config()
  train(config)