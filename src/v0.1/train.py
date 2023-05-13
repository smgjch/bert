import math
import os
import torch
from transformers import BertForMaskedLM, AdamW, get_scheduler
from Config import Config
from process import get_dataset
from tqdm.auto import tqdm

def train(config):
    print('training start')
    device = torch.device(config.device)
    if torch.cuda.device_count() > 1:
        torch.distributed.init_process_group(backend='nccl',init_method=config.init_method,rank=0,world_size=config.world_size)
        torch.distributed.barrier()
    print('data loading')
    train_dl, eval_dl = get_dataset(config)
    print('model loading')
    model = BertForMaskedLM.from_pretrained(config.initial_pretrain_model)
    optimizer = AdamW(model.parameters(), lr=config.learning_rate)
    num_training_steps = config.num_epochs * len(train_dl)
    lr_scheduler = get_scheduler("linear",optimizer=optimizer,num_warmup_steps=config.num_warmup_steps,num_training_steps=num_training_steps)
    model.to(device)
    if torch.cuda.device_count() > 1:
        model = torch.nn.parallel.DistributedDataParallel(model,  find_unused_parameters=True, broadcast_buffers=True)
    print('start to train')
    model.train()
    print("end")
    progress_bar = tqdm(range(num_training_steps))
    loss_best = math.inf
    print("end2")
    for epoch in range(config.num_epochs):
        for i, batch in enumerate(train_dl):
            batch.data = {k:v.to(device) for k,v in batch.data.items()}
            outputs = model(**batch)
            loss = outputs.loss
            loss = loss.mean()
            loss.backward()
            optimizer.step()
            
            lr_scheduler.step()
            optimizer.zero_grad()
            progress_bar.update(1)
            if i % 500 == 0:
                print('epoch:{0}  iter:{1}/{2}  loss:{3}'.format(epoch, i, len(train_dl), loss.item()))
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