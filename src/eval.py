import math
import torch
from transformers import BertForMaskedLM

def eval(eval_dataloader, model, epoch, device):
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    losses = []
    model.eval()
    model.to(device)
    for batch in eval_dataloader:
        batch.data = {k:v.to(device) for k,v in batch.data.items()}
        with torch.no_grad():
            outputs = model(**batch)
            print("end1")
        loss = outputs.loss
        loss = loss.unsqueeze(0)
        losses.append(loss)
        print("end2")
    losses = torch.cat(losses)
    losses_avg = torch.mean(losses)
    perplexity = math.exp(losses_avg)
    print('eval {0}: loss:{1}  perplexity:{2}'.format(epoch, losses_avg.item(), perplexity))
    return losses_avg
     
def load_lm(path):
    print('model from pretrained')
    path = './checkpoint/epoch_'+str(path)
    model = BertForMaskedLM.from_pretrained(path)
    print("end3")
    return model