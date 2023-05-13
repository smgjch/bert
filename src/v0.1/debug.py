from Config import Config
from eval import load_lm, eval
from transformers import AutoTokenizer
from process import data_process


model = load_lm(0)
config = Config()

tokenizer = AutoTokenizer.from_pretrained(config.initial_pretrain_tokenizer)     

eval_dataloader = data_process('testq.txt', tokenizer, config)


print (eval(eval_dataloader,model,0,"cuda"))
print (11111111)