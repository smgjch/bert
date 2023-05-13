class Config(object): # store hyperparameters, every object belongs to this class has the same attributes
  def __init__(self):
    self.device = 'cpu'
    self.whole_words_mask = True
    self.num_epochs = 10 # my device doesn't allow me to run 100 epochs...
    self.batch_size = 30
    self.learning_rate = 3e-4
    self.num_warmup_steps = 0.1 # initially, model weights are randomised -> unstable model -> smaller learning rate for the first few steps than the setting
    self.sen_max_length = 128
    self.padding = True
    self.initial_pretrain_model = 'bert-base-uncased'
    self.initial_pretrain_tokenizer = 'bert-base-uncased'
    self.path_model_save = './checkpoint/' # save model weights and configuration
    self.path_datasets = ''