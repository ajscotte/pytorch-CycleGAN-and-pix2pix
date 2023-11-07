import flwr as fl 
import torch

from collections import OrderedDict
import models.flowerfed_pix2pix_model as model
from train_fed import train 
from test_fed import test
from options.train_options import TrainOptions
from options.test_options import TestOptions
from data import create_dataset
from models import create_model



#maybe get rid of this DEVICE
# DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# net

#todo: move parse over here so only need in one file
# trainloader, testloader = load_data()
# net = model()

#Able to create model and everything else here just need to modify to allow access to gan and discr
opt_train = TrainOptions().parse()
train_data = create_dataset(opt_train)
opt_test = TestOptions().parse()
test_data = create_dataset(opt_test)

net = create_model(opt_train)
print("model created")

# Define Flower client
class FlowerClient(fl.client.NumPyClient):
  #probably have to add bot D and G parameters
  #this doesn't work since seperate models
  
  #todo change the model to work with both models at once either in tuple or dict form
  
  def get_parameters(self, config):
    
    #todo: create a state_dict for each model in the pix2pix model class
    # 
    # return [val.cpu().numpy() for _, val in net.state_dict().items()]
    # generator_state_dict = self.generator.state_dict()
    # discriminator_state_dict = self.discriminator.state_dict()

    #     # Create a dictionary to store the parameters
    # parameters_dict = {
    #       'generator_state_dict': generator_state_dict,
    #       'discriminator_state_dict': discriminator_state_dict,
    #       'config': config
    # }

    # return parameters_dict
    
    generator, discriminator = net.state_dict()
    g = [val.cpu().numpy() for _, val in generator.items()]
    d = [val.cpu().numpy() for _, val in discriminator.items()]
    
    model_weights = g + d
    
    return model_weights

  def set_parameters(self, parameters):
    # params_dict = zip(net.state_dict().keys(), parameters)
    # state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
    # net.load_state_dict(state_dict, strict=True)
    
    generator, discriminator = net.state_dict()
    
    len_gparam = len([val.cpu().numpy() for _, val in generator.items()])
    len_dparam = len([val.cpu().numpy() for _, val in discriminator.items()])
    
    params_dict = zip(generator.keys(), parameters[:len_gparam])
    gstate_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
    
    params_dict = zip(discriminator.keys(), parameters[len_gparam:len_dparam+len_gparam])
    dstate_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
    
    # params_dict = zip(self.g_ema.state_dict().keys(), parameters[-len_emaparam:])
    # g_emastate_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
    
    net.load_state_dict(gstate_dict, dstate_dict)
    
    
    # self.generator.load_state_dict(gstate_dict, strict=False)
    # self.discriminator.load_state_dict(dstate_dict, strict=False)
    
    # self.g_ema.load_state_dict(g_emastate_dict, strict=False)

  def fit(self, parameters, config):
    self.set_parameters(parameters)
    train(net, train_data, opt_train)
    # return self.get_parameters(config={}), len(trainloader.dataset), {}
    return self.get_parameters(config={}), len(train_data), {}

  def evaluate(self, parameters, config):
    self.set_parameters(parameters)
    loss, accuracy = test(net, test_data, opt_test)
    # return float(loss), len(testloader.dataset), {"accuracy": float(accuracy)}
    return float(loss), len(test_data), {"accuracy": float(accuracy)}

# Start Flower client
fl.client.start_numpy_client(server_address="127.0.0.1:8080", client=FlowerClient())

# train(net, train_data, opt_train)