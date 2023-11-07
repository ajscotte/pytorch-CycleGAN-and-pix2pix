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
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#Able to create model and everything else here just need to modify to allow access to gan and discr
# opt_test = TestOptions().parse()
# test_data = create_dataset(opt_test)
opt_train = TrainOptions().parse()
# train_data = create_dataset(opt_train)


net = create_model(opt_train)
print("model created")
print("setup")
net.setup(opt_train)   
# net.to(DEVICE)

# Define Flower client
class FlowerClient(fl.client.NumPyClient):
  #probably have to add bot D and G parameters
  #this doesn't work since seperate models
  
  #todo change the model to work with both models at once either in tuple or dict form
  
  def get_parameters(self, config):
    
    print("get1")
    generator, discriminator = net.state_dict()
    print("get2")
    g = [val.cpu().numpy() for _, val in generator.items()]
    d = [val.cpu().numpy() for _, val in discriminator.items()]
    
    model_weights = g + d
    print("get done")
    return model_weights

  def set_parameters(self, parameters):
    
    print("set1")
    generator, discriminator = net.state_dict()
    print("set2")
    len_gparam = len([val.cpu().numpy() for _, val in generator.items()])
    len_dparam = len([val.cpu().numpy() for _, val in discriminator.items()])
    
    params_dict = zip(generator.keys(), parameters[:len_gparam])
    gstate_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
    
    params_dict = zip(discriminator.keys(), parameters[len_gparam:len_dparam+len_gparam])
    dstate_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
    
    # params_dict = zip(self.g_ema.state_dict().keys(), parameters[-len_emaparam:])
    # g_emastate_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
    print("set load")
    net.load_state_dict(gstate_dict, dstate_dict)
    print("set done")
    
    # self.generator.load_state_dict(gstate_dict, strict=False)
    # self.discriminator.load_state_dict(dstate_dict, strict=False)
    
    # self.g_ema.load_state_dict(g_emastate_dict, strict=False)

  def fit(self, parameters, config):
    print("fit1")
    self.set_parameters(parameters)
    print("fit2")
    size = train(net, opt_train)
    print("train_done")
    # return self.get_parameters(config={}), len(trainloader.dataset), {}
    return self.get_parameters(config={}), size, {}

  def evaluate(self, parameters, config):
    print("eval1")
    self.set_parameters(parameters)
    print("eval2")
    # loss, accuracy = test(net, test_data, opt_test)
    print("eval3")
    # return float(loss), len(testloader.dataset), {"accuracy": float(accuracy)}
    return float(0), 0, {"accuracy": float(0)}

# Start Flower client
fl.client.start_numpy_client(server_address="127.0.0.1:8080", client=FlowerClient())

# train(net, train_data, opt_train)