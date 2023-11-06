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
opt_train = TrainOptions().parse()
train_data = create_dataset(opt_train)
opt_test = TestOptions().parse()
test_data = create_dataset(opt_test)

net = create_model(opt_train)

# Define Flower client
class FlowerClient(fl.client.NumPyClient):
  #probably have to add bot D and G parameters
  def get_parameters(self, config):
    return [val.cpu().numpy() for _, val in net.state_dict().items()]

  def set_parameters(self, parameters):
    params_dict = zip(net.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
    net.load_state_dict(state_dict, strict=True)

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