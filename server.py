import flwr as fl
import models.flowerfed_pix2pix_model as model
from models import create_model
from options.train_options import TrainOptions

opt = TrainOptions().parse()

#change this to load a model from a point in memory if you want to use a past model
net = create_model(opt)
net.setup(opt)
    
generator, discriminator = net.state_dict()
g = [val.cpu().numpy() for _, val in generator.items()]
d = [val.cpu().numpy() for _, val in discriminator.items()]
    
model_weights = g + d


fl.server.start_server(
    server_address="0.0.0.0:8080",
    config=fl.server.ServerConfig(num_rounds=1),
    strategy=fl.server.strategy.FedAvg(min_available_clients=1, min_evaluate_clients=1, min_fit_clients=1, initial_parameters=fl.common.ndarrays_to_parameters(model_weights)), 
)

# fl.server.start_server("localhost:8080", config={"num_rounds": 3})