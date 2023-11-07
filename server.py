import flwr as fl

fl.server.start_server(
    server_address="0.0.0.0:8080",
    config=fl.server.ServerConfig(num_rounds=3),
    strategy=fl.server.strategy.FedAvg(), 
)

# fl.server.start_server("localhost:8080", config={"num_rounds": 3})