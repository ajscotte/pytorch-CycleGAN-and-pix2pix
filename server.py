import flwr as fl

fl.server.start_server(
    server_address="0.0.0.0:8080",
    config=fl.server.ServerConfig(num_rounds=1),
    strategy=fl.server.strategy.FedAvg(min_available_clients=1, min_evaluate_clients=1, min_fit_clients=1), 
)

# fl.server.start_server("localhost:8080", config={"num_rounds": 3})