import flwr as fl

# fl.server.start_server(
#     server_address="0.0.0.0:8080",
#     config=fl.server.ServerConfig(num_rounds=3),
#     strategy=fl.server.strategy.FedAvg(min_available_clients=1), 
# )

# fl.server.start_server("localhost:8080", config={"num_rounds": 3})


    
fl.server.start_server(
    server_address="0.0.0.0:8080",
    config={
        "num_rounds": 3,
        "desired_number_of_clients": 1,  # Set to 1 to allow only one client at a time
    },
    strategy=fl.server.strategy.FedAvg(min_available_clients=1), 
)