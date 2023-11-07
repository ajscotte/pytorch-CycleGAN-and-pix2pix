# import flwr as fl

# # fl.server.start_server(
# #     server_address="0.0.0.0:8080",
# #     config=fl.server.ServerConfig(num_rounds=3),
# #     strategy=fl.server.strategy.FedAvg(min_available_clients=1), 
# # )

# # fl.server.start_server("localhost:8080", config={"num_rounds": 3})


    
# fl.server.start_server(
#     server_address="0.0.0.0:8080",
#     config={
#         "num_rounds": 3,
#         "desired_number_of_clients": 1,  # Set to 1 to allow only one client at a time
#     },
#     strategy=fl.server.strategy.FedAvg(min_available_clients=1), 
# )

import asyncio
import flwr as fl

class OneClientAtATimeStrategy(fl.server.strategy.FedAvg):
    def __init__(self):
        super().__init__()
        self.active_client = False

    async def get_parameters(self) -> fl.common.Weights:
        while self.active_client:
            await asyncio.sleep(1)
        self.active_client = True
        return await super().get_parameters()

    async def fit(self, parameters: fl.common.Weights) -> fl.common.Weights:
        updated_parameters = await super().fit(parameters)
        self.active_client = False
        return updated_parameters

# Start Flower server
server = fl.server.Server(strategy=OneClientAtATimeStrategy())
fl.server.start_server("[::]:8080", server, config={"num_rounds": 3})