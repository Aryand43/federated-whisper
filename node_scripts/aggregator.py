import flwr as fl

def weighted_average(results):
    weights = [r.num_examples for _, r in results]
    parameters = [r.parameters for _, r in results]

    averaged = [
        sum(w * p_i for w, p_i in zip(weights, ps)) / sum(weights)
        for ps in zip(*parameters)
    ]
    return averaged

fl.server.start_server(
    server_address="0.0.0.0:8080",
    config=fl.server.ServerConfig(num_rounds=3),
    strategy=fl.server.strategy.FedAvg(evaluate_metrics_aggregation_fn=weighted_average)
)
