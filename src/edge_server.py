import copy
from utils import calculate_model_drift

class EdgeServer:
    def __init__(self, edge_id, clients):
        self.edge_id = edge_id
        self.clients = clients
        self.local_model = None
        self.disabled_clients = set()

    def receive_global_model(self, global_weights):
        self.local_model = copy.deepcopy(global_weights)
        for client in self.clients:
            client.set_model_weights(self.local_model)

    def train_and_aggregate(self, threshold_percent, prev_drift):
        local_weights = []
        client_accuracies = []

        for client in self.clients:
            if client.client_id in self.disabled_clients:
                continue
            print(f"  🔄 Client {client.client_id} training...")
            client.train()
            acc = client.evaluate()
            client_accuracies.append((client.client_id, acc))
            local_weights.append((client.client_id, client.get_model_weights()))

        if not local_weights:
            return None, client_accuracies

        # FedAvg
        new_model = copy.deepcopy(local_weights[0][1])
        for key in new_model.keys():
            for i in range(1, len(local_weights)):
                new_model[key] += local_weights[i][1][key]
            new_model[key] /= len(local_weights)

        # Drift check
        if prev_drift is not None:
            drift = calculate_model_drift(self.local_model, new_model)
            threshold = prev_drift * threshold_percent
            if drift < prev_drift - threshold or drift > prev_drift + threshold:
                print(f"  ⚠️ Drift {drift:.4f} outside ±{threshold_percent * 100:.1f}% threshold of {prev_drift:.4f}")
                max_drift, outlier = -1, None
                for client_id, weights in local_weights:
                    d = calculate_model_drift(new_model, weights)
                    if d > max_drift:
                        max_drift = d
                        outlier = client_id
                if outlier is not None:
                    print(f"  ❌ Disabling Client {outlier} for this round due to high drift.")
                    self.disabled_clients.add(outlier)

        return new_model, client_accuracies
