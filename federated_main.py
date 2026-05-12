# federated_main.py
# without edge servers but with basic deduction logic 
'''
import os
from client import Client
from server import Server
from model import LogisticRegressionModel
from utils import calculate_model_drift
import torch

NUM_CLIENTS = 5
NUM_ROUNDS = 10
INPUT_SIZE = 9  # Adjust based on your CSV features
THRESHOLD_PERCENT = 0.
EPOCHS = 3
LR = 0.002
BATCH_SIZE = 32
ROUNDS = 10


CLIENT_DATA_DIR = "client_data"
clients = [
    Client(i, os.path.join(CLIENT_DATA_DIR, f"client_F0{i+1}.csv"), INPUT_SIZE)
    for i in range(NUM_CLIENTS)
]

server_model = LogisticRegressionModel(INPUT_SIZE, num_classes=3)
server = Server(INPUT_SIZE)
server.global_model = server_model.state_dict()

disabled_clients = set()
prev_drift = None  # For tracking drift from the previous round

for r in range(NUM_ROUNDS):
    print(f"\n🌐 Round {r + 1}")

    active_clients = [client for client in clients if client.client_id not in disabled_clients]

    local_weights = []
    client_accuracies = []

    for client in clients:  # All clients receive global model
        client.set_model_weights(server.global_model)

    for client in active_clients:
        print(f"  🔄 Client {client.client_id} training...")
        client.train()
        acc = client.evaluate()
        client_accuracies.append((client.client_id, acc))
        local_weights.append(client.get_model_weights())

    if not local_weights:
        print("  ❌ No active clients left to aggregate.")
        break

    # FedAvg
    new_global_weights = local_weights[0]
    for key in new_global_weights.keys():
        for i in range(1, len(local_weights)):
            new_global_weights[key] += local_weights[i][key]
        new_global_weights[key] /= len(local_weights)

    drift = calculate_model_drift(server.global_model, new_global_weights)
    print(f"  ✅ FedAvg completed. Drift = {drift:.4f}")
    print("  📈 Client Accuracies:", client_accuracies)

    if prev_drift is not None:
        threshold = prev_drift * THRESHOLD_PERCENT
        if drift < prev_drift - threshold or drift > prev_drift + threshold:
            print(f"  ⚠️ Drift {drift:.4f} outside ±10% threshold of {prev_drift:.4f}")

            # Find most deviating client from new global
            max_drift = -1
            outlier_client = None
            for client in active_clients:
                d = calculate_model_drift(new_global_weights, client.get_model_weights())
                if d > max_drift:
                    max_drift = d
                    outlier_client = client

            if outlier_client:
                print(f"  ❌ Disabling Client {outlier_client.client_id} for this round due to high drift.")
                disabled_clients.add(outlier_client.client_id)

    # Save drift and update global
    prev_drift = drift
    server.update_global_model(new_global_weights)

    # Re-enable all clients for next round (if temporary disabling is desired)
    disabled_clients.clear()
'''

# with edge server and basic deduction logic
'''
import os
from client import Client
from server import Server
from edge_server import EdgeServer
from model import LogisticRegressionModel
import torch

from utils import calculate_model_drift

NUM_CLIENTS = 5
NUM_ROUNDS = 10
INPUT_SIZE = 9
THRESHOLD_PERCENT = 0.2  # 10%
CLIENT_DATA_DIR = "client_data"

# Assign clients to two edge servers
clients = [Client(i, os.path.join(CLIENT_DATA_DIR, f"client_F0{i+1}.csv"), INPUT_SIZE) for i in range(NUM_CLIENTS)]
edge1 = EdgeServer(edge_id=1, clients=clients[:3])  # Clients 0,1,2
edge2 = EdgeServer(edge_id=2, clients=clients[3:])  # Clients 3,4

server = Server()
initial_model = LogisticRegressionModel(INPUT_SIZE, num_classes=3).state_dict()
server.update_global_model(initial_model)

prev_drift = None

for r in range(NUM_ROUNDS):
    print(f"\n🌐 Round {r + 1}")

    # Send global model to edges
    edge1.receive_global_model(server.global_model)
    edge2.receive_global_model(server.global_model)

    # Edge training
    edge_models = []
    all_client_accuracies = []

    for edge in [edge1, edge2]:
        new_model, accuracies = edge.train_and_aggregate(THRESHOLD_PERCENT, prev_drift)
        if new_model is not None:
            edge_models.append(new_model)
        all_client_accuracies.extend(accuracies)

    if not edge_models:
        print("❌ No edge models received, aborting training.")
        break

    # Global aggregation
    new_global = server.aggregate_from_edges(edge_models)
    drift = calculate_model_drift(server.global_model, new_global)

    print(f"  ✅ Global Aggregation Done. Drift = {drift:.4f}")
    print("  📈 Client Accuracies:", all_client_accuracies)

    prev_drift = drift
    server.update_global_model(new_global)
'''

# without edge servers and  deduction logic and without winsorisation
# federated_main.py
'''
import os
from client import Client
from server import Server
from model import LogisticRegressionModel
import torch

NUM_CLIENTS = 5
NUM_ROUNDS = 10
INPUT_SIZE = 9  # Number of features in your dataset
EPOCHS = 3

CLIENT_DATA_DIR = "client_data"

# Initialize all clients
clients = [
    Client(i, os.path.join(CLIENT_DATA_DIR, f"client_F0{i+1}.csv"), INPUT_SIZE)
    for i in range(NUM_CLIENTS)
]

# Initialize server
server_model = LogisticRegressionModel(INPUT_SIZE, num_classes=3)
server = Server(INPUT_SIZE)
server.global_model = server_model.state_dict()

for r in range(NUM_ROUNDS):
    print(f"\n🌐 Round {r + 1}")

    local_weights = []
    client_accuracies = []

    # Send global model to all clients
    for client in clients:
        client.set_model_weights(server.global_model)

    # Each client trains on local data
    for client in clients:
        print(f"  🔄 Client {client.client_id} training...")
        client.train(epochs=EPOCHS)
        acc = client.evaluate()
        client_accuracies.append((client.client_id, acc))
        local_weights.append(client.get_model_weights())

    # Federated Averaging (FedAvg)
    new_global_weights = server.aggregate(local_weights)
    server.update_global_model(new_global_weights)

    print("  ✅ Global Aggregation Done.")
    print("  📈 Client Accuracies:", client_accuracies)
'''

# without edge server and deduction logic and with winsorisation
# federated_main.py
'''

import os
from client import Client
from server import Server
from model import LogisticRegressionModel
from utils import calculate_model_drift

NUM_CLIENTS = 5
INPUT_SIZE = 9
EPOCHS = 3
ROUNDS = 15 # Increased rounds
WINSORIZE_AT_ROUND = 5  # Apply Winsorization after this

CLIENT_DATA_DIR = "client_data"
clients = [
    Client(i, os.path.join(CLIENT_DATA_DIR, f"client_F0{i+1}.csv"), INPUT_SIZE)
    for i in range(NUM_CLIENTS)
]

server_model = LogisticRegressionModel(INPUT_SIZE, num_classes=3)
server = Server(INPUT_SIZE)
server.global_model = server_model.state_dict()

for r in range(ROUNDS):
    print(f"\n🌐 Round {r + 1}")

    if r == WINSORIZE_AT_ROUND:
        print("🔧 Applying Winsorization to all clients...")
        for client in clients:
            client.winsorize_data()

    for client in clients:
        client.set_model_weights(server.global_model)

    local_weights = []
    client_accuracies = []

    for client in clients:
        print(f"  🔄 Client {client.client_id} training...")
        client.train(epochs=EPOCHS)
        acc = client.evaluate()
        client_accuracies.append((client.client_id, acc))
        local_weights.append(client.get_model_weights())

    # FedAvg
    new_global_weights = local_weights[0]
    for key in new_global_weights.keys():
        for i in range(1, len(local_weights)):
            new_global_weights[key] += local_weights[i][key]
        new_global_weights[key] /= len(local_weights)

    print(f"  ✅ Global Aggregation Done.")
    print("  📈 Client Accuracies:", client_accuracies)

    server.update_global_model(new_global_weights)

'''
# better logging and saving results
'''
import os
import pandas as pd
import torch
from client import Client
from server import Server
from model import LogisticRegressionModel

NUM_CLIENTS = 5
INPUT_SIZE = 9
EPOCHS = 3
ROUNDS = 15
WINSORIZE_AT_ROUND = 5
CLIENT_DATA_DIR = "client_data"

# Initialize clients
clients = [
    Client(i, os.path.join(CLIENT_DATA_DIR, f"client_F0{i+1}.csv"), INPUT_SIZE)
    for i in range(NUM_CLIENTS)
]

server_model = LogisticRegressionModel(INPUT_SIZE, num_classes=3)
server = Server(INPUT_SIZE)
server.global_model = server_model.state_dict()

# Storage for logging
log_data = []

for r in range(ROUNDS):
    print(f"\n🌐 Round {r + 1}")

    # Winsorisation trigger
    if r == WINSORIZE_AT_ROUND:
        print("🔧 Applying Winsorization to all clients...")
        for client in clients:
            client.winsorize_data()

    # Distribute global model to clients
    for client in clients:
        client.set_model_weights(server.global_model)

    local_weights = []
    client_accuracies = []

    # Train and evaluate clients
    for client in clients:
        print(f"  🔄 Client {client.client_id} training...")
        client.train(epochs=EPOCHS)
        acc = client.evaluate()
        client_accuracies.append(acc)
        local_weights.append(client.get_model_weights())

    # FedAvg aggregation
    new_global_weights = local_weights[0]
    for key in new_global_weights.keys():
        for i in range(1, len(local_weights)):
            new_global_weights[key] += local_weights[i][key]
        new_global_weights[key] /= len(local_weights)

    server.update_global_model(new_global_weights)

    # Evaluate global model accuracy on all client data combined
    total_correct, total_samples = 0, 0
    for client in clients:
        client.set_model_weights(server.global_model)
        with torch.no_grad():
            for batch_x, batch_y in client.data_loader:
                outputs = client.model(batch_x)
                _, predicted = torch.max(outputs.data, 1)
                total_samples += batch_y.size(0)
                total_correct += (predicted == batch_y).sum().item()

    global_acc = total_correct / total_samples

    # Log data
    log_entry = {
        "Round": r + 1,
        "Global_Accuracy": global_acc
    }
    for i, acc in enumerate(client_accuracies):
        log_entry[f"Client_{i}_Acc"] = acc
    log_data.append(log_entry)

    # Print summary
    print(f"  📊 Global Accuracy: {global_acc:.4f}")
    print("  📈 Client Accuracies:", [f"{acc:.3f}" for acc in client_accuracies])
    print("  ✅ Global Aggregation Done.")

# Save logs to CSV
df_log = pd.DataFrame(log_data)
df_log.to_csv("federated_training_log.csv", index=False)
print("\n📁 Training log saved to federated_training_log.csv")
'''

### final code with better logging and saving results and plots generated - without edge servers, no deduction logic, and with winsorisation
# federated_main.py
'''
import os
import pandas as pd
import torch
import matplotlib.pyplot as plt
from client import Client
from server import Server
from model import LogisticRegressionModel

# ====== Config ======
NUM_CLIENTS = 5
INPUT_SIZE = 9
EPOCHS = 3
ROUNDS = 15
WINSORIZE_AT_ROUND = 5
CLIENT_DATA_DIR = "client_data"
PLOT_DIR = "plots"
os.makedirs(PLOT_DIR, exist_ok=True)

# ====== Init Clients & Server ======
clients = [
    Client(i, os.path.join(CLIENT_DATA_DIR, f"client_F0{i+1}.csv"), INPUT_SIZE)
    for i in range(NUM_CLIENTS)
]

server_model = LogisticRegressionModel(INPUT_SIZE, num_classes=3)
server = Server(INPUT_SIZE)
server.global_model = server_model.state_dict()

# ====== Logging Storage ======
log_data = []

# ====== Federated Rounds ======
for r in range(ROUNDS):
    print(f"\n🌐 Round {r + 1}")
    print("=" * 40)

    # Winsorisation trigger
    if r == WINSORIZE_AT_ROUND:
        print("🔧 Applying Winsorization to all clients...")
        for client in clients:
            client.winsorize_data()

    # Distribute global model
    for client in clients:
        client.set_model_weights(server.global_model)

    local_weights = []
    client_accuracies = []

    # Train + Evaluate Clients
    for client in clients:
        print(f"  🔄 Client {client.client_id} training...")
        client.train(epochs=EPOCHS)
        acc = client.evaluate()
        client_accuracies.append(acc)
        local_weights.append(client.get_model_weights())

    # FedAvg
    new_global_weights = local_weights[0]
    for key in new_global_weights.keys():
        for i in range(1, len(local_weights)):
            new_global_weights[key] += local_weights[i][key]
        new_global_weights[key] /= len(local_weights)
    server.update_global_model(new_global_weights)

    # Evaluate Global Model
    total_correct, total_samples = 0, 0
    for client in clients:
        client.set_model_weights(server.global_model)
        with torch.no_grad():
            for batch_x, batch_y in client.data_loader:
                outputs = client.model(batch_x)
                _, predicted = torch.max(outputs.data, 1)
                total_samples += batch_y.size(0)
                total_correct += (predicted == batch_y).sum().item()
    global_acc = total_correct / total_samples

    # Log Entry
    log_entry = {"Round": r + 1, "Global_Accuracy": global_acc}
    for i, acc in enumerate(client_accuracies):
        log_entry[f"Client_{i}_Acc"] = acc
    log_data.append(log_entry)

    # Print Summary in Table Style
    print(f"  📊 Global Accuracy : {global_acc:.4f}")
    for i, acc in enumerate(client_accuracies):
        print(f"     - Client {i} Accuracy : {acc:.4f}")
    print("  ✅ Global Aggregation Done.")
    print("=" * 40)

# ====== Save CSV ======
df_log = pd.DataFrame(log_data)
df_log.to_csv("federated_training_log.csv", index=False)
print("\n📁 Training log saved to federated_training_log.csv")

# ====== Generate Plots ======
rounds = df_log["Round"]
global_accs = df_log["Global_Accuracy"]

# Mark Winsorization round
winsor_round = WINSORIZE_AT_ROUND + 1  # +1 because rounds are 1-indexed in log

# --- Plot Global Accuracy ---
plt.figure(figsize=(8, 5))
plt.plot(rounds, global_accs, marker='o', color='b', label="Global Accuracy")
plt.axvline(x=winsor_round, color='r', linestyle='--', label='Winsorization Applied')
plt.xlabel("Round")
plt.ylabel("Accuracy")
plt.title("Global Accuracy Over Rounds")
plt.grid(True)
plt.legend()
plt.savefig(os.path.join(PLOT_DIR, "global_accuracy.png"))
plt.close()

# --- Plot Client Accuracies ---
plt.figure(figsize=(8, 5))
for i in range(NUM_CLIENTS):
    plt.plot(rounds, df_log[f"Client_{i}_Acc"], marker='o', label=f"Client {i}")
plt.axvline(x=winsor_round, color='r', linestyle='--', label='Winsorization Applied')
plt.xlabel("Round")
plt.ylabel("Accuracy")
plt.title("Client Accuracies Over Rounds")
plt.grid(True)
plt.legend()
plt.savefig(os.path.join(PLOT_DIR, "client_accuracies.png"))
plt.close()

print(f"📊 Plots saved in '{PLOT_DIR}' directory with Winsorization marker.")
'''
# federated_main.py
### final code with better logging and saving results and plots generated - without edge servers, no deduction logic, and with winsorisation
## final with randomisation and seed logic
'''
import os
import pandas as pd
import torch
import matplotlib.pyplot as plt
import numpy as np
import random
from client import Client
from server import Server
from model import LogisticRegressionModel

# ====== Reproducibility ======
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# ====== Config ======
NUM_CLIENTS = 5
INPUT_SIZE = 9
EPOCHS = 3
ROUNDS = 15
WINSORIZE_AT_ROUND = 5  # will apply at round 6 (1-indexed)
CLIENT_DATA_DIR = "client_data"
PLOT_DIR = "plots"
os.makedirs(PLOT_DIR, exist_ok=True)

# ====== Init Clients & Server ======
clients = [
    Client(i, os.path.join(CLIENT_DATA_DIR, f"client_F0{i+1}.csv"), INPUT_SIZE)
    for i in range(NUM_CLIENTS)
]

server_model = LogisticRegressionModel(INPUT_SIZE, num_classes=3)
server = Server(INPUT_SIZE)
server.global_model = server_model.state_dict()

# ====== Logging Storage ======
log_data = []

# ====== Federated Rounds ======
for r in range(ROUNDS):
    print(f"\n🌐 Round {r + 1}")
    print("=" * 40)

    # Winsorisation trigger
    if r == WINSORIZE_AT_ROUND:
        print("🔧 Applying Winsorization to all clients...")
        for client in clients:
            client.winsorize_data()

    # Distribute global model
    for client in clients:
        client.set_model_weights(server.global_model)

    local_weights = []
    client_accuracies = []

    # Train + Evaluate Clients
    for client in clients:
        print(f"  🔄 Client {client.client_id} training...")
        client.train(epochs=EPOCHS)
        acc = client.evaluate()
        client_accuracies.append(acc)
        local_weights.append(client.get_model_weights())

    # FedAvg aggregation
    new_global_weights = local_weights[0]
    for key in new_global_weights.keys():
        for i in range(1, len(local_weights)):
            new_global_weights[key] += local_weights[i][key]
        new_global_weights[key] /= len(local_weights)
    server.update_global_model(new_global_weights)

    # Evaluate Global Model
    total_correct, total_samples = 0, 0
    for client in clients:
        client.set_model_weights(server.global_model)
        with torch.no_grad():
            for batch_x, batch_y in client.data_loader:
                outputs = client.model(batch_x)
                _, predicted = torch.max(outputs.data, 1)
                total_samples += batch_y.size(0)
                total_correct += (predicted == batch_y).sum().item()
    global_acc = total_correct / total_samples

    # Log Entry
    log_entry = {"Round": r + 1, "Global_Accuracy": global_acc}
    for i, acc in enumerate(client_accuracies):
        log_entry[f"Client_{i}_Acc"] = acc
    log_data.append(log_entry)

    # Print Summary
    print(f"  📊 Global Accuracy : {global_acc:.4f}")
    for i, acc in enumerate(client_accuracies):
        print(f"     - Client {i} Accuracy : {acc:.4f}")
    print("  ✅ Global Aggregation Done.")
    print("=" * 40)

# ====== Save CSV ======
df_log = pd.DataFrame(log_data)
df_log.to_csv("federated_training_log.csv", index=False)
print("\n📁 Training log saved to federated_training_log.csv")

# ====== Generate Plots ======
rounds = df_log["Round"]
global_accs = df_log["Global_Accuracy"]

# Mark Winsorization round
winsor_round = WINSORIZE_AT_ROUND + 1

# --- Plot Global Accuracy ---
plt.figure(figsize=(8, 5))
plt.plot(rounds, global_accs, marker='o', color='b', label="Global Accuracy")
plt.axvline(x=winsor_round, color='r', linestyle='--', label='Winsorization Applied')
plt.xlabel("Round")
plt.ylabel("Accuracy")
plt.title("Global Accuracy Over Rounds")
plt.grid(True)
plt.legend()
plt.savefig(os.path.join(PLOT_DIR, "global_accuracy.png"))
plt.close()

# --- Plot Client Accuracies ---
plt.figure(figsize=(8, 5))
for i in range(NUM_CLIENTS):
    plt.plot(rounds, df_log[f"Client_{i}_Acc"], marker='o', label=f"Client {i}")
plt.axvline(x=winsor_round, color='r', linestyle='--', label='Winsorization Applied')
plt.xlabel("Round")
plt.ylabel("Accuracy")
plt.title("Client Accuracies Over Rounds")
plt.grid(True)
plt.legend()
plt.savefig(os.path.join(PLOT_DIR, "client_accuracies.png"))
plt.close()

print(f"📊 Plots saved in '{PLOT_DIR}' directory.")
'''

## with edge servers, deduction logic, switching and winsorisation 
'''
# federated_main.py
import os
import pandas as pd
import torch
import matplotlib.pyplot as plt
import numpy as np
import random
from client import Client
from server import Server
from model import LogisticRegressionModel

# ====== Reproducibility ======
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# ====== Config ======
NUM_CLIENTS = 5
INPUT_SIZE = 9
EPOCHS = 3
ROUNDS = 15
WINSORIZE_AT_ROUND = 5  # still applies at round 6 (1-indexed)
CLIENT_DATA_DIR = "client_data"
FINAL_PLOT_DIR = "plots_final"
THRESHOLD = 0.05  # 20% of global accuracy
os.makedirs(FINAL_PLOT_DIR, exist_ok=True)

# ====== Init Clients & Server ======
clients = [
    Client(i, os.path.join(CLIENT_DATA_DIR, f"client_F0{i+1}.csv"), INPUT_SIZE)
    for i in range(NUM_CLIENTS)
]

server_model = LogisticRegressionModel(INPUT_SIZE, num_classes=3)
server = Server(INPUT_SIZE)
server.global_model = server_model.state_dict()

# ====== Edge Server Class ======
class EdgeServer:
    def __init__(self, id, assigned_client_ids):
        self.id = id
        self.assigned_client_ids = assigned_client_ids

    def filter_and_aggregate(self, client_results, last_global_acc):
        included_weights = []
        included_ids = []
        excluded_ids = []

        for cid, acc, weights in client_results:
            if last_global_acc is not None:
                if abs(acc - last_global_acc) > THRESHOLD * last_global_acc:
                    excluded_ids.append(cid)
                    continue
            included_weights.append(weights)
            included_ids.append(cid)

        # Aggregate included clients using FedAvg
        if included_weights:
            agg_weights = included_weights[0].copy()
            for key in agg_weights.keys():
                for i in range(1, len(included_weights)):
                    agg_weights[key] += included_weights[i][key]
                agg_weights[key] /= len(included_weights)
        else:
            agg_weights = None

        return agg_weights, included_ids, excluded_ids

# ====== Create Edge Servers ======
edge_servers = [
    EdgeServer(1, [0, 1, 2]),
    EdgeServer(2, [3, 4])
]

# ====== Logging Storage ======
log_data = []
last_global_acc = None

# ====== Federated Rounds ======
for r in range(ROUNDS):
    print(f"\n🌐 Round {r + 1}")
    print("=" * 40)

    # Apply winsorisation trigger if scheduled
    if r == WINSORIZE_AT_ROUND:
        print("🔧 Applying Winsorization to all clients...")
        for client in clients:
            client.winsorize_data()

    # Distribute global model to all clients
    for client in clients:
        client.set_model_weights(server.global_model)

    # Train & evaluate each client
    client_results = []  # (client_id, accuracy, weights)
    for client in clients:
        print(f"  🔄 Client {client.client_id} training...")
        client.train(epochs=EPOCHS)
        acc = client.evaluate()
        client_results.append((client.client_id, acc, client.get_model_weights()))
        print(f"     - Client {client.client_id} Accuracy: {acc:.4f}")

    # Process via edge servers
    edge_aggregates = []
    all_excluded_ids = set()

    for edge in edge_servers:
        edge_client_results = [
            res for res in client_results if res[0] in edge.assigned_client_ids
        ]
        agg_weights, included_ids, excluded_ids = edge.filter_and_aggregate(
            edge_client_results, last_global_acc
        )
        if agg_weights is not None:
            edge_aggregates.append(agg_weights)
        all_excluded_ids.update(excluded_ids)

        print(f"  🛠 Edge Server {edge.id} Included: {included_ids} | Excluded: {excluded_ids}")

    # Apply winsorisation to excluded clients before they rejoin in future
    for cid in all_excluded_ids:
        print(f"  ⚠ Winsorizing Client {cid} (excluded this round)")
        clients[cid].winsorize_data()

    # Central aggregation from edge server aggregates
    if edge_aggregates:
        new_global_weights = edge_aggregates[0].copy()
        for key in new_global_weights.keys():
            for i in range(1, len(edge_aggregates)):
                new_global_weights[key] += edge_aggregates[i][key]
            new_global_weights[key] /= len(edge_aggregates)
        server.update_global_model(new_global_weights)

    # Evaluate Global Model
    total_correct, total_samples = 0, 0
    for client in clients:
        client.set_model_weights(server.global_model)
        with torch.no_grad():
            for batch_x, batch_y in client.data_loader:
                outputs = client.model(batch_x)
                _, predicted = torch.max(outputs.data, 1)
                total_samples += batch_y.size(0)
                total_correct += (predicted == batch_y).sum().item()
    global_acc = total_correct / total_samples
    last_global_acc = global_acc

    # Log results
    log_entry = {"Round": r + 1, "Global_Accuracy": global_acc}
    for cid, acc, _ in client_results:
        log_entry[f"Client_{cid}_Acc"] = acc
    log_data.append(log_entry)

    # Round summary
    print(f"  📊 Global Accuracy : {global_acc:.4f}")
    print("=" * 40)

# ====== Save CSV ======
df_log = pd.DataFrame(log_data)
csv_path = "federated_training_log_final.csv"
df_log.to_csv(csv_path, index=False)
print(f"\n📁 Training log saved to {csv_path}")

# ====== Generate Plots ======
rounds = df_log["Round"]
global_accs = df_log["Global_Accuracy"]

# Mark Winsorization round
winsor_round = WINSORIZE_AT_ROUND + 1

# --- Plot Global Accuracy ---
plt.figure(figsize=(8, 5))
plt.plot(rounds, global_accs, marker='o', color='b', label="Global Accuracy")
plt.axvline(x=winsor_round, color='r', linestyle='--', label='Winsorization Applied')
plt.xlabel("Round")
plt.ylabel("Accuracy")
plt.title("Global Accuracy Over Rounds")
plt.grid(True)
plt.legend()
plt.savefig(os.path.join(FINAL_PLOT_DIR, "global_accuracy_final.png"))
plt.close()

# --- Plot Client Accuracies ---
plt.figure(figsize=(8, 5))
for i in range(NUM_CLIENTS):
    plt.plot(rounds, df_log[f"Client_{i}_Acc"], marker='o', label=f"Client {i}")
plt.axvline(x=winsor_round, color='r', linestyle='--', label='Winsorization Applied')
plt.xlabel("Round")
plt.ylabel("Accuracy")
plt.title("Client Accuracies Over Rounds")
plt.grid(True)
plt.legend()
plt.savefig(os.path.join(FINAL_PLOT_DIR, "client_accuracies_final.png"))
plt.close()

print(f"📊 Plots saved in '{FINAL_PLOT_DIR}' directory.")
'''

## with edge servers, deduction logic, switching and winsorisation  - this final setup with uneven data

# federated_main.py
import os
import pandas as pd
import torch
import matplotlib.pyplot as plt
import numpy as np
import random
from client import Client
from server import Server
from model import LogisticRegressionModel

# ====== Reproducibility ======
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# ====== Config ======
NUM_CLIENTS = 5
INPUT_SIZE = 9
EPOCHS = 3
ROUNDS = 15
WINSORIZE_AT_ROUND = 5  # still applies at round 6 (1-indexed)
CLIENT_DATA_DIR = "client_data_uneven"  # changed
FINAL_PLOT_DIR = "plots_uneven"         # changed
THRESHOLD = 0.05  # 20% of global accuracy
os.makedirs(FINAL_PLOT_DIR, exist_ok=True)

# ====== Init Clients & Server ======
clients = [
    Client(i, os.path.join(CLIENT_DATA_DIR, f"client_uneven_F0{i+1}.csv"), INPUT_SIZE)
    for i in range(NUM_CLIENTS)
]

server_model = LogisticRegressionModel(INPUT_SIZE, num_classes=3)
server = Server(INPUT_SIZE)
server.global_model = server_model.state_dict()

# ====== Edge Server Class ======
class EdgeServer:
    def __init__(self, id, assigned_client_ids):
        self.id = id
        self.assigned_client_ids = assigned_client_ids

    def filter_and_aggregate(self, client_results, last_global_acc):
        included_weights = []
        included_ids = []
        excluded_ids = []

        for cid, acc, weights in client_results:
            if last_global_acc is not None:
                if abs(acc - last_global_acc) > THRESHOLD * last_global_acc:
                    excluded_ids.append(cid)
                    continue
            included_weights.append(weights)
            included_ids.append(cid)

        # Aggregate included clients using FedAvg
        if included_weights:
            agg_weights = included_weights[0].copy()
            for key in agg_weights.keys():
                for i in range(1, len(included_weights)):
                    agg_weights[key] += included_weights[i][key]
                agg_weights[key] /= len(included_weights)
        else:
            agg_weights = None

        return agg_weights, included_ids, excluded_ids

# ====== Create Edge Servers ======
edge_servers = [
    EdgeServer(1, [0, 1, 2]),
    EdgeServer(2, [3, 4])
]

# ====== Logging Storage ======
log_data = []
last_global_acc = None

# ====== Federated Rounds ======
for r in range(ROUNDS):
    print(f"\n🌐 Round {r + 1}")
    print("=" * 40)

    if r == WINSORIZE_AT_ROUND:
        print("🔧 Applying Winsorization to all clients...")
        for client in clients:
            client.winsorize_data()

    for client in clients:
        client.set_model_weights(server.global_model)

    client_results = []
    for client in clients:
        print(f"  🔄 Client {client.client_id} training...")
        client.train(epochs=EPOCHS)
        acc = client.evaluate()
        client_results.append((client.client_id, acc, client.get_model_weights()))
        print(f"     - Client {client.client_id} Accuracy: {acc:.4f}")

    edge_aggregates = []
    all_excluded_ids = set()

    for edge in edge_servers:
        edge_client_results = [
            res for res in client_results if res[0] in edge.assigned_client_ids
        ]
        agg_weights, included_ids, excluded_ids = edge.filter_and_aggregate(
            edge_client_results, last_global_acc
        )
        if agg_weights is not None:
            edge_aggregates.append(agg_weights)
        all_excluded_ids.update(excluded_ids)

        print(f"  🛠 Edge Server {edge.id} Included: {included_ids} | Excluded: {excluded_ids}")

    for cid in all_excluded_ids:
        print(f"  ⚠ Winsorizing Client {cid} (excluded this round)")
        clients[cid].winsorize_data()

    if edge_aggregates:
        new_global_weights = edge_aggregates[0].copy()
        for key in new_global_weights.keys():
            for i in range(1, len(edge_aggregates)):
                new_global_weights[key] += edge_aggregates[i][key]
            new_global_weights[key] /= len(edge_aggregates)
        server.update_global_model(new_global_weights)

    total_correct, total_samples = 0, 0
    for client in clients:
        client.set_model_weights(server.global_model)
        with torch.no_grad():
            for batch_x, batch_y in client.data_loader:
                outputs = client.model(batch_x)
                _, predicted = torch.max(outputs.data, 1)
                total_samples += batch_y.size(0)
                total_correct += (predicted == batch_y).sum().item()
    global_acc = total_correct / total_samples
    last_global_acc = global_acc

    log_entry = {"Round": r + 1, "Global_Accuracy": global_acc}
    for cid, acc, _ in client_results:
        log_entry[f"Client_{cid}_Acc"] = acc
    log_data.append(log_entry)

    print(f"  📊 Global Accuracy : {global_acc:.4f}")
    print("=" * 40)

# ====== Save CSV ======
df_log = pd.DataFrame(log_data)
csv_path = "federated_uneven.csv"  # changed
df_log.to_csv(csv_path, index=False)
print(f"\n📁 Training log saved to {csv_path}")

# ====== Generate Plots ======
rounds = df_log["Round"]
global_accs = df_log["Global_Accuracy"]
winsor_round = WINSORIZE_AT_ROUND + 1

plt.figure(figsize=(8, 5))
plt.plot(rounds, global_accs, marker='o', color='b', label="Global Accuracy")
plt.axvline(x=winsor_round, color='r', linestyle='--', label='Winsorization Applied')
plt.xlabel("Round")
plt.ylabel("Accuracy")
plt.ylim(0.85, 1.2)
plt.title("Global Accuracy Over Rounds")
plt.grid(True)
plt.legend()
plt.savefig(os.path.join(FINAL_PLOT_DIR, "global_accuracy_uneven.png"))  # changed
plt.close()

plt.figure(figsize=(8, 5))
for i in range(NUM_CLIENTS):
    plt.plot(rounds, df_log[f"Client_{i}_Acc"], marker='o', label=f"Client {i}")
plt.axvline(x=winsor_round, color='r', linestyle='--', label='Winsorization Applied')
plt.xlabel("Round")
plt.ylabel("Accuracy")
plt.ylim(0.85, 1.2)
plt.title("Client Accuracies Over Rounds")
plt.grid(True)
plt.legend()
plt.savefig(os.path.join(FINAL_PLOT_DIR, "client_accuracies_uneven.png"))  # changed
plt.close()

print(f"📊 Plots saved in '{FINAL_PLOT_DIR}' directory.")
