# client.py
# without edge server but with basic deduction logic
'''
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
from model import LogisticRegressionModel

class Client:
    def __init__(self, client_id, csv_path, input_size):
        self.client_id = client_id
        self.csv_path = csv_path
        self.input_size = input_size
        self.model = LogisticRegressionModel(input_size, num_classes=3)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(self.model.parameters(), lr=0.01)
        self.data_loader = self._prepare_data()
        self.history = []

    # client.py
    def _prepare_data(self):
        df = pd.read_csv(self.csv_path)

    # Drop non-numeric columns (e.g., FieldID, Date, etc.)
        df = df.select_dtypes(include=["number"])

    # Features = all columns except "Cluster" (target)
        X = df.drop(columns=["Cluster"]).values.astype("float32")
        y = df["Cluster"].values.astype("int64")

        tensor_x = torch.tensor(X)
        tensor_y = torch.tensor(y)

        dataset = TensorDataset(tensor_x, tensor_y)
        return DataLoader(dataset, batch_size=16, shuffle=True)

    def train(self, epochs=1):
        self.model.train()
        for _ in range(epochs):
            for batch_x, batch_y in self.data_loader:
                self.optimizer.zero_grad()
                outputs = self.model(batch_x)
                loss = self.criterion(outputs, batch_y)
                loss.backward()
                self.optimizer.step()

    def evaluate(self):
        self.model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for batch_x, batch_y in self.data_loader:
                outputs = self.model(batch_x)
                _, predicted = torch.max(outputs.data, 1)
                total += batch_y.size(0)
                correct += (predicted == batch_y).sum().item()
        return correct / total

    def get_model_weights(self):
        return self.model.state_dict()

    def set_model_weights(self, state_dict):
        self.model.load_state_dict(state_dict)
'''

# with edge server and basic deduction logic
'''
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
from model import LogisticRegressionModel

class Client:
    def __init__(self, client_id, csv_path, input_size):
        self.client_id = client_id
        self.csv_path = csv_path
        self.input_size = input_size
        self.model = LogisticRegressionModel(input_size, num_classes=3)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(self.model.parameters(), lr=0.01)
        self.data_loader = self._prepare_data()

    def _prepare_data(self):
        df = pd.read_csv(self.csv_path)
        df = df.select_dtypes(include=["number"])
        X = df.drop(columns=["Cluster"]).values.astype("float32")
        y = df["Cluster"].values.astype("int64")
        tensor_x = torch.tensor(X)
        tensor_y = torch.tensor(y)
        dataset = TensorDataset(tensor_x, tensor_y)
        return DataLoader(dataset, batch_size=16, shuffle=True)

    def train(self, epochs=1):
        self.model.train()
        for _ in range(epochs):
            for batch_x, batch_y in self.data_loader:
                self.optimizer.zero_grad()
                outputs = self.model(batch_x)
                loss = self.criterion(outputs, batch_y)
                loss.backward()
                self.optimizer.step()

    def evaluate(self):
        self.model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for batch_x, batch_y in self.data_loader:
                outputs = self.model(batch_x)
                _, predicted = torch.max(outputs.data, 1)
                total += batch_y.size(0)
                correct += (predicted == batch_y).sum().item()
        return correct / total

    def get_model_weights(self):
        return self.model.state_dict()

    def set_model_weights(self, state_dict):
        self.model.load_state_dict(state_dict)
'''

# without edge server and deduction logic and without winsorisation
# client.py
'''
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
from model import LogisticRegressionModel

class Client:
    def __init__(self, client_id, csv_path, input_size):
        self.client_id = client_id
        self.csv_path = csv_path
        self.input_size = input_size
        self.model = LogisticRegressionModel(input_size, num_classes=3)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(self.model.parameters(), lr=0.01)
        self.data_loader = self._prepare_data()

    def _prepare_data(self):
        df = pd.read_csv(self.csv_path)
        df = df.select_dtypes(include=["number"])  # Drop non-numeric columns
        X = df.drop(columns=["Cluster"]).values.astype("float32")
        y = df["Cluster"].values.astype("int64")
        tensor_x = torch.tensor(X)
        tensor_y = torch.tensor(y)
        dataset = TensorDataset(tensor_x, tensor_y)
        return DataLoader(dataset, batch_size=16, shuffle=True)

    def train(self, epochs=1):
        self.model.train()
        for _ in range(epochs):
            for batch_x, batch_y in self.data_loader:
                self.optimizer.zero_grad()
                outputs = self.model(batch_x)
                loss = self.criterion(outputs, batch_y)
                loss.backward()
                self.optimizer.step()

    def evaluate(self):
        self.model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for batch_x, batch_y in self.data_loader:
                outputs = self.model(batch_x)
                _, predicted = torch.max(outputs.data, 1)
                total += batch_y.size(0)
                correct += (predicted == batch_y).sum().item()
        return correct / total

    def get_model_weights(self):
        return self.model.state_dict()

    def set_model_weights(self, state_dict):
        self.model.load_state_dict(state_dict)
'''
# without edge server and deduction logic and with winsorisation
# client.py
'''
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
from model import LogisticRegressionModel

class Client:
    def __init__(self, client_id, csv_path, input_size):
        self.client_id = client_id
        self.csv_path = csv_path
        self.input_size = input_size
        self.model = LogisticRegressionModel(input_size, num_classes=3)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(self.model.parameters(), lr=0.01)
        self.df = pd.read_csv(self.csv_path)  # store raw DataFrame
        self.data_loader = self._prepare_data(self.df)
        self.history = []

    def _prepare_data(self, df):
        # Drop non-numeric columns
        df = df.select_dtypes(include=["number"])
        X = df.drop(columns=["Cluster"]).values.astype("float32")
        y = df["Cluster"].values.astype("int64")
        tensor_x = torch.tensor(X)
        tensor_y = torch.tensor(y)
        dataset = TensorDataset(tensor_x, tensor_y)
        return DataLoader(dataset, batch_size=16, shuffle=True)

    def winsorize_data(self, lower=0.05, upper=0.95):
        for col in self.df.select_dtypes(include=["number"]).columns:
            if col != "Cluster":
                low = self.df[col].quantile(lower)
                high = self.df[col].quantile(upper)
                self.df[col] = self.df[col].clip(lower=low, upper=high)
        # Recreate dataloader
        self.data_loader = self._prepare_data(self.df)

    def train(self, epochs=1):
        self.model.train()
        for _ in range(epochs):
            for batch_x, batch_y in self.data_loader:
                self.optimizer.zero_grad()
                outputs = self.model(batch_x)
                loss = self.criterion(outputs, batch_y)
                loss.backward()
                self.optimizer.step()

    def evaluate(self):
        self.model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for batch_x, batch_y in self.data_loader:
                outputs = self.model(batch_x)
                _, predicted = torch.max(outputs.data, 1)
                total += batch_y.size(0)
                correct += (predicted == batch_y).sum().item()
        return correct / total

    def get_model_weights(self):
        return self.model.state_dict()

    def set_model_weights(self, state_dict):
        self.model.load_state_dict(state_dict)
'''
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
from model import LogisticRegressionModel

class Client:
    def __init__(self, client_id, csv_path, input_size, seed=42):
        self.client_id = client_id
        self.csv_path = csv_path
        self.input_size = input_size
        self.seed = seed

        # Model & optimizer
        torch.manual_seed(self.seed)  # make sure initial weights are reproducible
        self.model = LogisticRegressionModel(input_size, num_classes=3)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(self.model.parameters(), lr=0.01)

        # Load raw DataFrame
        self.df = pd.read_csv(self.csv_path)

        # Prepare DataLoader with fixed seed
        self.data_loader = self._prepare_data(self.df)

        self.history = []

    def _prepare_data(self, df):
        # Drop non-numeric columns
        df = df.select_dtypes(include=["number"])
        X = df.drop(columns=["Cluster"]).values.astype("float32")
        y = df["Cluster"].values.astype("int64")

        tensor_x = torch.tensor(X)
        tensor_y = torch.tensor(y)
        dataset = TensorDataset(tensor_x, tensor_y)

        # Fixed shuffle order using generator with seed
        g = torch.Generator()
        g.manual_seed(self.seed)

        return DataLoader(dataset, batch_size=16, shuffle=True, generator=g)

    def winsorize_data(self, lower=0.05, upper=0.95):
        for col in self.df.select_dtypes(include=["number"]).columns:
            if col != "Cluster":
                low = self.df[col].quantile(lower)
                high = self.df[col].quantile(upper)
                self.df[col] = self.df[col].clip(lower=low, upper=high)

        # Recreate dataloader with same seed so order is stable
        self.data_loader = self._prepare_data(self.df)

    def train(self, epochs=1):
        self.model.train()
        for _ in range(epochs):
            # Seed before every epoch for deterministic mini-batch order
            g = torch.Generator()
            g.manual_seed(self.seed)
            for batch_x, batch_y in DataLoader(
                self.data_loader.dataset, 
                batch_size=16, 
                shuffle=True, 
                generator=g
            ):
                self.optimizer.zero_grad()
                outputs = self.model(batch_x)
                loss = self.criterion(outputs, batch_y)
                loss.backward()
                self.optimizer.step()

    def evaluate(self):
        self.model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for batch_x, batch_y in self.data_loader:
                outputs = self.model(batch_x)
                _, predicted = torch.max(outputs.data, 1)
                total += batch_y.size(0)
                correct += (predicted == batch_y).sum().item()
        return correct / total

    def get_model_weights(self):
        return self.model.state_dict()

    def set_model_weights(self, state_dict):
        self.model.load_state_dict(state_dict)
