import json
import torch
import pytest
from loguru import logger

@pytest.fixture
def create_payload():
    def _create_payload(model_dim=(10,20,10), drop=[]):

        class SimpleModel(torch.nn.Module):
            def __init__(self, input_dim, hidden_dim, output_dim):
                super(SimpleModel, self).__init__()
                # Set up layers based on provided dimensions
                self.fc1 = torch.nn.Linear(input_dim, hidden_dim)
                self.fc2 = torch.nn.Linear(hidden_dim, output_dim)
                
            def forward(self, x):
                x = torch.relu(self.fc1(x))
                x = torch.relu(self.fc2(x))
                return x
            
        model = SimpleModel(model_dim[0], model_dim[1], model_dim[2])
            
        local = {
            "lr": 1e-6,
            "batch_size": 512,
            "rounds": 10,
            "patience": 10
        }

        federated = {
            "job_id": "test",
            "rounds": 100,
            "input": ["weights"],
            "return": ["weights"]
        }

        state_dict = model.state_dict()

        payload = {
            "local": local,
            "federated": federated,
            "model": {key: value.tolist() for key, value in state_dict.items()}
        }

        # remove columns
        for key in drop:
            del payload[key]

        return payload
    
    return _create_payload

@pytest.fixture
def save_local_file():
    def _save_local_file(file={
        "lr": 1e-6,
        "batch_size": 512,
        "rounds": 10,
        "patience": 10
    }):

        with open("test/local.json", 'w') as f:
            json.dump(file, f, indent=4)
            f.close()

    return _save_local_file

@pytest.fixture
def save_federated_file():
    def _save_federated_file(file={
        "job_id": "test",
        "rounds": 100,
        "input": ["weights"],
        "return": ["weights"]
    }):

        with open("test/federated.json", 'w') as f:
            json.dump(file, f, indent=4)
            f.close()

    return _save_federated_file


@pytest.fixture
def save_model():
    def _save_model(model_dim=(10, 20, 10)):
        class SimpleModel(torch.nn.Module):
            def __init__(self, input_dim, hidden_dim, output_dim):
                super(SimpleModel, self).__init__()
                # Set up layers based on provided dimensions
                self.fc1 = torch.nn.Linear(input_dim, hidden_dim)
                self.fc2 = torch.nn.Linear(hidden_dim, output_dim)
                    
            def forward(self, x):
                x = torch.relu(self.fc1(x))
                x = torch.relu(self.fc2(x))
                return x
                
        model = SimpleModel(model_dim[0], model_dim[1], model_dim[2])

        state_dict = model.state_dict()

        file = {key: value.tolist() for key, value in state_dict.items()}

        with open("test/model.json", 'w') as f:
            json.dump(file, f, indent=4)
            f.close()

    return _save_model