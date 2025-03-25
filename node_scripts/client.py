import flwr as fl
import torch
from transformers import WhisperForConditionalGeneration

# Load your local model
model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-medium")

class WhisperClient(fl.client.NumPyClient):
    def get_parameters(self, config):
        return [val.cpu().numpy() for _, val in model.state_dict().items()]

    def set_parameters(self, parameters):
        state_dict = model.state_dict()
        for k, v in zip(state_dict.keys(), parameters):
            state_dict[k] = torch.tensor(v)
        model.load_state_dict(state_dict)

    def fit(self, parameters, config):
        self.set_parameters(parameters)

        # TODO: fine-tune locally on your dataset here
        # (for now just simulate 1 step of training)

        return self.get_parameters(config={}), len(parameters), {}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        return 0.0, len(parameters), {}

fl.client.start_numpy_client(server_address="127.0.0.1:8080", client=WhisperClient())
