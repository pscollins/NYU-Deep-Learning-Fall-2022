import torch

class ModelWrapper(torch.nn.Module):
    def __init__(self, inner_model):
        super().__init__()
        self.inner_model = inner_model


    def forward(self, *args, **kwargs):
        print(f'GOT ARGS: {args}')
        print(f'GOT KWARGS: {kwargs}')
        return self.inner_model(*args, **kwargs)
