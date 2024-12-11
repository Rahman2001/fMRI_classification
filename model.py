# Stage 2: Adding Helper Methods

import os
from abc import ABC, abstractmethod
import torch
from torch import nn

class BaseModel(nn.Module, ABC):
    def __init__(self):
        super().__init__()
        self.best_loss = float('inf')  # Initialize with a very high loss value
        self.best_accuracy = 0         # Initialize best accuracy to 0

    @abstractmethod
    def forward(self, x):
        pass

    @property
    def device(self):
        return next(self.parameters()).device  # Returns the device on which the model is located

    def determine_shapes(self, encoder, dim):
        # Registers hooks to capture the input and output shapes of specific layers.
        def get_shape(module, input, output):
            module.input_shape = tuple(input[0].shape[-3:])
            module.output_shape = tuple(output[0].shape[-3:])

        hook1 = encoder.down_block1.register_forward_hook(get_shape)
        hook2 = encoder.down_block3.register_forward_hook(get_shape)
        input_shape = (1, 2,) + dim  # batch, norms, H, W, D, time
        x = torch.ones((input_shape))
        with torch.no_grad():
            encoder(x)
            del x
        self.shapes = {'dim_0': encoder.down_block1.input_shape,
                       'dim_1': encoder.down_block1.output_shape,
                       'dim_2': encoder.down_block3.input_shape,
                       'dim_3': encoder.down_block3.output_shape}
        hook1.remove()
        hook2.remove()

    def register_vars(self, **kwargs):
        # Configures dropout rates based on task type (e.g., fine-tuning or pretraining).
        intermediate_vec = 2640
        if kwargs.get('task') == 'fine_tune':
            self.dropout_rates = {'input': 0, 'green': 0.35, 'Up_green': 0, 'transformer': 0.1}
        else:
            self.dropout_rates = {'input': 0, 'green': 0.2, 'Up_green': 0.2, 'transformer': 0.1}

        self.BertConfig = BertConfig(hidden_size=intermediate_vec, vocab_size=1,
                                     num_hidden_layers=kwargs.get('transformer_hidden_layers'),
                                     num_attention_heads=16, max_position_embeddings=30,
                                     hidden_dropout_prob=self.dropout_rates['transformer'])

        self.label_num = 1
        self.inChannels = 2
        self.outChannels = 1
        self.model_depth = 4
        self.intermediate_vec = intermediate_vec
        self.use_cuda = kwargs.get('cuda')
        self.shapes = kwargs.get('shapes')
