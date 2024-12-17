# Stage 3: Adding Save and Load Methods

import os
from abc import ABC, abstractmethod
import torch
from torch import nn

import os
import torch
from torch import nn
from abc import ABC, abstractmethod
from transformers import BertConfig


class BaseModel(nn.Module, ABC):
    def __init__(self):
        super().__init__()
        self.best_loss = float('inf')  # Initialize with a very high loss value
        self.best_accuracy = 0  # Initialize best accuracy to 0

    @abstractmethod
    def forward(self, x):
        pass

    @property
    def device(self):
        return next(self.parameters()).device  # Returns the device on which the model is located

    def determine_shapes(self, encoder, dim):
        """
        Registers hooks to capture the input and output shapes of specific layers.
        """

        def get_shape(module, input, output):
            module.input_shape = tuple(input[0].shape[-3:])
            module.output_shape = tuple(output[0].shape[-3:])

        input_shape = (1, 2, *dim)  # batch, norms, H, W, D, time ++
        x = torch.ones(input_shape)

        hooks = [
            encoder.down_block1.register_forward_hook(get_shape),
            encoder.down_block3.register_forward_hook(get_shape)
        ]

        with torch.no_grad():
            encoder(x)
            del x

        self.shapes = {
            'dim_0': encoder.down_block1.input_shape,
            'dim_1': encoder.down_block1.output_shape,
            'dim_2': encoder.down_block3.input_shape,
            'dim_3': encoder.down_block3.output_shape
        }

        for hook in hooks:
            hook.remove()

    def register_vars(self, **kwargs):
        """
        Configures dropout rates and model parameters based on task type (e.g., fine-tuning or pretraining).
        """
        task_type = kwargs.get('task', 'pretrain')
        self.dropout_rates = {
            'input': 0,
            'green': 0.35 if task_type == 'fine_tune' else 0.2,
            'Up_green': 0 if task_type == 'fine_tune' else 0.2,
            'transformer': 0.1
        }

        intermediate_vec = 2640
        self.BertConfig = BertConfig(
            hidden_size=intermediate_vec,
            vocab_size=1,
            num_hidden_layers=kwargs.get('transformer_hidden_layers', 12),
            num_attention_heads=16,
            max_position_embeddings=30,
            hidden_dropout_prob=self.dropout_rates['transformer']
        )

        self.label_num = 1
        self.inChannels = 2
        self.outChannels = 1
        self.model_depth = 4
        self.intermediate_vec = intermediate_vec
        self.use_cuda = kwargs.get('cuda', False)
        self.shapes = kwargs.get('shapes')

    def load_partial_state_dict(self, state_dict, load_cls_embedding):
        """
        Loads parameters from a state_dict, handling mismatched parameters gracefully.
        """
        print('Loading parameters onto new model...')
        own_state = self.state_dict()
        loaded = {name: False for name in own_state.keys()}

        for name, param in state_dict.items():
            if name not in own_state:
                print(f'Notice: {name} is not part of new model and was not loaded.')
                continue

            if 'cls_embedding' in name and not load_cls_embedding:
                continue

            if 'position' in name and param.shape != own_state[name].shape:
                continue

            own_state[name].copy_(param.data)
            loaded[name] = True

        for name, was_loaded in loaded.items():
            if not was_loaded:
                print(f'Notice: named parameter - {name} is randomly initialized')


class Encoder(BaseModel):
    def __init__(self, **kwargs):
        super(Encoder, self).__init__()
        self.register_vars(**kwargs)
        self.down_block1 = nn.Sequential(OrderedDict([
            ('conv0', nn.Conv3d(self.inChannels, self.model_depth, kernel_size=3, stride=1, padding=1)),
            ('sp_drop0', nn.Dropout3d(self.dropout_rates['input'])),
            ('green0', GreenBlock(self.model_depth, self.model_depth, self.dropout_rates['green'])),
            ('downsize_0', nn.Conv3d(self.model_depth, self.model_depth * 2, kernel_size=3, stride=2, padding=1))
        ]))
        self.down_block2 = nn.Sequential(OrderedDict([
            ('green10', GreenBlock(self.model_depth * 2, self.model_depth * 2, self.dropout_rates['green'])),
            ('green11', GreenBlock(self.model_depth * 2, self.model_depth * 2, self.dropout_rates['green'])),
            ('downsize_1', nn.Conv3d(self.model_depth * 2, self.model_depth * 4, kernel_size=3, stride=2, padding=1))
        ]))
        self.down_block3 = nn.Sequential(OrderedDict([
            ('green20', GreenBlock(self.model_depth * 4, self.model_depth * 4, self.dropout_rates['green'])),
            ('green21', GreenBlock(self.model_depth * 4, self.model_depth * 4, self.dropout_rates['green'])),
            ('downsize_2', nn.Conv3d(self.model_depth * 4, self.model_depth * 8, kernel_size=3, stride=2, padding=1))
        ]))
        self.final_block = nn.Sequential(OrderedDict([
            ('green30', GreenBlock(self.model_depth * 8, self.model_depth * 8, self.dropout_rates['green'])),
            ('green31', GreenBlock(self.model_depth * 8, self.model_depth * 8, self.dropout_rates['green'])),
            ('green32', GreenBlock(self.model_depth * 8, self.model_depth * 8, self.dropout_rates['green'])),
            ('green33', GreenBlock(self.model_depth * 8, self.model_depth * 8, self.dropout_rates['green']))
        ]))

    def forward(self, x):
        x = self.down_block1(x)
        x = self.down_block2(x)
        x = self.down_block3(x)
        x = self.final_block(x)
        return x

    class Decoder(BaseModel):
        def __init__(self, **kwargs):
            super(Decoder, self).__init__()
            self.register_vars(**kwargs)
            self.decode_block = nn.Sequential(OrderedDict([
                ('upgreen0', UpGreenBlock(self.model_depth * 8, self.model_depth * 4, self.shapes['dim_2'],
                                          self.dropout_rates['Up_green'])),
                ('upgreen1', UpGreenBlock(self.model_depth * 4, self.model_depth * 2, self.shapes['dim_1'],
                                          self.dropout_rates['Up_green'])),
                ('upgreen2', UpGreenBlock(self.model_depth * 2, self.model_depth, self.shapes['dim_0'],
                                          self.dropout_rates['Up_green'])),
                ('blue_block', nn.Conv3d(self.model_depth, self.model_depth, kernel_size=3, stride=1, padding=1)),
                ('output_block',
                 nn.Conv3d(in_channels=self.model_depth, out_channels=self.outChannels, kernel_size=1, stride=1))
            ]))

        def forward(self, x):
            x = self.decode_block(x)
            return x

