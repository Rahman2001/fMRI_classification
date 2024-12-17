
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
        """Registers hooks to capture the input and output shapes of specific layers."""

        def get_shape(module, input, output):
            module.input_shape = tuple(input[0].shape[-3:])
            module.output_shape = tuple(output.shape[-3:])

        hooks = [
            encoder.down_block1.register_forward_hook(get_shape),
            encoder.down_block3.register_forward_hook(get_shape)
        ]

        input_shape = (1, 2) + dim  # batch, channels, H, W, D, time ++
        x = torch.ones(input_shape)

        with torch.no_grad():
            encoder(x)

        self.shapes = {
            'dim_0': encoder.down_block1.input_shape,
            'dim_1': encoder.down_block1.output_shape,
            'dim_2': encoder.down_block3.input_shape,
            'dim_3': encoder.down_block3.output_shape
        }

        for hook in hooks:
            hook.remove()

    def register_vars(self, **kwargs):
        """Configures model-specific parameters like dropout rates."""
        intermediate_vec = 2640
        task = kwargs.get('task', 'pretrain')
        self.dropout_rates = {
            'input': 0,
            'green': 0.35 if task == 'fine_tune' else 0.2,
            'Up_green': 0 if task == 'fine_tune' else 0.2,
            'transformer': 0.1
        }

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
        self.shapes = kwargs.get('shapes', {})

    def load_partial_state_dict(self, state_dict, load_cls_embedding):
        """Loads parameters from a state_dict, handling mismatched parameters gracefully."""
        print('Loading parameters onto the new model...')
        own_state = self.state_dict()
        loaded = {name: False for name in own_state.keys()}

        for name, param in state_dict.items():
            if name not in own_state:
                print(f'Notice: {name} is not part of the new model and was not loaded.')
                continue
            if 'cls_embedding' in name and not load_cls_embedding:
                continue
            if 'position' in name and param.shape != own_state[name].shape:
                print(f'Notice: Shape mismatch for {name}, skipping...')
                continue
            own_state[name].copy_(param.data)
            loaded[name] = True

        for name, was_loaded in loaded.items():
            if not was_loaded:
                print(f'Notice: Named parameter {name} is randomly initialized.')

    def save_checkpoint(self, directory, title, epoch, loss, accuracy=None, optimizer=None, schedule=None):
        """
        Saves the current state of the model and optimizer.
        Updates the best model if the loss or accuracy improves.
        """
        os.makedirs(directory, exist_ok=True)

        # Build checkpoint dict to save.
        ckpt_dict = {
            'model_state_dict': self.state_dict(),
            'epoch': epoch,
            'loss_value': loss
        }

        if optimizer:
            ckpt_dict['optimizer_state_dict'] = optimizer.state_dict()
        if accuracy is not None:
            ckpt_dict['accuracy'] = accuracy
        if schedule:
            ckpt_dict.update({
                'schedule_state_dict': schedule.state_dict(),
                'lr': schedule.get_last_lr()[0]
            })
        if hasattr(self, 'loaded_model_weights_path'):
            ckpt_dict['loaded_model_weights_path'] = self.loaded_model_weights_path

        # Save the last checkpoint
        torch.save(ckpt_dict, os.path.join(directory, f"{title}_last_epoch.pth"))

        # Save the best checkpoint based on validation loss
        if self.best_loss > loss:
            self.best_loss = loss
            torch.save(ckpt_dict, os.path.join(directory, f"{title}_BEST_val_loss.pth"))
            print('Best validation loss model updated.')

        # Save the best checkpoint based on validation accuracy
        if accuracy is not None and self.best_accuracy < accuracy:
            self.best_accuracy = accuracy
            torch.save(ckpt_dict, os.path.join(directory, f"{title}_BEST_val_accuracy.pth"))
            print('Best validation accuracy model updated.')

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
class BottleNeck_in(BaseModel):
    def __init__(self, **kwargs):
        super(BottleNeck_in, self).__init__()
        self.register_vars(**kwargs)
        self.reduce_dimension = nn.Sequential(OrderedDict([
            ('group_normR', nn.GroupNorm(num_channels=self.model_depth * 8, num_groups=8)),
            # ('norm0', nn.BatchNorm3d(model_depth * 8)),
            ('reluR0', nn.LeakyReLU(inplace=True)),
            ('convR0',
             nn.Conv3d(self.model_depth * 8, self.model_depth // 2, kernel_size=(3, 3, 3), stride=1, padding=1)),
        ]))
        flat_factor = tuple_prod(self.shapes['dim_3'])
        self.flatten = nn.Flatten()
        if (flat_factor * self.model_depth // 2) == self.intermediate_vec:
            self.into_bert = nn.Identity()
            print('flattened vec identical to intermediate vector...\ndroppping fully conneceted bottleneck...')
        else:
            self.into_bert = nn.Linear(in_features=(self.model_depth // 2) * flat_factor,
                                       out_features=self.intermediate_vec)

    def forward(self, inputs):
        x = self.reduce_dimension(inputs)
        x = self.flatten(x)
        x = self.into_bert(x)

        return x


class BottleNeck_out(BaseModel):
    def __init__(self, **kwargs):
        super(BottleNeck_out, self).__init__()
        self.register_vars(**kwargs)
        flat_factor = tuple_prod(self.shapes['dim_3'])
        minicube_shape = (self.model_depth // 2,) + self.shapes['dim_3']
        self.out_of_bert = nn.Linear(in_features=self.intermediate_vec,
                                     out_features=(self.model_depth // 2) * flat_factor)
        self.expand_dimension = nn.Sequential(OrderedDict([
            ('unflatten', nn.Unflatten(1, minicube_shape)),
            ('group_normR', nn.GroupNorm(num_channels=self.model_depth // 2, num_groups=2)),
            # ('norm0', nn.BatchNorm3d(model_depth * 8)),
            ('reluR0', nn.LeakyReLU(inplace=True)),
            ('convR0',
             nn.Conv3d(self.model_depth // 2, self.model_depth * 8, kernel_size=(3, 3, 3), stride=1, padding=1)),
        ]))

    def forward(self, x):
        x = self.out_of_bert(x)
        return self.expand_dimension(x)


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


class AutoEncoder(BaseModel):
    def __init__(self, dim, **kwargs):
        super(AutoEncoder, self).__init__()
        # ENCODING
        self.task = 'autoencoder_reconstruction'
        self.encoder = Encoder(**kwargs)
        self.determine_shapes(self.encoder, dim)
        kwargs['shapes'] = self.shapes
        # BottleNeck into bert
        self.into_bert = BottleNeck_in(**kwargs)

        # BottleNeck out of bert
        self.from_bert = BottleNeck_out(**kwargs)

        # DECODER
        self.decoder = Decoder(**kwargs)

    def forward(self, x):
        if x.isnan().any():
            print('nans in data!')
        batch_size, Channels_in, W, H, D, T = x.shape
        x = x.permute(0, 5, 1, 2, 3, 4).reshape(batch_size * T, Channels_in, W, H, D)
        encoded = self.encoder(x)
        encoded = self.into_bert(encoded)
        encoded = self.from_bert(encoded)
        reconstructed_image = self.decoder(encoded)
        _, Channels_out, W, H, D = reconstructed_image.shape
        reconstructed_image = reconstructed_image.reshape(batch_size, T, Channels_out, W, H, D).permute(0, 2, 3, 4, 5,
                                                                                                        1)
        return {'reconstructed_fmri_sequence': reconstructed_image}


class Transformer_Block(BertPreTrainedModel, BaseModel):
    def __init__(self, config, **kwargs):
        super(Transformer_Block, self).__init__(config)
        self.register_vars(**kwargs)
        self.cls_pooling = True
        self.bert = BertModel(self.BertConfig, add_pooling_layer=self.cls_pooling)
        self.init_weights()
        self.cls_embedding = nn.Sequential(nn.Linear(self.BertConfig.hidden_size, self.BertConfig.hidden_size),
                                           nn.LeakyReLU())
        self.register_buffer('cls_id', torch.ones((kwargs.get('batch_size'), 1, self.BertConfig.hidden_size)) * 0.5,
                             persistent=False)

    def concatenate_cls(self, x):
        cls_token = self.cls_embedding(self.cls_id)
        return torch.cat([cls_token, x], dim=1)

    def forward(self, x):
        inputs_embeds = self.concatenate_cls(x=x)
        outputs = self.bert(input_ids=None,
                            attention_mask=None,
                            token_type_ids=None,
                            position_ids=None,
                            head_mask=None,
                            inputs_embeds=inputs_embeds,
                            encoder_hidden_states=None,
                            encoder_attention_mask=None,
                            output_attentions=None,
                            output_hidden_states=None,
                            return_dict=self.BertConfig.use_return_dict
                            )

        sequence_output = outputs[0][:, 1:, :]
        pooled_cls = outputs[1]

        return {'sequence': sequence_output, 'cls': pooled_cls}


class Encoder_Transformer_Decoder(BaseModel):
    def __init__(self, dim, **kwargs):
        super(Encoder_Transformer_Decoder, self).__init__()
        self.task = 'transformer_reconstruction'
        self.register_vars(**kwargs)
        # ENCODING
        self.encoder = Encoder(**kwargs)
        self.determine_shapes(self.encoder, dim)
        kwargs['shapes'] = self.shapes
        # BottleNeck into bert
        self.into_bert = BottleNeck_in(**kwargs)

        # transformer
        self.transformer = Transformer_Block(self.BertConfig, **kwargs)

        # BottleNeck out of bert
        self.from_bert = BottleNeck_out(**kwargs)

        # DECODER
        self.decoder = Decoder(**kwargs)

    def forward(self, x):
        batch_size, inChannels, W, H, D, T = x.shape
        x = x.permute(0, 5, 1, 2, 3, 4).reshape(batch_size * T, inChannels, W, H, D)
        encoded = self.encoder(x)
        encoded = self.into_bert(encoded)
        encoded = encoded.reshape(batch_size, T, -1)
        transformer_dict = self.transformer(encoded)
        out = transformer_dict['sequence'].reshape(batch_size * T, -1)
        out = self.from_bert(out)
        reconstructed_image = self.decoder(out)
        reconstructed_image = reconstructed_image.reshape(batch_size, T, self.outChannels, W, H, D).permute(0, 2, 3, 4,
                                                                                                            5, 1)
        return {'reconstructed_fmri_sequence': reconstructed_image}


class Encoder_Transformer_finetune(BaseModel):
    def __init__(self, dim, **kwargs):
        super(Encoder_Transformer_finetune, self).__init__()
        self.task = kwargs.get('fine_tune_task')
        self.register_vars(**kwargs)
        # ENCODING
        self.encoder = Encoder(**kwargs)
        self.determine_shapes(self.encoder, dim)
        kwargs['shapes'] = self.shapes
        # BottleNeck into bert
        self.into_bert = BottleNeck_in(**kwargs)

        # transformer
        self.transformer = Transformer_Block(self.BertConfig, **kwargs)
        # finetune classifier
        if kwargs.get('fine_tune_task') == 'regression':
            self.final_activation_func = nn.LeakyReLU()
        elif kwargs.get('fine_tune_task') == 'binary_classification':
            self.final_activation_func = nn.Sigmoid()
            self.label_num = 1
        self.regression_head = nn.Sequential(nn.Linear(self.BertConfig.hidden_size, self.label_num),
                                             self.final_activation_func)

    def forward(self, x):
        batch_size, inChannels, W, H, D, T = x.shape
        x = x.permute(0, 5, 1, 2, 3, 4).reshape(batch_size * T, inChannels, W, H, D)
        encoded = self.encoder(x)
        encoded = self.into_bert(encoded)
        encoded = encoded.reshape(batch_size, T, -1)
        transformer_dict = self.transformer(encoded)
        CLS = transformer_dict['cls']
        prediction = self.regression_head(CLS)
        return {self.task: prediction}

