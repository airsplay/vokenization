import torch
from torch import nn
import torchvision.models as models
from transformers import *

from .frozen_batch_norm import FrozenBatchNorm2d


LANG_MODELS = {
          'bert':    (BertModel,       BertTokenizer,       'bert-base-uncased'),
          'bert-large':  (BertModel,       BertTokenizer,       'bert-large-uncased'),
          'gpt':     (OpenAIGPTModel,  OpenAIGPTTokenizer,  'openai-gpt'),
          'gpt2':    (GPT2Model,       GPT2Tokenizer,       'gpt2'),
          'ctrl':    (CTRLModel,       CTRLTokenizer,       'ctrl'),
          'xl':      (TransfoXLModel,  TransfoXLTokenizer,  'transfo-xl-wt103'),
          'xlnet':   (XLNetModel,      XLNetTokenizer,      'xlnet-base-cased'),
          'xlm':     (XLMModel,        XLMTokenizer,        'xlm-mlm-enfr-1024'),
          'distil':  (DistilBertModel, DistilBertTokenizer, 'distilbert-base-cased'),
          'roberta': (RobertaModel,    RobertaTokenizer,    'roberta-base'),
          'xlm-roberta': (XLMRobertaModel, XLMRobertaTokenizer, 'xlm-roberta-base'),
}


def get_visn_arch(arch):
    try:
        return getattr(models, arch)
    except AttributeError as e:
        print(e)
        print("There is no arch %s in torchvision." % arch)


class VisnModel(nn.Module):
    def __init__(self, dim, arch='resnet50', pretrained=True, finetuning=False):
        """
        :param dim: dimension of the output
        :param arch: backbone architecture,
        :param pretrained: load feature with pre-trained vector
        :param finetuning: finetune the model
        """
        super().__init__()
        self.finetuning = finetuning

        # Setup Backbone
        resnet = get_visn_arch(arch)(pretrained=pretrained)
        backbone_dim = resnet.fc.in_features
        if not self.finetuning:
            for param in resnet.parameters():
                param.requires_grad = False
        resnet.fc = nn.Identity()
        self.backbone = resnet

        # Surgery on the Networks
        # 1. Frozen Batch Norm
        #    Note that BatchNorm modules have been in-place replaced!
        #    This piece of code is copied from Detectron2, and it was copied from mask-rcnn?
        self.backbone = FrozenBatchNorm2d.convert_frozen_batchnorm(
            self.backbone)
        # print(self.backbone)
        # 2. Frozen the first two (blocks of) layers
        for module in [self.backbone.conv1,
                       self.backbone.layer1]:
            for param in module.parameters():
                param.requires_grad = False

        print(f"Visn Model: {arch}, Finetune: {finetuning}, Pre-trained: {pretrained}")
        print(f"Visn Model: backbone dim {backbone_dim} --> output dim {dim}")

        # Setup follow-up layers
        self.mlp = nn.Sequential(
            nn.Linear(backbone_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, dim),
        )

    def forward(self, img):
        """
        :param img: a tensor of shape [batch_size, H, W, C]
        :return: a tensor of [batch_size, d]
        """
        if not self.finetuning:
            with torch.no_grad():
                x = self.backbone(img)
                x = x.detach()
        else:
            x = self.backbone(img)
        x = self.mlp(x)         # [b, dim]
        x = x / x.norm(2, dim=-1, keepdim=True)
        return x


class LangModel(nn.Module):
    def __init__(self, dim, arch='BERT', layers=(-1,), pretrained=True, finetuning=False):
        """
        :param dim: dimension of the output
        :param arch: backbone architecture,
        :param aggregate: one of 'last4',
        :param pretrained: load feature with pre-trained vector
        :param finetuning: finetune the model
        """
        super().__init__()
        self.finetuning = finetuning

        # Setup Backbone
        Model, Tokenizer, weight = LANG_MODELS[arch]
        bert = Model.from_pretrained(
            weight,
            output_hidden_states=True
        )
        if not pretrained:
            bert.init_weights()

        if not self.finetuning:
            for param in bert.parameters():
                param.requires_grad = False
        backbone_dim = bert.config.hidden_size
        self.backbone = bert
        self.layers = sorted(layers)

        print(f"Language Model: {arch} with weight {weight}; Fine-tuning: {finetuning}, Pre-trained: {pretrained}.")
        print(f"Language Model: using layers {self.layers}, result in backbone dim {backbone_dim * len(self.layers)} "
              f"--> output dim {dim}.")

        # Setup follow-up layers
        self.mlp = nn.Sequential(
            nn.Linear(backbone_dim * len(self.layers), 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, dim),
        )

    def forward(self, input_ids, attention_mask, token_type_ids=None):
        """
        :param input_ids: [batch_size, max_len]
        :param attention_mask: [batch_size, max_len]
        :param token_type_ids: [batch_size, max_len]
        :return: [batch_size, max_len, dim]
        """
        if not self.finetuning:
            with torch.no_grad():
                x = self.backbone(
                    input_ids,
                    attention_mask=attention_mask,
                    token_type_ids=token_type_ids,
                )
        else:
            x = self.backbone(
                input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
            )

        # sequence_output, pooled_output, (hidden_states), (attentions) --> seq_output
        if type(self.backbone) is XLNetModel:
            output, hidden_states = x[:2]
        else:
            output, pooled_output, hidden_states = x[:3]

        # gather the layers
        if type(self.backbone) is XLNetModel:
            x = torch.cat(list(hidden_states[layer].permute(1, 0, 2) for layer in self.layers), -1)
        else:
            x = torch.cat(list(hidden_states[layer] for layer in self.layers), -1)

        if not self.finetuning:
            x = x.detach()

        # [batch_size, max_len, backbone_dim] -->
        # [batch_size, max_len, output_dim]
        x = self.mlp(x)
        x = x / x.norm(2, dim=-1, keepdim=True)
        return x


class JointModel(nn.Module):
    def __init__(self, lang_model, visn_model):
        super().__init__()
        self.lang_model = lang_model
        self.visn_model = visn_model

    def forward(self, lang_input, visn_input):
        lang_output = self.lang_model(*lang_input)
        visn_output = self.visn_model(*visn_input)
        return lang_output, visn_output


