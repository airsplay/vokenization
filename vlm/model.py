import math

import torch
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss, MSELoss, SmoothL1Loss
from torch import nn
from transformers import (
    BertConfig,
    BertForMaskedLM,
)

from transformers.modeling_bert import BertOnlyMLMHead


BertLayerNorm = torch.nn.LayerNorm


# The GLUE function is copied from huggingface transformers:
# https://github.com/huggingface/transformers/blob/c6acd246ec90857b70f449dcbcb1543f150821fc/src/transformers/activations.py
def _gelu_python(x):
    """ Original Implementation of the gelu activation function in Google Bert repo when initially created.
        For information: OpenAI GPT's gelu is slightly different (and gives slightly different results):
        0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
        Also see https://arxiv.org/abs/1606.08415
    """
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


if torch.__version__ < "1.4.0":
    gelu = _gelu_python
else:
    gelu = F.gelu


class CoLBertConfig(BertConfig):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.voken_size = None
        self.voken_dim = None
        self.do_voken_cls = False
        self.do_voken_reg = False
        self.do_voken_ctr = False
        self.shared_head = False
        self.verbose = False


class BertSharedHead(BertOnlyMLMHead):
    """Bert Head for masked language modeling."""

    def __init__(self, config):
        super().__init__(config)
        self.do_voken_cls = config.do_voken_cls
        self.do_voken_ctr = config.do_voken_ctr

        assert int(self.do_voken_cls) + int(self.do_voken_ctr) == 1
        if self.do_voken_cls:
            self.visn_decoder = nn.Linear(config.hidden_size, config.voken_size, bias=True)

        if self.do_voken_ctr:
            self.visn_decoder = nn.Linear(config.voken_dim, config.hidden_size, bias=True)

    def forward(self, features, **kwargs):
        """
        :param features: [batch, length, dim]
        :return: lang_scores [batch, length, vocab_size],
                 visn_scores [batch, length, voken_size]
        """
        x = self.predictions.transform(features)    # batch_size, length, dim

        lang_scores = self.predictions.decoder(x) + self.predictions.bias

        if self.do_voken_cls:
            visn_scores = self.visn_decoder(x)
        elif self.do_voken_ctr:
            voken_feats = kwargs['voken_feats']
            y = self.visn_decoder(voken_feats)  # voken_size, dim
            visn_scores = torch.einsum('bik,jk->bij', x, y)
        else:
            assert False

        return lang_scores, visn_scores


class BertVLMClassificationHead(nn.Module):
    """Bert Head for masked language modeling."""

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.layer_norm = BertLayerNorm(config.hidden_size, eps=config.layer_norm_eps)

        self.decoder = nn.Linear(config.hidden_size, config.voken_size, bias=True)
        # self.decoder = nn.Sequential(
        #     nn.Linear(config.hidden_size, 256, bias=True),
        #     nn.Linear(256, config.voken_size, bias=True),
        # )
        if config.verbose:
            print(f"VLM Classification Head: Build model with voken_size {config.voken_size}")

    def forward(self, features, **kwargs):
        x = self.dense(features)
        x = gelu(x)
        x = self.layer_norm(x)
        x = self.decoder(x)

        return x


class BertVLMContrastiveHeadNew(nn.Module):
    """Bert Head for masked language modeling."""

    def __init__(self, config):
        super().__init__()
        self.joint_dim = 512
        print(f"Contrastive Head: Using joint dim {self.joint_dim}")
        self.voken_size = config.voken_size
        self.dense = nn.Linear(config.hidden_size, self.joint_dim)
        self.layer_norm_x = BertLayerNorm(self.joint_dim, eps=config.layer_norm_eps)

        self.decoder_voken_feat = nn.Linear(config.voken_dim, self.joint_dim, bias=False)
        self.layer_norm_y = BertLayerNorm(self.joint_dim, eps=config.layer_norm_eps)

    def forward(self, bert_output, voken_feats, **kwargs):
        # Process the bert output
        x = self.dense(bert_output)
        x = gelu(x)
        x = self.layer_norm_x(x)

        # Process the pre-trained voken feats.
        y = self.decoder_voken_feat(voken_feats)      # [v, f] --> [v, 64]
        y = self.layer_norm_y(y)

        score = torch.einsum('ijf,vf->ijv', x, y) / math.sqrt(self.joint_dim)
        assert score.dim() == 3 and score.shape[2] == self.voken_size

        return score


class BertVLMContrastiveHead(nn.Module):
    """Bert Head for masked language modeling."""

    def __init__(self, config):
        super().__init__()
        self.voken_size = config.voken_size
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.layer_norm = BertLayerNorm(config.hidden_size, eps=config.layer_norm_eps)

        self.joint_dim = 64
        self.decoder_bert_output = nn.Linear(config.hidden_size, self.joint_dim, bias=False)
        self.decoder_voken_feat = nn.Linear(config.voken_dim, self.joint_dim, bias=False)

    def forward(self, bert_output, voken_feats, **kwargs):
        # Process the bert output
        x = self.dense(bert_output)
        x = gelu(x)
        x = self.layer_norm(x)
        x = self.decoder_bert_output(x)                   # [b, l, f] --> [b, l, 64]

        # Process the pre-trained voken feats.
        y = self.decoder_voken_feat(voken_feats)      # [v, f] --> [v, 64]

        score = torch.einsum('ijf,vf->ijv', x, y) / math.sqrt(self.joint_dim)
        assert score.dim() == 3 and score.shape[2] == self.voken_size

        return score


class BertVLMRegressionHead(nn.Module):
    """Bert Head for masked language modeling."""

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.layer_norm = BertLayerNorm(config.hidden_size, eps=config.layer_norm_eps)

        self.decoder = nn.Linear(config.hidden_size, config.voken_dim, bias=True)

    def forward(self, features, **kwargs):
        x = self.dense(features)
        x = gelu(x)
        x = self.layer_norm(x)

        # project back to size of vocabulary with bias
        x = self.decoder(x)

        return x


class CoLwithBert(BertForMaskedLM):
    config_class = CoLBertConfig

    def __init__(self, config):
        super().__init__(config)
        self.do_voken_cls = config.do_voken_cls
        self.do_voken_reg = config.do_voken_reg
        self.do_voken_ctr = config.do_voken_ctr
        self.shared_head = config.shared_head
        self.verbose = config.verbose

        if self.verbose:
            print(f"Model: do voken cls -- {self.do_voken_cls}, do_voken_reg -- {self.do_voken_reg},"
                  f" do voken ctr -- {self.do_voken_ctr}")

        self.token_cls_loss_fct = CrossEntropyLoss()

        if self.shared_head:
            if self.verbose:
                print("Model: Using shared head for Voken and Token predictions.")
            self.cls = BertSharedHead(config)
            # Reinit the weight of the new head.
            self.init_weights()
        else:
            # Voken Classification
            if config.do_voken_cls:
                self.visual_cls_head = BertVLMClassificationHead(config)

            # Voken Regression
            if config.do_voken_reg:
                assert config.voken_dim is not None, "you need to set voken dim in the config."
                self.visual_reg_head = BertVLMRegressionHead(config)

            # Voken Constrastive
            if config.do_voken_ctr:
                assert config.voken_dim is not None, "you need to set voken dim in the config."
                self.visual_ctr_head = BertVLMContrastiveHeadNew(config)

        # Build voken features embeddings if needed.
        if self.do_voken_ctr or self.do_voken_reg:
            # The voken emb will be preloaded by func "init_voken_feat_emb"
            self.voken_feat_emb = nn.Embedding(
                config.voken_size,
                config.voken_dim
            )
            # Freeze this embedding
            for p in self.voken_feat_emb.parameters():
                p.requires_grad = False

        # Build Loss functions
        if config.do_voken_cls:
            # Voken Classification
            self.voken_cls_loss_fct = CrossEntropyLoss()
        if config.do_voken_reg:
            # Voken Regression
            self.voken_reg_loss_fct = SmoothL1Loss(reduction='none')
            # self.voken_reg_loss_fct = torch.nn.L1Loss(reduction='none')
        if config.do_voken_ctr:
            # Voken Constrastive
            self.voken_ctr_loss_fct = CrossEntropyLoss()

    def init_voken_feat_emb(self, feats):
        if self.verbose:
            print(f"Model: load the voken features with shape {feats.shape}")
            print("\tBefore Loading, std and mean are: ", self.voken_feat_emb.weight.std(), self.voken_feat_emb.weight.mean())
        assert feats.shape == (self.config.voken_size, self.config.voken_dim)
        self.voken_feat_emb.weight.data[:] = torch.Tensor(feats)
        self.original_voken_feats = torch.Tensor(feats).clone()
        self.original_voken_feats = self.original_voken_feats.half()
        if self.verbose:
            print("\tAfter Loading, std and mean are: ", self.voken_feat_emb.weight.std(), self.voken_feat_emb.weight.mean())
            print("\tThe 1st, 2nd, and last voken feats are: ")
            print("\t", self.voken_feat_emb.weight[0])
            print("\t", self.voken_feat_emb.weight[1])
            print("\t", self.voken_feat_emb.weight[-1])
        assert not self.voken_feat_emb.weight.requires_grad
        # print(self.voken_feat_emb.weight.dtype)
        # assert torch.all(torch.eq(self.voken_feat_emb.weight.cuda(),
        #                           self.original_voken_feats)), "The voken feats have been updated during training."

    def to(self, *args):
        if self.do_voken_ctr or self.do_voken_reg:
            self.original_voken_feats = self.original_voken_feats.to(*args)
        return super().to(*args)

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            masked_lm_labels=None,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            lm_labels=None,
            voken_labels=None,
    ):
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
        )
        sequence_output = outputs[0]

        if not self.shared_head:
            voken_loss = 0.
            if self.do_voken_cls:
                assert voken_labels is not None
                voken_scores = self.visual_cls_head(sequence_output)
                voken_cls_loss = self.voken_cls_loss_fct(voken_scores.view(-1, self.config.voken_size), voken_labels.view(-1))
                voken_loss += voken_cls_loss

            if self.do_voken_reg:
                assert voken_labels is not None
                voken_prediction = self.visual_reg_head(sequence_output)

                # Get the mask and pre-trained features
                voken_label_mask = (voken_labels != -100)               # Get a mask of [0, 1, 1, ...., 1, 0], [b, len]
                safe_voken_labels = voken_labels.clone()
                safe_voken_labels[~voken_label_mask] = 0
                voken_feats = self.voken_feat_emb(safe_voken_labels)         # [b, len] --> [b, len, f]

                # Loss
                voken_reg_loss = self.voken_reg_loss_fct(voken_prediction, voken_feats)   # [b, len, f]

                # [b, l, f] * ([b,l] --> [b, l, 1]) = [b, l, f]
                voken_reg_loss = (voken_reg_loss * voken_label_mask.float().unsqueeze(-1))

                # [b, l, f] --sum-> [b, l] --mean-> [1,]
                voken_reg_loss = voken_reg_loss.sum(-1).mean()

                voken_loss += voken_reg_loss

            if self.do_voken_ctr:
                assert torch.all(torch.eq(self.voken_feat_emb.weight,
                                          self.original_voken_feats)), "The voken feats have been updated during training."

                voken_scores = self.visual_ctr_head(
                    sequence_output, self.voken_feat_emb.weight
                )
                voken_ctr_loss = self.voken_ctr_loss_fct(
                    voken_scores.view(-1, self.config.voken_size),
                    voken_labels.view(-1)
                )
                voken_loss += voken_ctr_loss

            if masked_lm_labels is not None:
                prediction_scores = self.cls(sequence_output)
                token_loss = self.token_cls_loss_fct(
                    prediction_scores.view(-1, self.config.vocab_size),
                    masked_lm_labels.view(-1))
            else:
                token_loss = torch.tensor(0.)
        else:
            voken_loss, token_loss = self.calculate_shared_loss(
                sequence_output,
                masked_lm_labels,
                voken_labels,
            )

        return voken_loss, token_loss

    def calculate_shared_loss(self, sequence_output, masked_lm_labels, voken_labels):
        if self.do_voken_cls:
            lang_scores, visn_scores = self.cls(sequence_output)
        else:
            lang_scores, visn_scores = self.cls(
                sequence_output,
                voken_feats=self.voken_feat_emb.weight
            )

        assert voken_labels is not None

        voken_loss_func = self.voken_cls_loss_fct if self.do_voken_cls else self.voken_ctr_loss_fct
        voken_loss = voken_loss_func(
            visn_scores.view(-1, self.config.voken_size),
            voken_labels.view(-1)
        )

        if masked_lm_labels is not None:
            token_loss = self.token_cls_loss_fct(
                lang_scores.view(-1, self.config.vocab_size),
                masked_lm_labels.view(-1)
            )
        else:
            token_loss = torch.tensor(0.)

        return voken_loss, token_loss


class SimpleBertForMaskedLM(BertForMaskedLM):

    def __init__(self, config):
        super().__init__(config)

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            masked_lm_labels=None,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            lm_labels=None,
    ):
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
        )
        sequence_output = outputs[0]

        prediction_scores = self.cls(sequence_output)
        loss_fct = CrossEntropyLoss()
        token_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), masked_lm_labels.view(-1))

        return token_loss,
