from transformers.file_utils import cached_path, hf_bucket_url
from importlib.machinery import SourceFileLoader
import os
from transformers import EncoderDecoderModel, AutoConfig, AutoModel, EncoderDecoderConfig, RobertaForCausalLM, \
    RobertaModel
from transformers.modeling_utils import PreTrainedModel, logging
import torch
from torch.nn import CrossEntropyLoss, Parameter
from transformers.modeling_outputs import Seq2SeqLMOutput, CausalLMOutputWithCrossAttentions, \
    ModelOutput
from attentions import ScaledDotProductAttention, MultiHeadAttention
from collections import namedtuple
from typing import Dict, Any, Optional, Tuple
from dataclasses import dataclass
import random
from model_config_handling import EncoderDecoderSpokenNormConfig, DecoderSpokenNormConfig, PretrainedConfig

cache_dir = './cache'
model_name = 'nguyenvulebinh/envibert'

if not os.path.exists(cache_dir):
    os.makedirs(cache_dir)
logger = logging.get_logger(__name__)


@dataclass
class SpokenNormOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    logits_spoken_tagging: torch.FloatTensor = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    decoder_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    decoder_attentions: Optional[Tuple[torch.FloatTensor]] = None
    cross_attentions: Optional[Tuple[torch.FloatTensor]] = None
    encoder_last_hidden_state: Optional[torch.FloatTensor] = None
    encoder_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    encoder_attentions: Optional[Tuple[torch.FloatTensor]] = None




def collect_spoken_phrases_features(encoder_hidden_states, word_src_lengths, spoken_label):
    list_features = []
    list_features_mask = []
    max_length = word_src_lengths.max()
    feature_pad = torch.zeros_like(encoder_hidden_states[0, :1, :])
    for hidden_state, word_length, list_idx in zip(encoder_hidden_states, word_src_lengths, spoken_label):
        for idx in list_idx:
            if idx > 0:
                start = sum(word_length[:idx])
                end = start + word_length[idx]
                remain_length = max_length - word_length[idx]
                list_features_mask.append(torch.cat([torch.ones_like(spoken_label[0, 0]).expand(word_length[idx]),
                                                     torch.zeros_like(
                                                         spoken_label[0, 0].expand(remain_length))]).unsqueeze(0))
                spoken_phrases_feature = hidden_state[start: end]

                list_features.append(torch.cat([spoken_phrases_feature,
                                                feature_pad.expand(remain_length, feature_pad.size(-1))]).unsqueeze(0))
    return torch.cat(list_features), torch.cat(list_features_mask)


def collect_spoken_phrases_labels(decoder_input_ids, labels, labels_bias, word_tgt_lengths, spoken_idx):
    list_decoder_input_ids = []
    list_labels = []
    list_labels_bias = []
    max_length = word_tgt_lengths.max()
    init_decoder_ids = torch.tensor([0], device=labels.device, dtype=labels.dtype)
    pad_decoder_ids = torch.tensor([1], device=labels.device, dtype=labels.dtype)
    eos_decoder_ids = torch.tensor([2], device=labels.device, dtype=labels.dtype)
    none_labels_bias = torch.tensor([0], device=labels.device, dtype=labels.dtype)
    ignore_labels_bias = torch.tensor([-100], device=labels.device, dtype=labels.dtype)

    for decoder_inputs, decoder_label, decoder_label_bias, word_length, list_idx in zip(decoder_input_ids,
                                                                                        labels, labels_bias,
                                                                                        word_tgt_lengths, spoken_idx):
        for idx in list_idx:
            if idx > 0:
                start = sum(word_length[:idx - 1])
                end = start + word_length[idx - 1]
                remain_length = max_length - word_length[idx - 1]
                remain_decoder_input_ids = max_length - len(decoder_inputs[start + 1:end + 1])
                list_decoder_input_ids.append(torch.cat([init_decoder_ids,
                                                         decoder_inputs[start + 1:end + 1],
                                                         pad_decoder_ids.expand(remain_decoder_input_ids)]).unsqueeze(0))
                list_labels.append(torch.cat([decoder_label[start:end],
                                              eos_decoder_ids,
                                              ignore_labels_bias.expand(remain_length)]).unsqueeze(0))
                list_labels_bias.append(torch.cat([decoder_label_bias[start:end],
                                                   none_labels_bias,
                                                   ignore_labels_bias.expand(remain_length)]).unsqueeze(0))

    decoder_input_ids = torch.cat(list_decoder_input_ids)
    labels = torch.cat(list_labels)
    labels_bias = torch.cat(list_labels_bias)

    return decoder_input_ids, labels, labels_bias


class EncoderDecoderSpokenNorm(EncoderDecoderModel):
    config_class = EncoderDecoderSpokenNormConfig

    def __init__(
            self,
            config: Optional[PretrainedConfig] = None,
            encoder: Optional[PreTrainedModel] = None,
            decoder: Optional[PreTrainedModel] = None,
    ):
        if config is None and (encoder is None or decoder is None):
            raise ValueError("Either a configuration or an encoder and a decoder has to be provided.")
        if config is None:
            config = EncoderDecoderConfig.from_encoder_decoder_configs(encoder.config, decoder.config)
        else:
            if not isinstance(config, self.config_class):
                raise ValueError(f"Config: {config} has to be of type {self.config_class}")

        if config.decoder.cross_attention_hidden_size is not None:
            if config.decoder.cross_attention_hidden_size != config.encoder.hidden_size:
                raise ValueError(
                    "If `cross_attention_hidden_size` is specified in the decoder's configuration, "
                    "it has to be equal to the encoder's `hidden_size`. "
                    f"Got {config.decoder.cross_attention_hidden_size} for `config.decoder.cross_attention_hidden_size` "
                    f"and {config.encoder.hidden_size} for `config.encoder.hidden_size`."
                )

        # initialize with config
        super().__init__(config)

        if encoder is None:
            from transformers.models.auto.modeling_auto import AutoModel

            encoder = AutoModel.from_config(config.encoder)

        if decoder is None:
            # from transformers.models.auto.modeling_auto import AutoModelForCausalLM

            decoder = DecoderSpokenNorm._from_config(config.decoder)

        self.encoder = encoder
        self.decoder = decoder

        if self.encoder.config.to_dict() != self.config.encoder.to_dict():
            logger.warning(
                f"Config of the encoder: {self.encoder.__class__} is overwritten by shared encoder config: {self.config.encoder}"
            )
        if self.decoder.config.to_dict() != self.config.decoder.to_dict():
            logger.warning(
                f"Config of the decoder: {self.decoder.__class__} is overwritten by shared decoder config: {self.config.decoder}"
            )

        # make sure that the individual model's config refers to the shared config
        # so that the updates to the config will be synced
        self.encoder.config = self.config.encoder
        self.decoder.config = self.config.decoder

        # encoder outputs might need to be projected to different dimension for decoder
        if (
                self.encoder.config.hidden_size != self.decoder.config.hidden_size
                and self.decoder.config.cross_attention_hidden_size is None
        ):
            self.enc_to_dec_proj = torch.nn.Linear(self.encoder.config.hidden_size, self.decoder.config.hidden_size)

        if self.encoder.get_output_embeddings() is not None:
            raise ValueError(
                f"The encoder {self.encoder} should not have a LM Head. Please use a model without LM Head"
            )

        # spoken tagging
        self.dropout = torch.nn.Dropout(0.3)
        # 0: "O", 1: "B", 2: "I"
        self.spoken_tagging_classifier = torch.nn.Linear(config.encoder.hidden_size, 3)

        # tie encoder, decoder weights if config set accordingly
        self.tie_weights()

    @classmethod
    def from_encoder_decoder_pretrained(
            cls,
            encoder_pretrained_model_name_or_path: str = None,
            decoder_pretrained_model_name_or_path: str = None,
            *model_args,
            **kwargs
    ) -> PreTrainedModel:

        kwargs_encoder = {
            argument[len("encoder_"):]: value for argument, value in kwargs.items() if argument.startswith("encoder_")
        }

        kwargs_decoder = {
            argument[len("decoder_"):]: value for argument, value in kwargs.items() if argument.startswith("decoder_")
        }

        # remove encoder, decoder kwargs from kwargs
        for key in kwargs_encoder.keys():
            del kwargs["encoder_" + key]
        for key in kwargs_decoder.keys():
            del kwargs["decoder_" + key]

        # Load and initialize the encoder and decoder
        # The distinction between encoder and decoder at the model level is made
        # by the value of the flag `is_decoder` that we need to set correctly.
        encoder = kwargs_encoder.pop("model", None)
        if encoder is None:
            if encoder_pretrained_model_name_or_path is None:
                raise ValueError(
                    "If `encoder_model` is not defined as an argument, a `encoder_pretrained_model_name_or_path` has "
                    "to be defined."
                )

            if "config" not in kwargs_encoder:
                encoder_config = AutoConfig.from_pretrained(encoder_pretrained_model_name_or_path)
                if encoder_config.is_decoder is True or encoder_config.add_cross_attention is True:
                    logger.info(
                        f"Initializing {encoder_pretrained_model_name_or_path} as a encoder model "
                        "from a decoder model. Cross-attention and casual mask are disabled."
                    )
                    encoder_config.is_decoder = False
                    encoder_config.add_cross_attention = False

                kwargs_encoder["config"] = encoder_config

            encoder = AutoModel.from_pretrained(encoder_pretrained_model_name_or_path, *model_args,
                                                **kwargs_encoder)

        decoder = kwargs_decoder.pop("model", None)
        if decoder is None:
            if decoder_pretrained_model_name_or_path is None:
                raise ValueError(
                    "If `decoder_model` is not defined as an argument, a `decoder_pretrained_model_name_or_path` has "
                    "to be defined."
                )

            if "config" not in kwargs_decoder:
                decoder_config = DecoderSpokenNormConfig.from_pretrained(decoder_pretrained_model_name_or_path)
                if decoder_config.is_decoder is False or decoder_config.add_cross_attention is False:
                    logger.info(
                        f"Initializing {decoder_pretrained_model_name_or_path} as a decoder model. "
                        f"Cross attention layers are added to {decoder_pretrained_model_name_or_path} "
                        f"and randomly initialized if {decoder_pretrained_model_name_or_path}'s architecture allows for "
                        "cross attention layers."
                    )
                    decoder_config.is_decoder = True
                    decoder_config.add_cross_attention = True

                kwargs_decoder["config"] = decoder_config

            if kwargs_decoder["config"].is_decoder is False or kwargs_decoder["config"].add_cross_attention is False:
                logger.warning(
                    f"Decoder model {decoder_pretrained_model_name_or_path} is not initialized as a decoder. "
                    f"In order to initialize {decoder_pretrained_model_name_or_path} as a decoder, "
                    "make sure that the attributes `is_decoder` and `add_cross_attention` of `decoder_config` "
                    "passed to `.from_encoder_decoder_pretrained(...)` are set to `True` or do not pass a "
                    "`decoder_config` to `.from_encoder_decoder_pretrained(...)`"
                )

            decoder = DecoderSpokenNorm.from_pretrained(decoder_pretrained_model_name_or_path, **kwargs_decoder)

        # instantiate config with corresponding kwargs
        config = EncoderDecoderSpokenNormConfig.from_encoder_decoder_configs(encoder.config, decoder.config, **kwargs)
        return cls(encoder=encoder, decoder=decoder, config=config)

    def get_encoder(self):
        def forward(input_ids=None,
                    attention_mask=None,
                    bias_input_ids=None,
                    bias_attention_mask=None,
                    return_dict=True,
                    output_attentions=False,
                    output_hidden_states=False,
                    word_src_lengths=None,
                    spoken_idx=None,
                    **kwargs_encoder):
            encoder_outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                inputs_embeds=None,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                **kwargs_encoder,
            )
            encoder_outputs.word_src_lengths = word_src_lengths
            encoder_outputs.spoken_tagging_output = self.spoken_tagging_classifier(self.dropout(encoder_outputs[0]))
            if spoken_idx is not None:
                encoder_outputs.spoken_idx = spoken_idx
            else:
                pass

            encoder_bias_outputs = self.forward_bias(bias_input_ids,
                                                     bias_attention_mask,
                                                     output_attentions=output_attentions,
                                                     return_dict=return_dict,
                                                     output_hidden_states=output_hidden_states,
                                                     **kwargs_encoder)
            # d = {
            #     "encoder_bias_outputs": None,
            #     "bias_attention_mask": None,
            #     "last_hidden_state": None,
            #     "pooler_output": None
            #
            # }
            # encoder_bias_outputs = namedtuple('Struct', d.keys())(*d.values())
            # if bias_input_ids is not None:
            #     encoder_bias_outputs = self.encoder(
            #         input_ids=bias_input_ids,
            #         attention_mask=bias_attention_mask,
            #         inputs_embeds=None,
            #         output_attentions=output_attentions,
            #         output_hidden_states=output_hidden_states,
            #         return_dict=return_dict,
            #         **kwargs_encoder,
            #     )
            #     encoder_bias_outputs.bias_attention_mask = bias_attention_mask
            return encoder_outputs, encoder_bias_outputs

        return forward

    def forward_bias(self,
                     bias_input_ids,
                     bias_attention_mask,
                     output_attentions=False,
                     return_dict=True,
                     output_hidden_states=False,
                     **kwargs_encoder):
        d = {
            "encoder_bias_outputs": None,
            "bias_attention_mask": None,
            "last_hidden_state": None,
            "pooler_output": None

        }
        encoder_bias_outputs = namedtuple('Struct', d.keys())(*d.values())
        if bias_input_ids is not None:
            encoder_bias_outputs = self.encoder(
                input_ids=bias_input_ids,
                attention_mask=bias_attention_mask,
                inputs_embeds=None,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                **kwargs_encoder,
            )
            encoder_bias_outputs.bias_attention_mask = bias_attention_mask
        return encoder_bias_outputs

    def _prepare_encoder_decoder_kwargs_for_generation(
            self, input_ids: torch.LongTensor, model_kwargs, model_input_name
    ) -> Dict[str, Any]:
        if "encoder_outputs" not in model_kwargs:
            # retrieve encoder hidden states
            encoder = self.get_encoder()
            encoder_kwargs = {
                argument: value
                for argument, value in model_kwargs.items()
                if not (argument.startswith("decoder_") or argument.startswith("cross_attn"))
            }
            encoder_outputs, encoder_bias_outputs = encoder(input_ids, return_dict=True, **encoder_kwargs)
            model_kwargs["encoder_outputs"]: ModelOutput = encoder_outputs
            model_kwargs["encoder_bias_outputs"]: ModelOutput = encoder_bias_outputs

        return model_kwargs

    def _prepare_decoder_input_ids_for_generation(
            self,
            batch_size: int,
            decoder_start_token_id: int = None,
            bos_token_id: int = None,
            model_kwargs: Optional[Dict[str, torch.Tensor]] = None,
    ) -> torch.LongTensor:

        if model_kwargs is not None and "decoder_input_ids" in model_kwargs:
            return model_kwargs.pop("decoder_input_ids")
        else:
            decoder_start_token_id = self._get_decoder_start_token_id(decoder_start_token_id, bos_token_id)
            num_spoken_phrases = (model_kwargs['encoder_outputs'].spoken_idx >= 0).view(-1).sum()
            return torch.ones((num_spoken_phrases, 1), dtype=torch.long, device=self.device) * decoder_start_token_id

    def prepare_inputs_for_generation(
            self, input_ids, past=None, attention_mask=None, use_cache=None, encoder_outputs=None, **kwargs
    ):
        decoder_inputs = self.decoder.prepare_inputs_for_generation(input_ids, past=past)
        decoder_attention_mask = decoder_inputs["attention_mask"] if "attention_mask" in decoder_inputs else None
        input_dict = {
            "attention_mask": attention_mask,
            "decoder_attention_mask": decoder_attention_mask,
            "decoder_input_ids": decoder_inputs["input_ids"],
            "encoder_outputs": encoder_outputs,
            "encoder_bias_outputs": kwargs["encoder_bias_outputs"],
            "past_key_values": decoder_inputs["past_key_values"],
            "use_cache": use_cache,
        }
        return input_dict

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            decoder_input_ids=None,
            bias_input_ids=None,
            bias_attention_mask=None,
            labels_bias=None,
            decoder_attention_mask=None,
            encoder_outputs=None,
            encoder_bias_outputs=None,
            past_key_values=None,
            inputs_embeds=None,
            decoder_inputs_embeds=None,
            labels=None,
            use_cache=None,
            spoken_label=None,
            word_src_lengths=None,
            word_tgt_lengths=None,
            spoken_idx=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
            inputs_length=None,
            outputs=None,
            outputs_length=None,
            text=None,
            **kwargs,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        kwargs_encoder = {argument: value for argument, value in kwargs.items() if not argument.startswith("decoder_")}

        kwargs_decoder = {
            argument[len("decoder_"):]: value for argument, value in kwargs.items() if argument.startswith("decoder_")
        }
        spoken_tagging_output = None
        if encoder_outputs is None:
            encoder_outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                **kwargs_encoder,
            )
            spoken_tagging_output = self.spoken_tagging_classifier(self.dropout(encoder_outputs[0]))
        # else:
        #     word_src_lengths = encoder_outputs.word_src_lengths
        #     spoken_tagging_output = encoder_outputs.spoken_tagging_output

        if encoder_bias_outputs is None:
            encoder_bias_outputs = self.encoder(
                input_ids=bias_input_ids,
                attention_mask=bias_attention_mask,
                inputs_embeds=inputs_embeds,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                **kwargs_encoder,
            )
            encoder_bias_outputs.bias_attention_mask = bias_attention_mask

        encoder_hidden_states = encoder_outputs[0]

        if spoken_idx is not None:
            encoder_hidden_states, attention_mask = collect_spoken_phrases_features(encoder_hidden_states,
                                                                                    word_src_lengths,
                                                                                    spoken_idx)

            decoder_input_ids, labels, labels_bias = collect_spoken_phrases_labels(decoder_input_ids,
                                                                                   labels, labels_bias,
                                                                                   word_tgt_lengths,
                                                                                   spoken_idx)
        # optionally project encoder_hidden_states
        if (
                self.encoder.config.hidden_size != self.decoder.config.hidden_size
                and self.decoder.config.cross_attention_hidden_size is None
        ):
            encoder_hidden_states = self.enc_to_dec_proj(encoder_hidden_states)

        # Decode
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_bias_pooling=encoder_bias_outputs.pooler_output,
            # encoder_bias_hidden_states=encoder_bias_outputs[0],
            encoder_bias_hidden_states=encoder_bias_outputs.last_hidden_state,
            bias_attention_mask=encoder_bias_outputs.bias_attention_mask,
            encoder_attention_mask=attention_mask,
            inputs_embeds=decoder_inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            use_cache=use_cache,
            past_key_values=past_key_values,
            return_dict=return_dict,
            labels_bias=labels_bias,
            **kwargs_decoder,
        )

        if self.decoder.config.add_cross_attention:
            import debug_cross_attention
            if debug_cross_attention.is_debug:
                debug_cross_attention.add_cross_attention(decoder_input_ids, decoder_outputs)
                debug_cross_attention.add_bias_attention_values(decoder_outputs.bias_indicate_output)

        # Compute loss independent from decoder (as some shift the logits inside them)
        loss = None
        if labels is not None:
            logits = decoder_outputs.logits if return_dict else decoder_outputs[1]
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.reshape(-1, self.decoder.config.vocab_size), labels.view(-1))
            loss = loss + decoder_outputs.loss

        if spoken_label is not None:
            loss_fct = CrossEntropyLoss()
            spoken_tagging_loss = loss_fct(spoken_tagging_output.reshape(-1, 3), spoken_label.view(-1))
            loss = loss + spoken_tagging_loss

        if not return_dict:
            if loss is not None:
                return (loss,) + decoder_outputs + encoder_outputs
            else:
                return decoder_outputs + encoder_outputs

        return SpokenNormOutput(
            loss=loss,
            logits=decoder_outputs.logits,
            logits_spoken_tagging=spoken_tagging_output,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            # encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            # encoder_hidden_states=encoder_outputs.hidden_states,
            # encoder_attentions=encoder_outputs.attentions,
        )


class DecoderSpokenNorm(RobertaForCausalLM):
    config_class = DecoderSpokenNormConfig

    # Copied from transformers.models.bert.modeling_bert.BertModel.__init__ with Bert->Roberta
    def __init__(self, config):
        super().__init__(config)
        self.dense_query_copy = torch.nn.Linear(config.hidden_size, config.hidden_size)
        self.mem_no_entry = Parameter(torch.randn(config.hidden_size).unsqueeze(0))
        self.bias_attention_layer = MultiHeadAttention(config.hidden_size)
        self.copy_attention_layer = MultiHeadAttention(config.hidden_size)

    def forward_bias_attention(self, query, values, values_mask):
        """
        :param query: batch * output_steps * hidden_state
        :param values: batch * output_steps * max_bias_steps * hidden_state
        :param values_mask: batch * output_steps * max_bias_steps
        :return: batch * output_steps * hidden_state
        """
        batch, output_steps, hidden_state = query.size()
        _, _, max_bias_steps, _ = values.size()

        query = query.view(batch * output_steps, 1, hidden_state)
        values = values.view(-1, max_bias_steps, hidden_state)
        values_mask = 1 - values_mask.view(-1, max_bias_steps)
        result_attention, attention_score = self.bias_attention_layer(query=query,
                                                                      key=values,
                                                                      value=values,
                                                                      mask=values_mask.bool())
        result_attention = result_attention.squeeze(1).view(batch, output_steps, hidden_state)
        return result_attention

    def forward_copy_attention(self, query, values, values_mask):
        """
        :param query: batch * output_steps * hidden_state
        :param values: batch * max_encoder_steps * hidden_state
        :param values_mask: batch * output_steps * max_encoder_steps
        :return: batch * output_steps * hidden_state
        """
        dot_attn_score = torch.bmm(query, values.transpose(2, 1))
        attn_mask = (1 - values_mask.clone().unsqueeze(1)).bool()
        dot_attn_score.masked_fill_(attn_mask, -float('inf'))
        dot_attn_score = torch.softmax(dot_attn_score, dim=-1)
        result_attention = torch.bmm(dot_attn_score, values)
        return result_attention

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            encoder_bias_pooling=None,
            encoder_bias_hidden_states=None,
            bias_attention_mask=None,
            inputs_embeds=None,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            labels=None,
            labels_bias=None,
            past_key_values=None,
            use_cache=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        if labels is not None:
            use_cache = False

        # attention with input encoded
        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # Query for bias
        sequence_output = outputs[0]
        bias_indicate_output = None

        # output copy attention
        query_copy = torch.relu(self.dense_query_copy(sequence_output))
        sequence_atten_copy_output = self.forward_copy_attention(query_copy,
                                                                 encoder_hidden_states,
                                                                 encoder_attention_mask)

        if encoder_bias_pooling is not None:

            # Make bias features
            encoder_bias_pooling = torch.cat([self.mem_no_entry, encoder_bias_pooling], dim=0)
            mem_no_entry_feature = torch.zeros_like(encoder_bias_hidden_states[0]).unsqueeze(0)
            mem_no_entry_mask = torch.ones_like(bias_attention_mask[0]).unsqueeze(0)
            encoder_bias_hidden_states = torch.cat([mem_no_entry_feature, encoder_bias_hidden_states], dim=0)
            bias_attention_mask = torch.cat([mem_no_entry_mask, bias_attention_mask], dim=0)

            # Compute ranking score
            b, s, h = sequence_output.size()
            bias_ranking_score = sequence_output.view(b * s, h).mm(encoder_bias_pooling.T)
            bias_ranking_score = bias_ranking_score.view(b, s, encoder_bias_pooling.size(0))

            # teacher force with bias label
            if not self.training:
                bias_indicate_output = torch.argmax(bias_ranking_score, dim=-1)
            else:
                if random.random() < 0.5:
                    bias_indicate_output = labels_bias.clone()
                    bias_indicate_output[torch.where(bias_indicate_output < 0)] = 0
                else:
                    bias_indicate_output = torch.argmax(bias_ranking_score, dim=-1)

            # Bias encoder hidden state
            _, max_len, _ = encoder_bias_hidden_states.size()
            bias_encoder_hidden_states = torch.index_select(input=encoder_bias_hidden_states,
                                                            dim=0,
                                                            index=bias_indicate_output.view(b * s)).view(b, s, max_len,
                                                                                                         h)
            bias_encoder_attention_mask = torch.index_select(input=bias_attention_mask,
                                                             dim=0,
                                                             index=bias_indicate_output.view(b * s)).view(b, s, max_len)

            sequence_atten_bias_output = self.forward_bias_attention(sequence_output,
                                                                     bias_encoder_hidden_states,
                                                                     bias_encoder_attention_mask)

            # Find output words
            prediction_scores = self.lm_head(sequence_output + sequence_atten_bias_output + sequence_atten_copy_output)
        else:
            prediction_scores = self.lm_head(sequence_output + sequence_atten_copy_output)

        # run attention with bias

        bias_ranking_loss = None
        if labels_bias is not None:
            loss_fct = CrossEntropyLoss()
            bias_ranking_loss = loss_fct(bias_ranking_score.view(-1, encoder_bias_pooling.size(0)),
                                         labels_bias.view(-1))

        if not return_dict:
            output = (prediction_scores,) + outputs[2:]
            return ((bias_ranking_loss,) + output) if bias_ranking_loss is not None else output

        result = CausalLMOutputWithCrossAttentions(
            loss=bias_ranking_loss,
            logits=prediction_scores,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            cross_attentions=outputs.cross_attentions,
        )

        result.bias_indicate_output = bias_indicate_output

        return result


def download_tokenizer_files():
    resources = ['envibert_tokenizer.py', 'dict.txt', 'sentencepiece.bpe.model']
    for item in resources:
        if not os.path.exists(os.path.join(cache_dir, item)):
            tmp_file = hf_bucket_url(model_name, filename=item)
            tmp_file = cached_path(tmp_file, cache_dir=cache_dir)
            os.rename(tmp_file, os.path.join(cache_dir, item))


def init_tokenizer():
    download_tokenizer_files()
    tokenizer = SourceFileLoader("envibert.tokenizer",
                                 os.path.join(cache_dir,
                                              'envibert_tokenizer.py')).load_module().RobertaTokenizer(cache_dir)
    tokenizer.model_input_names = ["input_ids",
                                   "attention_mask",
                                   "bias_input_ids",
                                   "bias_attention_mask",
                                   "labels"
                                   "labels_bias"]
    return tokenizer


def init_model():
    download_tokenizer_files()
    tokenizer = SourceFileLoader("envibert.tokenizer",
                                 os.path.join(cache_dir,
                                              'envibert_tokenizer.py')).load_module().RobertaTokenizer(cache_dir)
    tokenizer.model_input_names = ["input_ids",
                                   "attention_mask",
                                   "bias_input_ids",
                                   "bias_attention_mask",
                                   "labels"
                                   "labels_bias"]
    # set encoder decoder tying to True
    roberta_shared = EncoderDecoderSpokenNorm.from_encoder_decoder_pretrained(model_name,
                                                                              model_name,
                                                                              tie_encoder_decoder=False)

    # set special tokens
    roberta_shared.config.decoder_start_token_id = tokenizer.bos_token_id
    roberta_shared.config.eos_token_id = tokenizer.eos_token_id
    roberta_shared.config.pad_token_id = tokenizer.pad_token_id

    # sensible parameters for beam search
    # set decoding params
    roberta_shared.config.max_length = 500
    roberta_shared.config.early_stopping = True
    roberta_shared.config.no_repeat_ngram_size = 3
    roberta_shared.config.length_penalty = 2.0
    roberta_shared.config.num_beams = 1
    roberta_shared.config.vocab_size = roberta_shared.config.encoder.vocab_size

    return roberta_shared, tokenizer
