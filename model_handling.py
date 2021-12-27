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
from typing import Dict, Any, Optional
import random
from model_config_handling import EncoderDecoderSpokenNormConfig, DecoderSpokenNormConfig, PretrainedConfig

cache_dir = './cache'
model_name = 'nguyenvulebinh/envibert'

if not os.path.exists(cache_dir):
    os.makedirs(cache_dir)
logger = logging.get_logger(__name__)


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

            return encoder_outputs, encoder_bias_outputs

        return forward

    def _prepare_encoder_decoder_kwargs_for_generation(
            self, input_ids: torch.LongTensor, model_kwargs
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
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
            inputs_length=None,
            outputs=None,
            outputs_length=None,
            **kwargs,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        kwargs_encoder = {argument: value for argument, value in kwargs.items() if not argument.startswith("decoder_")}

        kwargs_decoder = {
            argument[len("decoder_"):]: value for argument, value in kwargs.items() if argument.startswith("decoder_")
        }

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

        if not return_dict:
            if loss is not None:
                return (loss,) + decoder_outputs + encoder_outputs
            else:
                return decoder_outputs + encoder_outputs

        return Seq2SeqLMOutput(
            loss=loss,
            logits=decoder_outputs.logits,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
        )


class DecoderSpokenNorm(RobertaForCausalLM):
    config_class = DecoderSpokenNormConfig

    # Copied from transformers.models.bert.modeling_bert.BertModel.__init__ with Bert->Roberta
    def __init__(self, config):
        super().__init__(config)
        self.dense_query_bias = torch.nn.Linear(config.hidden_size, config.hidden_size)
        self.mem_no_entry = Parameter(torch.randn(config.hidden_size).unsqueeze(0))
        # self.bias_attention_layer = ScaledDotProductAttention(config.hidden_size)
        self.bias_attention_layer = MultiHeadAttention(config.hidden_size)

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

            # random mask input features
            if self.training:
                mask_indicate_output = bias_indicate_output.clone()
                mask_indicate_output[torch.where(mask_indicate_output > 0)] = 1
                if random.random() < 0.5:
                    sequence_output = sequence_output * (1 - mask_indicate_output.unsqueeze(-1).repeat(1, 1, h))

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
            prediction_scores = self.lm_head(sequence_output + sequence_atten_bias_output)
        else:
            prediction_scores = self.lm_head(sequence_output)

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
