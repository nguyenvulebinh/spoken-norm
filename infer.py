from transformers import EncoderDecoderModel, DataCollatorForSeq2Seq
from transformers.file_utils import cached_path, hf_bucket_url
from importlib.machinery import SourceFileLoader
import os
import json
from data_handling import preprocess_function
import torch
import model_handling
from data_handling import DataCollatorForNormSeq2Seq
from model_handling import EncoderDecoderSpokenNorm
import debug_cross_attention

# os.environ["CUDA_VISIBLE_DEVICES"] = "3"

tokenizer = model_handling.init_tokenizer()
model = EncoderDecoderSpokenNorm.from_pretrained('./trained_checkpoints')
data_collator = DataCollatorForNormSeq2Seq(tokenizer)

inputs = tokenizer(['cô vít gây ảnh hưởng lớn tới vin gờ rúp'])
input_ids = inputs['input_ids']
attention_mask = inputs['attention_mask']

# bias_list = []
bias_list = ['covid', 'vingroup', 'google' ]
if len(bias_list) > 0:
    bias = data_collator.encode_list_string(bias_list)
    bias_input_ids = bias['input_ids']
    bias_attention_mask = bias['attention_mask']
else:
    bias_input_ids = None
    bias_attention_mask = None

inputs = {
    "input_ids": torch.tensor(input_ids),
    "attention_mask": torch.tensor(attention_mask),
    "bias_input_ids": bias_input_ids,
    "bias_attention_mask": bias_attention_mask,
}

if debug_cross_attention.is_debug:
    model.decoder.config.add_cross_attention = True
    debug_cross_attention.tokenizer = tokenizer
    debug_cross_attention.input_token = tokenizer.decode(inputs["input_ids"][0]).split()

    debug_cross_attention.bias_input = ['None'] + bias_list


outputs = model.generate(**inputs, output_attentions=True, num_beams=1, num_return_sequences=1)
# print(outputs)

for output in outputs.cpu().detach().numpy().tolist():
    if debug_cross_attention.is_debug:
        debug_cross_attention.print_cross_attention(output)
    print(tokenizer.sp_model.DecodePieces(tokenizer.decode(output, skip_special_tokens=True).split()))
