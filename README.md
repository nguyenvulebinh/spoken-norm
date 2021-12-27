# Transformation spoken text to written text

![Model architecture](./spoken_norm_model.svg)

# Infer model

- Play around at [Hugginface Space](https://huggingface.co/spaces/nguyenvulebinh/spoken-norm)


```python
import torch
import model_handling
from data_handling import DataCollatorForNormSeq2Seq
from model_handling import EncoderDecoderSpokenNorm
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
```

## Init tokenizer and model


```python
tokenizer = model_handling.init_tokenizer()
model = EncoderDecoderSpokenNorm.from_pretrained('nguyenvulebinh/spoken-norm', cache_dir=model_handling.cache_dir)
data_collator = DataCollatorForNormSeq2Seq(tokenizer)
```

## Infer sample


```python
bias_list = ['scotland', 'covid', 'delta', 'beta']
input_str = 'ngày hai tám tháng tư cô vít bùng phát ở sờ cốt lờn chiếm tám mươi phần trăm là biến chủng đen ta và bê ta'
```


```python
inputs = tokenizer([input_str])
input_ids = inputs['input_ids']
attention_mask = inputs['attention_mask']
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
```

## Format input text **with** bias phrases


```python
outputs = model.generate(**inputs, output_attentions=True, num_beams=1, num_return_sequences=1)

for output in outputs.cpu().detach().numpy().tolist():
    # print('\n', tokenizer.decode(output, skip_special_tokens=True).split(), '\n')
    print(tokenizer.sp_model.DecodePieces(tokenizer.decode(output, skip_special_tokens=True).split()))
```

    28/4 covid bùng phát ở scotland chiếm 80 % là biến chủng delta và beta


## Format input text **without** bias phrases


```python
outputs = model.generate(**{
    "input_ids": torch.tensor(input_ids),
    "attention_mask": torch.tensor(attention_mask),
    "bias_input_ids": None,
    "bias_attention_mask": None,
}, output_attentions=True, num_beams=1, num_return_sequences=1)

for output in outputs.cpu().detach().numpy().tolist():
    # print('\n', tokenizer.decode(output, skip_special_tokens=True).split(), '\n')
    print(tokenizer.sp_model.DecodePieces(tokenizer.decode(output, skip_special_tokens=True).split()))
```

    28/4 cô vít bùng phát ở sờ cốt lờn chiếm 80 % là biến chủng đen ta và bê ta
