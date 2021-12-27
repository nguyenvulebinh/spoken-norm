import torch
import matplotlib.pyplot as plt
import numpy as np

is_debug = False
bias_input = None
input_token = None
output_token = []
tokenizer = None
cross_attention_values = []
decoder_inputs_values = []
bias_attention_values = []
layer_print = -1
head_print = 0


def add_bias_attention_values(bias_ranking):
    if bias_ranking is None:
        bias_attention_values.append(0)
    else:
        bias_attention_values.append(bias_ranking.squeeze().numpy().tolist())


def add_cross_attention(decoder_inputs, decoder_outputs):
    decoder_inputs_values.append(decoder_inputs.squeeze().numpy().tolist())
    cross_attention_values.append(decoder_outputs.cross_attentions)
    # print(torch.softmax(decoder_outputs.cross_attentions[-1].squeeze(), dim=-1)[0].topk(2).indices)


def print_cross_attention(output_ids):
    global output_token
    map_attention_by_head = dict({})
    for bias, de_in, cross in zip(bias_attention_values, decoder_inputs_values[1:] + [tokenizer.eos_token_id], cross_attention_values):
        output_token.append('[{}] {}'.format(bias_input[bias], tokenizer.decode([de_in])))
        multi_head = cross[layer_print].squeeze()
        for i, head in enumerate(multi_head):
            if map_attention_by_head.get(i, None) is None:
                map_attention_by_head[i] = []
            cross_attention = torch.softmax(head, dim=-1).numpy().tolist()
            map_attention_by_head[i].append(cross_attention)

    fig, ax_list = plt.subplots(6, figsize=(20, 60))

    for head, ax in zip(map_attention_by_head.keys(), ax_list):
        ax.matshow(np.array(map_attention_by_head[head]), cmap='Blues')
        ax.set_xticks(list(range(len(input_token))))
        ax.set_xticklabels(input_token, rotation_mode="anchor", rotation=45)

        ax.set_yticks(list(range(len(output_token))))
        ax.set_yticklabels(output_token)

        ax.grid()
    fig.show()
    # map_attention_by_head[head] =

    # print(map_attention_by_head)
