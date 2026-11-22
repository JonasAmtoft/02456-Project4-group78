import os
import argparse
import torch
import matplotlib.pyplot as plt
from omegaconf import OmegaConf
from transformers import AutoTokenizer

from src.data_utils.corpus import Corpus
from src.retrievers.models.encoder import TransformerEncoder
from src.retrievers.models.bert_encoder import BertStyleEncoder

MODELS = {
    'squad': {'baseline': ('encoder', 'checkpoints/squad_scratch'),
              'bert_small': ('bert_small', 'checkpoints/squad_bert_small')},
    'fiqa': {'bert_small': ('bert_small', 'checkpoints/bert_small')},
}
NAMES = {'baseline': 'Baseline', 'bert_small': 'BERT-Small'}

plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif']


def load_encoder(model_cfg_name: str, weights_path: str) -> torch.nn.Module:
    cfg = OmegaConf.load(f'config/model/{model_cfg_name}.yaml')
    encoder_class = BertStyleEncoder if model_cfg_name == 'bert_small' else TransformerEncoder
    encoder = encoder_class(n_layers=cfg.n_layers, n_heads=cfg.n_heads, emb_dim=cfg.emb_dim,
                            hidden_dim=cfg.hidden_dim, feed_forward_dim=cfg.feed_forward_dim,
                            pooling_type=cfg.pooling_type, dropout=cfg.dropout)
    encoder.load_state_dict(torch.load(weights_path, map_location='cpu'))
    return encoder.eval()


def load_example(dataset: str, index: int) -> tuple[str, str]:
    data_dir = 'data' if os.path.exists(f'data/{dataset}') else 'retrieval_data'
    corpus = Corpus(dataset_name=dataset, base_dir=data_dir)
    row = corpus.test[index]
    return row['question'], row['text'][0]


@torch.no_grad()
def get_attention_maps(encoder, tokenizer, text: str, max_len: int):
    tokenizer_args = {"padding": True, "truncation": True, "return_tensors": "pt"}
    batch_text = tokenizer(text,max_length=max_len,**tokenizer_args)
    tokens = tokenizer.convert_ids_to_tokens(batch_text["input_ids"][0])
    encoder(batch_text["input_ids"],batch_text["attention_mask"])

    maps = torch.stack([layer.attention.attn[0] for layer in encoder.layers])
    return tokens, maps


def plot_head(attention, tokens: list[str], title: str, path: str):
    fig, ax = plt.subplots(figsize=(3.6, 3.4))
    im = ax.imshow(attention.cpu().numpy(), cmap='viridis')
    ax.set_xticks(range(len(tokens)))
    ax.set_yticks(range(len(tokens)))
    ax.set_xticklabels(tokens, rotation=90, fontsize=4.5)
    ax.set_yticklabels(tokens, fontsize=4.5)
    ax.tick_params(length=1.5, width=0.5)
    ax.set_xlabel('Attended token', fontsize=7)
    ax.set_ylabel('Token', fontsize=7)
    ax.set_title(title, fontsize=8)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cbar.ax.tick_params(labelsize=5)
    cbar.set_label('Attention weight', fontsize=6)
    fig.savefig(path, bbox_inches='tight')
    plt.close(fig)


def plot_all_heads(maps, path: str):
    n_layers, n_heads = maps.shape[:2]
    fig, axes = plt.subplots(n_layers, n_heads, figsize=(7, 3.6), squeeze=False)
    for layer in range(n_layers):
        for head in range(n_heads):
            ax = axes[layer, head]
            ax.imshow(maps[layer, head].cpu().numpy(), cmap='viridis', vmin=0, vmax=1)
            ax.set_xticks([])
            ax.set_yticks([])
            if layer == 0:
                ax.set_title(f'Head {head + 1}', fontsize=7)
            if head == 0:
                ax.set_ylabel(f'Layer {layer + 1}', fontsize=7)
    fig.tight_layout(pad=0.3, w_pad=0.3, h_pad=0.3)
    fig.savefig(path, bbox_inches='tight')
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='squad')
    parser.add_argument('--example', type=int, default=0)
    args = parser.parse_args()

    out_dir = 'outputs/attention'
    os.makedirs(out_dir, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    question, passage = load_example(args.dataset, args.example)

    for model_name, (model_cfg_name, checkpoint_dir) in MODELS[args.dataset].items():
        inputs = {'query': (f'{checkpoint_dir}/query_encoder.pt', question, 64),
                  'doc': (f'{checkpoint_dir}/document_encoder.pt', passage, 48)}
        for side, (weights_path, text, max_len) in inputs.items():
            encoder = load_encoder(model_cfg_name, weights_path)
            tokens, maps = get_attention_maps(encoder, tokenizer, text, max_len)
            name = f'{args.dataset}_{model_name}_{side}_ex{args.example}'
            plot_head(maps[-1, 0], tokens, f'{NAMES[model_name]} (last layer, head 1)',
                      f'{out_dir}/{name}_last_layer.pdf')
            plot_all_heads(maps, f'{out_dir}/{name}_all_heads.pdf')


if __name__ == '__main__':
    main()
