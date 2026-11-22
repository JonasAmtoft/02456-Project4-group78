import torch
import torch.nn as nn
from transformers import BertModel

from src.retrievers.models.encoder import MultiHeadedAttention, TransformerEncoder


class BertStyleEncoderLayer(nn.Module):
    def __init__(self, emb_dim, n_heads, feed_forward_dim, dropout=0.1):
        super().__init__()
        self.attention = MultiHeadedAttention(n_heads,emb_dim,dropout)
        self.norm1 = nn.LayerNorm(emb_dim,eps=1e-12)
        self.norm2 = nn.LayerNorm(emb_dim,eps=1e-12)
        self.feed_forward = nn.Sequential(nn.Linear(emb_dim,feed_forward_dim), nn.GELU(), nn.Linear(feed_forward_dim,emb_dim))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        x = self.norm1(x + self.dropout(self.attention(x, x, x, mask)))
        x = self.norm2(x + self.dropout(self.feed_forward(x)))
        return x


class BertStyleEncoder(TransformerEncoder):

    def setup_model(self):
        self.embedding = nn.Embedding(self.vocab_size,self.emb_dim)
        self.position_embedding = nn.Embedding(self.max_len, self.emb_dim)
        self.embedding_norm = nn.LayerNorm(self.emb_dim, eps=1e-12)
        self.embedding_dropout = nn.Dropout(self.dropout)
        self.layers = nn.ModuleList([BertStyleEncoderLayer(self.emb_dim, self.n_heads, self.feed_forward_dim,self.dropout) for _ in range(self.n_layers)]) 

    def encode(self, x: torch.tensor, mask: torch.tensor) -> torch.tensor:
        positions = torch.arange(x.size(1), device=x.device)
        x = self.embedding(x) + self.position_embedding(positions)
        x = self.embedding_dropout(self.embedding_norm(x))
        for layer in self.layers:
            x = layer(x, mask)

        return x

    def forward(self, x: torch.tensor, mask: torch.tensor) -> torch.tensor:
        return self.pool(self.encode(x, mask), mask)


LAYER_MAPPING = [
    ('attention.self.query', 'attention.linears.0'),
    ('attention.self.key', 'attention.linears.1'),
    ('attention.self.value', 'attention.linears.2'),
    ('attention.output.dense', 'attention.linears.3'),
    ('attention.output.LayerNorm', 'norm1'),
    ('intermediate.dense', 'feed_forward.0'),
    ('output.dense', 'feed_forward.2'),
    ('output.LayerNorm', 'norm2'),
]


def load_pretrained_bert(model: BertStyleEncoder, name: str = 'google/bert_uncased_L-4_H-512_A-8'):
    hf =BertModel.from_pretrained(name).state_dict()

    ours = {
        'embedding.weight': hf['embeddings.word_embeddings.weight'] + hf['embeddings.token_type_embeddings.weight'][0],
        'position_embedding.weight': hf['embeddings.position_embeddings.weight'],
        'embedding_norm.weight': hf['embeddings.LayerNorm.weight'],
        'embedding_norm.bias': hf['embeddings.LayerNorm.bias'],
    }
    for i in range(model.n_layers):
        for bert_name, our_name in LAYER_MAPPING:
            for p in ('weight', 'bias'):
                ours[f'layers.{i}.{our_name}.{p}'] = hf[f'encoder.layer.{i}.{bert_name}.{p}']

    model.load_state_dict(ours, strict=True)
    return model
