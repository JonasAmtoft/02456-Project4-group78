import os
import hydra
from transformers import AutoTokenizer

from src.data_utils.corpus import Corpus
from src.data_utils.dpr_data import make_dataloader
from src.retrievers.models.bert_encoder import BertStyleEncoder, load_pretrained_bert
from src.training.trainer import Trainer

@hydra.main(config_path='config', config_name='train_bert.yaml', version_base=None)
def main(cfg):
    data_dir = (
        f'{cfg.base_path}/data'
        if os.path.exists(f'{cfg.base_path}/data/{cfg.dataset.name}')
        else f'{cfg.base_path}/retrieval_data'
    )
    corpus = Corpus(dataset_name=cfg.dataset.name, base_dir=data_dir)

    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

    train_loader = make_dataloader(split=corpus.train,
                                   tokenizer=tokenizer,
                                   batch_size=cfg.training.batch_size,
                                   shuffle=True,
                                   max_query_len=cfg.training.max_query_len,
                                   max_doc_len=cfg.training.max_doc_len)

    val_loader = make_dataloader(split=corpus.val,
                                   tokenizer=tokenizer,
                                   batch_size=cfg.training.batch_size,
                                   shuffle=True,
                                   max_query_len=cfg.training.max_query_len,
                                   max_doc_len=cfg.training.max_doc_len)

    dropout = cfg.model.dropout
    query_encoder = BertStyleEncoder(n_layers=cfg.model.n_layers,
                                       n_heads=cfg.model.n_heads,
                                       emb_dim=cfg.model.emb_dim,
                                       hidden_dim=cfg.model.hidden_dim,
                                       feed_forward_dim=cfg.model.feed_forward_dim,
                                       pooling_type=cfg.model.pooling_type,
                                       dropout=dropout)

    document_encoder = BertStyleEncoder(n_layers=cfg.model.n_layers,
                                          n_heads=cfg.model.n_heads,
                                          emb_dim=cfg.model.emb_dim,
                                          hidden_dim=cfg.model.hidden_dim,
                                          feed_forward_dim=cfg.model.feed_forward_dim,
                                          pooling_type=cfg.model.pooling_type,
                                          dropout=dropout)

    load_pretrained_bert(query_encoder, cfg.model.pretrained_name)
    load_pretrained_bert(document_encoder, cfg.model.pretrained_name)

    trainer = Trainer(lr=cfg.training.lr, weight_decay=cfg.training.weight_decay, device=cfg.device)
    train_losses, val_losses = trainer.train(query_encoder=query_encoder,
                                                 document_encoder=document_encoder,
                                                 train_loader=train_loader,
                                                 val_loader=val_loader,
                                                 num_epochs=cfg.training.num_epochs,
                                                 checkpoint_dir=cfg.training.checkpoint_dir)

    print(f'Final train loss: {train_losses[-1]:.4f}   final val loss: {val_losses[-1]:.4f}')


if __name__ == '__main__':
    main()
