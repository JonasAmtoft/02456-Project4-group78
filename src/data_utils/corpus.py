import os
import shutil
from datasets import Dataset, DatasetDict, load_from_disk
from src.data_utils.download_retrieval_datasets import get_fiqa_dataset, get_squad_dataset
import random

class Corpus:
    def __init__(self, dataset_name: str, base_dir: str):
        self.dataset_name = dataset_name
        if not os.path.exists(f'{base_dir}/{self.dataset_name}'):
            raw_data_dir = f'{base_dir}/raw'
            os.makedirs(raw_data_dir, exist_ok=True)
            # load and process data
            self.ds = self.__load_data(base_dir=raw_data_dir)
            self.__rename_columns()
            self.relevant_ids = self.__assign_ids()
            os.makedirs(f'{base_dir}/{self.dataset_name}', exist_ok=True)
            DatasetDict(self.ds).save_to_disk(f'{base_dir}/{self.dataset_name}')
            print(f'Saved {self.dataset_name} dataset to path {base_dir}/{self.dataset_name}.')
            shutil.rmtree(f'{base_dir}/raw')
        else:
            self.ds = load_from_disk(f'{base_dir}/{self.dataset_name}')
            self.relevant_ids = {query_id: document_ids
                                 for split in self.ds.values()
                                 for query_id, document_ids in zip(split['query_id'], split['document_id'])}


    @property
    def train(self):
        return self.ds.get('train', None)
    
    @property
    def val(self):
        return self.ds.get('validation', None)

    @property
    def test(self):
        return self.ds.get('test', None)

    @property
    def all_documents(self):
        train_text, train_ids = list(self.train['text']), list(self.train['document_id'])
        val_text, val_ids =     list(self.val['text']), list(self.val['document_id'])
        
        all_text = train_text + val_text
        all_ids = train_ids   + val_ids

        if self.test is not None:
            test_text, test_ids = list(self.test['text']), list(self.test['document_id'])
            all_text += test_text
            all_ids  += test_ids

        seen = set()
        texts = []
        ids = []
        for row_text, row_ids in zip(all_text,all_ids):
            for t, i in zip(row_text,row_ids):
                if i not in seen:
                    seen.add(i)
                    texts.append(t)
                    ids.append(i)

        return Dataset.from_dict({'text': [[t] for t in texts], 'document_id': [[i] for i in ids]})


    def __load_data(self, base_dir) -> dict[Dataset]:
        assert self.dataset_name in ['squad', 'fiqa'], f'Dataset {self.dataset_name} not supported. Options are ["squad", "fiqa"]'
        
        if self.dataset_name == 'fiqa':
            ds = get_fiqa_dataset(base_dir)

        elif self.dataset_name == 'squad':
            ds = get_squad_dataset(base_dir)
            contexts = sorted(set(ds['validation']['context']))
            random.Random(0).shuffle(contexts)
            test_contexts = set(contexts[:len(contexts) // 2])
            test = ds['validation'].filter(lambda r: r['context'] in test_contexts)
            validation = ds['validation'].filter(lambda r: r['context'] not in test_contexts)

            ds = {'train': ds['train'], 'validation': validation, 'test': test}

        return {key: ds[key] for key in ds.keys()}

    def __rename_columns(self) -> dict[Dataset]:
        column_mapping = {'fiqa':  'ground_truths',
                          'squad': 'context'
                         }
        
        for key in self.ds:
            self.ds[key] = self.ds[key].rename_column(column_mapping[self.dataset_name], 'text')
            if self.dataset_name == 'squad':
                self.ds[key] = self.ds[key].map(lambda r: {'text': [r['text']]})

        return self.ds

    def __assign_ids(self):
        q_id_counter = 0
        doc_id_counter = 0
        relevant_document_ids = {}
        text_to_id = {}

        for key in self.ds:
            query_ids = []
            document_ids = []
            texts = self.ds[key]['text']
            for i in range(len(self.ds[key])):
                query_ids.append(f'q{q_id_counter}')
                documents = texts[i]
                relevant_ids = []
                for doc in documents:
                    if doc in text_to_id:
                        relevant_ids.append(text_to_id[doc])
                    else:
                        new_id = f'doc{doc_id_counter}'
                        text_to_id[doc] = new_id
                        relevant_ids.append(new_id)
                        doc_id_counter += 1
                
                document_ids.append(relevant_ids)
                
                relevant_document_ids[f'q{q_id_counter}'] = relevant_ids
                q_id_counter += 1
                
            self.ds[key] = self.ds[key].add_column('query_id', query_ids)
            self.ds[key] = self.ds[key].add_column('document_id', document_ids)

        return relevant_document_ids
            