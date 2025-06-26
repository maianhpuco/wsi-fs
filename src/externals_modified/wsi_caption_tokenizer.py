import json
import re
from collections import Counter
import os


class Tokenizer(object):
    def __init__(self, args):
        self.ann_path = args.ann_path              # Path to PathText.json
        self.threshold = args.threshold            # Frequency threshold
        self.dataset_name = 'BRCA'                 # Hardcoded for now
        if self.dataset_name == 'BRCA':
            self.clean_report = self.clean_report_brca

        # Build vocab
        self.token2idx, self.idx2token = self.create_vocabulary()

    def create_vocabulary(self):
        total_tokens = []

        # Load JSON annotation file directly
        with open(self.ann_path, 'r') as f:
            annotations = json.load(f)

        for item in annotations:
            caption = item["caption"]
            tokens = self.clean_report(caption).split()
            total_tokens.extend(tokens)

        counter = Counter(total_tokens)
        vocab = [k for k, v in counter.items() if v >= self.threshold] + ['<unk>']
        vocab.sort()

        token2idx, idx2token = {}, {}
        for idx, token in enumerate(vocab):
            token2idx[token] = idx + 1
            idx2token[idx + 1] = token

        return token2idx, idx2token

    def clean_report_brca(self, report):
        report_cleaner = lambda t: (t.replace('\n', ' ')
                                      .replace('  ', ' ')
                                      .replace(' 10. ', ' ')
                                      .replace(' 11. ', ' ')
                                      .replace(' 12. ', ' ')
                                      .replace(' 13. ', ' ')
                                      .replace(' 14.', ' ')
                                      .replace(' 1. ', ' ')
                                      .replace(' 2. ', ' ')
                                      .replace(' 3. ', ' ')
                                      .replace(' 4. ', ' ')
                                      .replace(' 5. ', ' ')
                                      .replace(' 6. ', ' ')
                                      .replace(' 7. ', ' ')
                                      .replace(' 8. ', ' ')
                                      .replace(' 9. ', ' ')
                                      .strip().lower() + ' ').split('. ')
        sent_cleaner = lambda t: re.sub(r'[#,\?;*!^&_+():\-\[\]{}]', '', t.replace('"', '')
                                                                             .replace('\\', '')
                                                                             .replace("'", '')
                                                                             .strip().lower())
        tokens = [sent_cleaner(sent) for sent in report_cleaner(report) if sent_cleaner(sent)]
        return ' . '.join(tokens)

    def get_token_by_id(self, idx):
        return self.idx2token.get(idx, '<unk>')

    def get_id_by_token(self, token):
        return self.token2idx.get(token, self.token2idx['<unk>'])

    def get_vocab_size(self):
        return len(self.token2idx)

    def __call__(self, report):
        tokens = self.clean_report(report).split()
        ids = [self.get_id_by_token(token) for token in tokens]
        ids = [0] + ids + [0]  # add BOS and EOS
        return ids

    def decode(self, ids):
        txt = ''
        for i, idx in enumerate(ids):
            if idx > 0:
                if i >= 1:
                    txt += ' '
                txt += self.idx2token.get(idx, '<unk>')
            else:
                break
        return txt

    def decode_batch(self, ids_batch):
        return [self.decode(ids) for ids in ids_batch]
