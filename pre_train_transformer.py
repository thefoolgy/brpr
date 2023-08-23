from transformers import BertTokenizer, BertForMaskedLM


class ProteinBertTokenizer(BertTokenizer):
    def __init__(self, tokens, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._tokenizer = ProteinTokenizer(tokens)

    def _tokenize(self, text):
        return self._tokenizer.encode(text)

    def _convert_token_to_id(self, token):
        return self._tokenizer.token2idx.get(token, self.unk_token_id)

    def _convert_id_to_token(self, index):
        return self._tokenizer.idx2token.get(index, self.unk_token)


class ProteinTokenizer:
    def __init__(self, tokens):
        self.tokens = tokens
        self.token2idx = {token: idx for idx, token in enumerate(tokens)}
        self.idx2token = {idx: token for token, idx in self.token2idx.items()}

    def encode(self, sequence):
        return [self.token2idx[token] for token in sequence if token in self.tokens]

    def decode(self, token_ids):
        return ''.join([self.idx2token[token_id] for token_id in token_ids])


tokens = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']
tokenizer = ProteinBertTokenizer(tokens=tokens, vocab_file=None)
model = BertForMaskedLM.from_pretrained('bert-base-uncased')
