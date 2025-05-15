
import torch
import torch.nn as nn
from torchcrf import CRF

class BiGRUCRFClass(nn.Module):
    def __init__(self, embedding_matrix, hidden_dim, output_dim):
        super(BiGRUCRFClass, self).__init__()
        vocab_size, embed_dim = embedding_matrix.shape
        self.embed = nn.Embedding.from_pretrained(
            torch.tensor(embedding_matrix).float(),
            freeze=False,
            padding_idx=0
        )
        self.bigru = nn.GRU(embed_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(hidden_dim * 2, output_dim)
        self.crf = CRF(output_dim, batch_first=True)

    def forward(self, x, labels=None, mask=None):
        x = self.embed(x)
        x, _ = self.bigru(x)
        x = self.dropout(x)
        emissions = self.fc(x)
        if labels is not None:
            return -self.crf(emissions, labels, mask=mask, reduction='mean')
        else:
            return self.crf.decode(emissions, mask=mask)
