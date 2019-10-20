import torch
from torch import nn
from torch.autograd import Variable


class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, embedding_size):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.embedding_size = embedding_size

        self.embedding = nn.Embedding(input_size, self.embedding_size)
        self.embedding.weight.data.normal_(0, 1 / self.embedding_size ** 0.5)
        self.gru = nn.GRU(embedding_size, hidden_size, bidirectional=True, batch_first=True)

    def forward(self, input, hidden, lengths):
        # iput batch must be sorted by sequence length
        input = input.masked_fill(input > self.embedding.num_embeddings,
                                  3)  # replace OOV words with <UNK> before embedding
        embedded = self.embedding(input)

        # pack_padded_sequence: 有很多个batch及对应的lengths，每个batch内sample的len不一样，取每个batch内的max_len
        packed_embedded = torch.nn.utils.rnn.pack_padded_sequence(embedded, lengths, batch_first=True)
        self.gru.flatten_parameters()
        output, hidden = self.gru(packed_embedded, hidden)
        # 只需要pad output（因为output是最后一层各个step的输出，不定长），hidden是每一层最后一个step的隐状态输出
        output, output_lengths = torch.nn.utils.rnn.pad_packed_sequence(output, batch_first=True)
        return output, hidden

    def init_hidden(self, batch_size):
        hidden = Variable(torch.zeros(2, batch_size, self.hidden_size))  # bidirectional rnn
        if next(self.parameters()).is_cuda:
            return hidden.cuda()
        else:
            return hidden
