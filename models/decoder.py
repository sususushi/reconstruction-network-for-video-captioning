import torch
import torch.nn as nn
import torch.nn.functional as F


class Decoder(nn.Module):
    def __init__(self, model_name, n_layers, encoder_size, embedding_size, embedding_scale, hidden_size,
                 attn_size, output_size, embedding_dropout, dropout, out_dropout):
        super(Decoder, self).__init__()
        self.model_name = model_name
        self.n_layers = n_layers
        self.encoder_size = encoder_size
        self.embedding_size = embedding_size
        self.embedding_scale = embedding_scale
        self.hidden_size = hidden_size
        self.attn_size = attn_size
        self.output_size = output_size
        self.embedding_dropout_p = embedding_dropout
        self.dropout_p = dropout
        self.out_dropout_p = out_dropout

        self.embedding = nn.Embedding(self.output_size, self.embedding_size)
        self.embedding_dropout = nn.Dropout(self.embedding_dropout_p)
        
         # word embedding的定义可以这么理解，例如nn.Embedding(2, 4)
         # 2表示有2个词，4表示4维度，其实也就是一个2x4的矩阵，
         # 如果有100个词，每个词10维，就可以写为nn.Embedding(100, 10)
         # 注意这里的词向量的建立只是初始的词向量，并没有经过任何修改优化
         # 需要建立神经网络通过learning的办法修改word embedding里面的参数
         # 使得word embedding每一个词向量能够表示每一个不同的词。

        self.attn_W = nn.Linear(self.hidden_size, self.attn_size, bias=False)
        self.attn_U = nn.Linear(self.encoder_size, self.attn_size, bias=False)
        self.attn_b = nn.Parameter(torch.ones(self.attn_size), requires_grad=True)
        self.attn_tanh = nn.Tanh()
        self.attn_w = nn.Linear(self.attn_size, 1, bias=False)
        self.attn_softmax = nn.Softmax()

        if self.model_name == "LSTM":
            rnn_unit = nn.LSTM
        else:
            rnn_unit = nn.GRU
        self.rnn = rnn_unit(
            input_size=self.embedding_size + self.encoder_size,
            hidden_size=self.hidden_size,
            num_layers=self.n_layers,
            dropout=self.dropout_p)

        self.out = nn.Linear(self.hidden_size, self.output_size)
        self.out_dropout = nn.Dropout(self.out_dropout_p)

    def forward(self, input, hidden, encoder_outputs):
        embedded = self.embedding(input)
        embedded = embedded * self.embedding_scale
        embedded = self.embedding_dropout(embedded)

        if self.model_name == "LSTM":
            Wh = self.attn_W(hidden[0][-1])
        else:
            Wh = self.attn_W(hidden[-1])
        Uv = self.attn_U(encoder_outputs)
        Wh = Wh.unsqueeze(1).expand_as(Uv)
        alphas = Wh + Uv + self.attn_b
        alphas = self.attn_tanh(alphas)
        alphas = self.attn_w(alphas)
        alphas = alphas.expand_as(encoder_outputs)
        weighted_encoder_outputs = alphas * encoder_outputs
        attn_encoder_feature = weighted_encoder_outputs.mean(dim=1)
        attn_encoder_feature = attn_encoder_feature.unsqueeze(0)

        input_combined = torch.cat((embedded, attn_encoder_feature), dim=2)

        output, hidden = self.rnn(input_combined, hidden)

        output = self.out(output[0])
        output = self.out_dropout(output)
        return output, hidden

