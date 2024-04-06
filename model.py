import torch
import torch.nn as nn
import torch.nn.functional as F

# Encoder with single-layer bidirectional LSTM
class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True, bidirectional=True)
        self.reduce_h_n = nn.Linear(2 * hidden_size, hidden_size)
        self.reduce_c_n = nn.Linear(2 * hidden_size, hidden_size)

    def forward(self, input, input_len):
        packed_input = nn.utils.rnn.pack_padded_sequence(input, input_len, batch_first=True, enforce_sorted=False)
        packed_output, (h_n, c_n) = self.lstm(packed_input)
        output, _ = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True)
        h_n = torch.cat((h_n[-2, :, :], h_n[-1, :, :]), dim=1)
        c_n = torch.cat((c_n[-2, :, :], c_n[-1, :, :]), dim=1)
        h_n = self.reduce_h_n(h_n)
        c_n = self.reduce_c_n(c_n)
        h_n = F.relu(h_n)
        c_n = F.relu(c_n)
        return output, (h_n, c_n)

# Attention mechanism
class Attention(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.Ea = nn.Linear(hidden_size, hidden_size, bias=False)
        self.Da = nn.Linear(hidden_size, hidden_size, bias=True)
        self.Va = nn.Linear(hidden_size, 1, bias=False)

    def forward(self, input, encoder_outputs, pad_mask):
        enc = self.Ea(encoder_outputs)
        dec = self.Da(input)
        scores = torch.tanh(enc + dec.unsqueeze(1))
        scores = self.Va(scores).squeeze(2)
        attn_dist = torch.softmax(scores, dim=-1)
        context = torch.bmm(attn_dist.unsqueeze(1), encoder_outputs).squeeze(1)
        return attn_dist, context

# Decoder with attention
class Decoder(nn.Module):
    def __init__(self, hidden_size, output_size):
        super().__init__()
        self.lstm = nn.LSTM(hidden_size * 2, hidden_size, batch_first=True)
        self.attention = Attention(hidden_size)
        self.out = nn.Linear(hidden_size, output_size)

    def forward(self, input, hidden, encoder_outputs):
        output, hidden = self.lstm(input, hidden)
        attn_dist, context = self.attention(hidden[-1].unsqueeze(0), encoder_outputs, None)
        output = torch.cat((output.squeeze(1), context), 1)
        output = self.out(output)
        return F.log_softmax(output, dim=1), hidden
