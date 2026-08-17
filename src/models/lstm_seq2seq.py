import torch
import torch.nn as nn
import torch.nn.functional as F
import random

class LSTMEncoder(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers=2, dropout=0.3):
        super(LSTMEncoder, self).__init__()
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(
            embedding_dim, 
            hidden_dim, 
            num_layers=num_layers,
            bidirectional=True,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        self.dropout = nn.Dropout(dropout)
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
    def forward(self, src, src_lengths=None):
        # src: (batch_size, src_len)
        embedded = self.dropout(self.embedding(src))
        
        if src_lengths is not None:
            embedded = nn.utils.rnn.pack_padded_sequence(
                embedded, 
                src_lengths.cpu(), 
                batch_first=True, 
                enforce_sorted=False
            )
        
        outputs, (hidden, cell) = self.lstm(embedded)
        
        if src_lengths is not None:
            outputs, _ = nn.utils.rnn.pad_packed_sequence(outputs, batch_first=True)
        
        return outputs, hidden, cell

class Attention(nn.Module):
    def __init__(self, hidden_dim):
        super(Attention, self).__init__()
        self.encoder_proj = nn.Linear(hidden_dim * 2, hidden_dim)
        self.decoder_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v = nn.Linear(hidden_dim, 1, bias=False)
        
    def forward(self, decoder_hidden, encoder_outputs, mask=None):
        batch_size, src_len, _ = encoder_outputs.shape
        
        encoder_projected = self.encoder_proj(encoder_outputs)
        decoder_projected = self.decoder_proj(decoder_hidden)
        decoder_projected = decoder_projected.unsqueeze(1).repeat(1, src_len, 1)
        
        energy = torch.tanh(encoder_projected + decoder_projected)
        attention_scores = self.v(energy).squeeze(2)
        
        if mask is not None:
            attention_scores = attention_scores.masked_fill(mask == 0, -1e10)
            
        attention_weights = F.softmax(attention_scores, dim=1)
        return attention_weights

class LSTMDecoder(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers=2, dropout=0.3):
        super(LSTMDecoder, self).__init__()
        
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.attention = Attention(hidden_dim)
        
        # Project bidirectional context to decoder dimension
        self.context_proj = nn.Linear(hidden_dim * 2, hidden_dim)
        
        self.lstm = nn.LSTM(
            embedding_dim + hidden_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        self.fc_out = nn.Linear(hidden_dim + hidden_dim + embedding_dim, vocab_size)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, input, hidden, cell, encoder_outputs, mask=None):
        input = input.unsqueeze(1)
        embedded = self.dropout(self.embedding(input))
        
        attn_weights = self.attention(hidden[-1], encoder_outputs, mask)
        attn_weights = attn_weights.unsqueeze(1)
        context = torch.bmm(attn_weights, encoder_outputs)
        
        # Project context to decoder hidden dimension
        context_projected = self.context_proj(context)
        
        lstm_input = torch.cat((embedded, context_projected), dim=2)
        lstm_output, (hidden, cell) = self.lstm(lstm_input, (hidden, cell))
        
        combined_output = torch.cat((lstm_output, context_projected, embedded), dim=2)
        prediction = self.fc_out(combined_output.squeeze(1))
        
        return prediction, hidden, cell, attn_weights.squeeze(1)

class LSTMSeq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super(LSTMSeq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        
        # Projection layer for hidden state initialization
        if self.encoder.lstm.bidirectional:
            self.init_hidden_proj = nn.Linear(encoder.hidden_dim * 2, decoder.hidden_dim)
        else:
            self.init_hidden_proj = nn.Linear(encoder.hidden_dim, decoder.hidden_dim)
        
    def create_mask(self, src):
        return (src != 0)
    
    def forward(self, src, trg, teacher_forcing_ratio=0.5, src_lengths=None):
        """Forward pass with optional sequence lengths for packed sequences"""
        batch_size = trg.shape[0]
        trg_len = trg.shape[1]
        trg_vocab_size = self.decoder.vocab_size
        
        outputs = torch.zeros(batch_size, trg_len, trg_vocab_size).to(self.device)
        
        # ✅ UPDATED: Pass src_lengths to encoder
        encoder_outputs, encoder_hidden, encoder_cell = self.encoder(src, src_lengths)
        
        # ✅ FIXED: Proper hidden state initialization (the bidirectional fix)
        if self.encoder.lstm.bidirectional:
            num_layers = self.encoder.num_layers
            hidden_dim = self.encoder.hidden_dim
            
            encoder_hidden_reshaped = encoder_hidden.view(
                num_layers, 2, batch_size, hidden_dim
            )
            
            last_forward = encoder_hidden_reshaped[-1, 0]
            last_backward = encoder_hidden_reshaped[-1, 1]
            
            decoder_hidden_init = torch.cat([last_forward, last_backward], dim=1)
            decoder_hidden = self.init_hidden_proj(decoder_hidden_init)
            decoder_hidden = decoder_hidden.unsqueeze(0).repeat(self.decoder.num_layers, 1, 1)
        else:
            decoder_hidden = encoder_hidden[-1].unsqueeze(0).repeat(self.decoder.num_layers, 1, 1)
        
        decoder_cell = torch.zeros_like(decoder_hidden)
        
        input = trg[:, 0]
        mask = self.create_mask(src)
        
        for t in range(1, trg_len):
            output, decoder_hidden, decoder_cell, _ = self.decoder(
                input, decoder_hidden, decoder_cell, encoder_outputs, mask
            )
            
            outputs[:, t] = output
            
            teacher_force = random.random() < teacher_forcing_ratio
            top1 = output.argmax(1)
            input = trg[:, t] if teacher_force else top1
            
        return outputs

def initialize_lstm_model(roman_vocab_size, devanagari_vocab_size, device, 
                         embedding_dim=256, hidden_dim=512, num_layers=2, dropout=0.3):
    
    encoder = LSTMEncoder(
        vocab_size=roman_vocab_size,
        embedding_dim=embedding_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        dropout=dropout
    )
    
    decoder = LSTMDecoder(
        vocab_size=devanagari_vocab_size,
        embedding_dim=embedding_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        dropout=dropout
    )
    
    model = LSTMSeq2Seq(encoder, decoder, device)
    return model.to(device)