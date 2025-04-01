import torch.nn as nn
from torchvision import models
import torch
import spacy
from collections import Counter
import pickle

class Vocabulary:
  def __init__(self, min_freq):
    # Making a dictionary that maps integers to tokens
    # Adding the pre-defined special tokens
    self.int_to_str = {0:"<PAD>",1:"<SOS>",2:"<EOS>",3:"<UNK>"}
    self.spacy_eng = spacy.load("en_core_web_sm")

    # Making the reverse dictionary as well
    self.str_to_int = {v:k for k, v in self.int_to_str.items()}

    # Minimum no. of occurances a word needs to have to add it in vocabulary
    self.freq_threshold = min_freq

  def __len__(self):
    """Returns the length of the vocabulary"""
    return len(self.int_to_str)

  def tokenize_text(self, text):
    """Tokenizes a body of text"""
    return [token.text.lower() for token in self.spacy_eng.tokenizer(text)]

  def build_vocab(self, sentence_list):
    """Builds the vocabulary for a list of sentences"""
    frequencies = Counter()
    idx = 4  # indices 0-3 were already filled by the special tokens before

    for sentence in sentence_list:
      for word in self.tokenize_text(sentence):
        frequencies[word] += 1

        # Adding the word to the vocabulary
        if frequencies[word] == self.freq_threshold:
          self.str_to_int[word] = idx
          self.int_to_str[idx] = word
          idx += 1

  def numericalize(self, text):
    """Returns a list of indices of each token from input text in vocab"""
    tokenized_text = self.tokenize_text(text)
    return [self.str_to_int[token] if token in self.str_to_int else self.str_to_int["<UNK>"] for token in tokenized_text]

device = 'cuda'
dataVocab = None
with open('attention_cnn_lstm/dataVocab.pkl', 'rb') as f:
    dataVocab = pickle.load(f)

class Encoder(nn.Module):
    def __init__(self, embed_size, dropout_prob):
        super(Encoder, self).__init__()
        resnet = models.resnet50(pretrained=True)
        for params in resnet.parameters():
            params.requires_grad_(False)
        
        modules = list(resnet.children())[:-2]  # Removing the final 2 layers - the classifier and the global pooling layer. These are removed so that instead of receiving output in 1D and losing all the spatial information, we instead extract the 2D feature map to use in attention mechanism
        # Here, since we've removed the pooling layer, we keep the 7x7 feature map across all 2048 dimensions rather than just a 1x1 embedding in case of global pooling        
        self.resnet = nn.Sequential(*modules)
        
    def forward(self, images):
        # images is of dims [batch_size, num_channels, height, width] -> [batch_size, 3, 224, 224]
        features = self.resnet(images) # features is [batch_size, 2048, 7, 7]
        features = features.permute(0, 2, 3, 1)  # features is [batch_size, 7, 7, 2048]
        features = features.view(features.size(0), -1, features.size(-1))  # features is [batch_size, 49, 2048]
        return features

class Attention(nn.Module):
    def __init__(self, encoder_dim, decoder_dim, attention_dim):
        super(Attention, self).__init__()
        self.encoderAtt = nn.Linear(encoder_dim, attention_dim)   # attention value for encoder features
        self.decoderAtt = nn.Linear(decoder_dim, attention_dim)   # attention value for decoder features 
        
        self.fullAtt = nn.Linear(attention_dim, 1)  # Score for each region
        self.softmax = nn.Softmax(dim=1) 
    
    def forward(self, encoder_features, hidden_state):
        # encoder features -> [batch_size, num_regions, encoder_dim]  -> num_regions = 7*7 = 49
        # hidden state -> [batch_size, decoder_dim]
        
        encoderAttention = self.encoderAtt(encoder_features)  # [batch_size, num_regions, encoder_dim]
        decoderAttention = self.decoderAtt(hidden_state.unsqueeze(1))   # [batch_size, 1, decoder_dim]
        fullAttention = self.fullAtt(torch.tanh(encoderAttention + decoderAttention))  # [batch_size, num_regions, 1]   
        
        fullAttention = fullAttention.squeeze(2)  # [batch_size, num_regions]  -> These are the attention scores for each pixel in our 7*7 input feature map
        attention_weights = self.softmax(fullAttention)  # calculating attention weights for embeddings [batch_size, num_regions]
        
        attention_weighted_encoding = (encoder_features*attention_weights.unsqueeze(2)).sum(dim=1)
        # Final encoding -> [batch_size, encoder_dim]
        return attention_weights, attention_weighted_encoding
    
class Decoder(nn.Module):
    def __init__(self, embed_size, vocab_size, hidden_dim, num_layers, dropout_prob, encoder_dim, attention_dim):
        super().__init__()
        self.attention_dim = attention_dim
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.attention = Attention(encoder_dim, hidden_dim, attention_dim)
        self.lstm_cell = nn.LSTMCell(embed_size + encoder_dim, hidden_dim, bias=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)
        self.dropout = nn.Dropout(p=dropout_prob)
        self.hidden_dim = hidden_dim
        self.num_layers=num_layers
        self.init_h = nn.Linear(encoder_dim, hidden_dim)
        self.init_c = nn.Linear(encoder_dim, hidden_dim)
        self.vocab_size=vocab_size
        
    def init_hidden_state(self, encoder_features):
        # encoder features -> [batch_size, num_regions, encoder_dim]
        mean_encoder_out = encoder_features.mean(dim=1)
        h = self.init_h(mean_encoder_out)  # [batch_size, decoder_dim]
        c = self.init_c(mean_encoder_out)  # [batch_size, decoder_dim]
        return h, c
        
    def forward(self, encoder_features, captions):
        # encoder_features -> [batch_size, num_regions, encoder_dim]
        # captions -> [batch_size, seq_len]
        
        batch_size, _, encoder_dim = encoder_features.shape
        
        h, c = self.init_hidden_state(encoder_features)   # Initializing initial hidden and cell states for the LSTM using mean of encoder features
        # h, c both are  -> [batch_size, decoder_dim]

        seq_len = len(captions[0])-1
        embeds = self.embedding(captions)  # [batch_size, seq_len, embed_size]
        preds = torch.zeros(batch_size, seq_len, self.vocab_size).to(device)
        num_features = encoder_features.size(1)
        attention_weights = torch.zeros(batch_size, seq_len, num_features).to(device)
        
        for s in range(seq_len):
          
          attention_weight, context_vector = self.attention(encoder_features, h)   
          # attention_weight -> [batch_size, num_regions], context_vector -> [batch_size, encoder_dim]
          
          lstm_input = torch.cat((embeds[:, s], context_vector), dim=1)    # lstm_input -> [batch_size, embed_size + encoder_dim]
          h, c = self.lstm_cell(lstm_input, (h, c))   
          # h, c both are -> [batch_size, decoder_dim]
          output = self.fc(self.dropout(h))  # output is -> [batch_size, vocab_size]
          preds[:, s] = output
          attention_weights[:, s] = attention_weight
        
        return preds, attention_weights
    
    def generate_captions(self, encoder_features, vocab, max_len=20):
      # encoder_features -> [batch_size, num_regions, encoder_dim]  
      batch_size, _, encoder_dim = encoder_features.shape
      attention_weights = []
      
      word = torch.tensor(vocab.str_to_int['<SOS>']).view(1, -1).to(device)  # [batch_size, 1]
      h, c = self.init_hidden_state(encoder_features)   # [batch_size, decoder_dim]
      embeds = self.embedding(word)  # [batch_size, seq_len(=1), embed_size]
      
      captions = []
      for _ in range(max_len):
        attention_weight, context_vector = self.attention(encoder_features, h)
        attention_weights.append(attention_weight.cpu().detach().numpy())
        
        lstm_input = torch.cat((embeds[:, 0], context_vector), dim=1)
        h, c = self.lstm_cell(lstm_input, (h, c))
        
        output = self.fc(self.dropout(h))
        output = output.view(batch_size, -1)  # This extra statement is added because words are generated auto regressively, so we don't need get the entire seq_len. This was not used in forward method since seq_len is needed for cross entropy loss
        
        predicted_word_idx = output.argmax(dim=1)  # [batch_size]
        captions.append(predicted_word_idx.item())
        if vocab.int_to_str[predicted_word_idx.item()] == '<EOS>':
          break
        
        embeds = self.embedding(predicted_word_idx.unsqueeze(0))
      
      predicted_caption = ""
      for i in range(1, len(captions)-1):
        predicted_caption += vocab.int_to_str[captions[i]]
        predicted_caption += " "
          
      return predicted_caption, attention_weights
    
class EncoderDecoderUsingAttention(nn.Module):
  def __init__(self, embed_size, vocab_size, hidden_dim, num_layers, dropout_prob, encoder_dim, attention_dim):
    super(EncoderDecoderUsingAttention, self).__init__()
    self.encoder = Encoder(embed_size, dropout_prob)
    self.decoder = Decoder(embed_size, vocab_size, hidden_dim, num_layers, dropout_prob, encoder_dim, attention_dim)

  def forward(self, images, captions):
    features = self.encoder(images)   # features is [batch_size, num_regions=49, embed_size]
    outputs = self.decoder(features, captions)
    return outputs
        
        
        
          
                    