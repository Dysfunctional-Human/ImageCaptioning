import torch
import torch.nn as nn
import torchvision.models as models
import spacy
from collections import Counter
    
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

class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()
        resnet = models.resnet50(pretrained=True).to(device)    # Getting the pre-trained ResNet50
        for param in resnet.parameters():
            param.requires_grad_(False)    # Setting all of it's parameters to be untrainable
            
        modules = list(resnet.children())[:-1]   # Getting all layers of the resnet except the last layer
        self.resnet = nn.Sequential(*modules)  
        self.embed = nn.Linear(resnet.fc.in_features, embed_size).to(device)   # Mapping the resnet's output features to shape [batch_size, embed_size]
        
    def forward(self, x):
        # x is a vector of images. x -> [batch_size, num_channels, height, width]
        features = self.resnet(x)    # [batch_size, 2048, 1, 1]
        features = features.view(features.size(0), -1)  # [batch_size, 2048]
        features = self.embed(features)   # [batch_size, embed_size]
        return features

class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, vocab, num_layers=1, drop_prob=0.3):
        super(DecoderRNN, self).__init__()
        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embed_size)
        self.lstm = nn.LSTM(input_size=embed_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        self.fcn = nn.Linear(hidden_size, vocab_size)
        self.drop = nn.Dropout(p=drop_prob)
        self.vocab = vocab
        
    def forward(self, features, captions):
        # features -> [batch_size, embed_size] | captions -> [batch_size, seq_len]
        embeds = self.embedding(captions[:, :-1])  # converting caption to embedding except the final <EOS> token embeds -> [batch_size, seq_len, embed_size]
        x = torch.cat((features.unsqueeze(1), embeds), dim=1)   # Concatenating image features and caption's embeddings -> [batch_size, seq_len, embed_size]
        x, _ = self.lstm(x)   # getting the final output of the LSTM 
        # x is now of the form [batch_size, seq_len, hidden_size]. x contains the outputs of our LSTM at every timestep
        x = self.fcn(x)   # A fully connected layer converts x to [batch_size, seq_len, vocab_size]
        # x now contains the logits for each word in vocab and at every timestep t
        # We will use a cross entropy loss function that internally takes a softmax of the above logits to get word probabilities for loss calculation
        return x

    def generate_caption(self, features, hidden=None, max_len=20):
        batch_size = features.size(0)
        vocab = self.vocab
        captions=[]
        
        for _ in range(max_len):
            output, hidden = self.lstm(features, hidden)
            output = self.fcn(output)
            output = output.view(batch_size, -1)  # [batch_size, vocab_size]  since seq_len will be 1 for our generated words
            
            predicted_word_idx = output.argmax(dim=1)  # Selecting the word with the highest probability
            
            captions.append(predicted_word_idx.item())
            
            if vocab.int_to_str[predicted_word_idx.item()] == '<EOS>':
                break
            
            features = self.embedding(predicted_word_idx.unsqueeze(0))
        
        predicted_caption = ""
        for i in range(1, len(captions)-1):
          predicted_caption += vocab.int_to_str[captions[i]]
          predicted_caption += " "
        # predicted_caption = [vocab.int_to_str[token] for token in captions]
        # return " ".join(predicted_caption)
        return predicted_caption

class EncoderDecoder(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab, num_layers=1, drop_prob=0.3):
        super(EncoderDecoder, self).__init__()
        self.vocab_size = len(vocab)
        self.encoder = EncoderCNN(embed_size=embed_size)
        self.decoder = DecoderRNN(embed_size=embed_size, hidden_size=hidden_size, vocab_size=self.vocab_size, vocab=vocab, num_layers=num_layers, drop_prob=drop_prob).to(device)
    
    def forward(self, images, captions):
        features = self.encoder(images)
        outputs = self.decoder(features, captions)
        return outputs
        