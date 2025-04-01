import torchvision.models as models
from torch.autograd import Variable
import torch
import math
import torch.nn as nn
import random

device = 'cuda'

class ImageEncoder():
    def __init__(self):
        super(ImageEncoder, self).__init__()
        self.resnet50 = models.resnet50(pretrained=True).to(device)
        self.resnet50.eval()
        self.resNet50Layer4 = self.resnet50._modules.get('layer4')
        
    def get_vector(self, image):
        t_img = Variable(image).to(device)
        my_embedding = torch.zeros(1, 2048, 7, 7).to(device)
        
        def copy_data(m, i, o):
            my_embedding.copy_(o.data)
        
        h = self.resNet50Layer4.register_forward_hook(copy_data)
        self.resnet50(t_img)
        
        h.remove()
        return my_embedding    # [1, 2048, 7, 7]
    
# temp = ImageEncoder()
# ans = temp.get_vector(torch.zeros(1, 3, 224, 224).to(device))
# print(ans)
max_seq_len = 40

class PositionalEncoding(nn.Module):
  def __init__(self, embed_size, dropout_prob=0.1, max_len=max_seq_len):
    super(PositionalEncoding, self).__init__()
    self.dropout = nn.Dropout(p=dropout_prob)

    pe = torch.zeros(max_len, embed_size)   # Creating positional encoding vector for each word in max_seq_len and of dimension embed_size -> [max_len, embed_size]
    position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)   # Creating a position vector with numbers in ascending order of dims [max_len,] and unsqueezing it to [max_len, 1] which represents position index of each word in our sequence
    div_term = torch.exp(torch.arange(0, embed_size, 2).float() * (-math.log(10000.0) / embed_size)) # Generates values for only even indices. This is a scaling factor that prevents position values from becoming too large
    pe[:, 0::2] = torch.sin(position*div_term)
    pe[:, 1::2] = torch.cos(position*div_term)
    
    pe = pe.unsqueeze(0)  # Giving our final positional encoding pe the 'batch' dimension -> [1, max_len, embed_size]
    # Note that pe doesn't vary with batch_size or number of training examples. pe is generated for each word in a caption using the above written formulas and does not have any trainable parameters. So we make each input sequence simply go through the above transformations
    self.register_buffer('pe', pe)  # Storing pe as a non-trainable tensor
    
  def forward(self, x):
    if self.pe.size(0) < x.size(0):
        self.pe.repeat(x.size(0), 1, 1).to(device)   # Copy pasting the same value of pe if input is given in batch format
    self.pe = self.pe[:x.size(0), : , : ]
    
    x = x + self.pe   # Adding positional encoding to the original embeddings
    return self.dropout(x)

class ImageCaptioningTransformerArchitecture(nn.Module):
    def __init__(self, n_heads, n_decoder_layers, vocab_size, embed_size):
        super(ImageCaptioningTransformerArchitecture, self).__init__()
        self.pos_encoder = PositionalEncoding(embed_size=embed_size, dropout_prob=0.1)
        self.TransformerDecoderLayer = nn.TransformerDecoderLayer(d_model=embed_size, nhead=n_heads)  # Will take input and give output as [seq_len, batch_size, embed_size]
        self.TransformerDecoder = nn.TransformerDecoder(decoder_layer=self.TransformerDecoderLayer, num_layers=n_decoder_layers)
        self.embed_size = embed_size
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.last_linear_layer = nn.Linear(embed_size, vocab_size)
        self.init_weights()
        
    def init_weights(self):
        # Uniform weight initialization for better training stability
        init_range = 0.1
        self.embedding.weight.data.uniform_(-init_range, init_range)
        self.last_linear_layer.bias.data.zero_()
        self.last_linear_layer.weight.data.uniform_(-init_range, init_range)
        
    def generate_masks(self, size, decoder_input):
        # Decoder input is the captions and size is the seq_len of the provided caption to be masked
        decoder_input_mask = (torch.triu(torch.ones(size, size)) == 1).transpose(0, 1)    # Creates a mask to hide the later words so that our model can't cheat
        decoder_input_mask = decoder_input_mask.float().masked_fill(decoder_input_mask == 0, float('-inf')).masked_fill(decoder_input_mask == 1, float(0.0))
        decoder_input_pad_mask = decoder_input.float().masked_fill(decoder_input == 0, float(0.0)).masked_fill(decoder_input > 0, float(1.0))  # Creates a padding mask to tell our model whether the current token is <PAD> and thus has to be ignored or not. This is done to prevent attention heads from learning uneccessary relationship of different word tokens with <PAD> token
        # In our vocab, 0 means <PAD> and any index > 0 is a valid word
        decoder_input_pad_mask_bool = decoder_input == 0
        
        return decoder_input_mask, decoder_input_pad_mask, decoder_input_pad_mask_bool

    def forward(self, encoder_features, decoder_input):
        # Encoder features are the extracted image features
        # Both encoder features and decoder_input are of dims -> [batch_size, seq_len, embed_size]
        encoder_features = encoder_features.permute(1, 0, 2)  # Converting to [seq_len, batch_size, embed_size]
        
        decoder_input_embed = self.embedding(decoder_input) * math.sqrt(self.embed_size)
        decoder_input_embed = self.pos_encoder(decoder_input_embed)
        decoder_input_embed = decoder_input_embed.permute(1, 0, 2)  # Converting to [seq_len, batch_size, embed_size]
        
        decoder_input_mask, decoder_input_pad_mask, decoder_input_pad_mask_bool = self.generate_masks(size=decoder_input.size(1), decoder_input=decoder_input) 
        decoder_input_mask = decoder_input_mask.to(device)
        decoder_input_pad_mask = decoder_input_pad_mask.to(device)
        decoder_input_pad_mask_bool = decoder_input_pad_mask_bool.to(device)
        
        decoder_output = self.TransformerDecoder(tgt=decoder_input_embed,   # The input sequence
                                                 memory=encoder_features,   # The encoder features
                                                 tgt_mask=decoder_input_mask,   # Mask for input sequence for hiding future words
                                                 tgt_key_padding_mask=decoder_input_pad_mask_bool)  # Mask for input sequence for ignoring <PAD> tokens
        # decoder output is of dims -> [seq_len, batch_size, embed_size]
        final_output = self.last_linear_layer(decoder_output)
        # final output is of dims -> [seq_len, batch_size, vocab_size]
        return final_output, decoder_input_pad_mask

def generate_caps(model, img, str_to_int, int_to_str, K=1, max_len=40):
    model.eval()
    img = img.to(device)     # [batch_size, 3, 224, 224]
    encoder = ImageEncoder()
    img_embed = encoder.get_vector(img)   # [batch_size, 2048, 7, 7]
    img_embed = img_embed.permute(0, 2, 3, 1)   # [batch_size, 7, 7, 2048]
    img_embed = img_embed.view(img_embed.size(0), -1, img_embed.size(3))   # [batch_size, 49, 2048]
    
    pad_token = str_to_int['<PAD>']
    start_token = str_to_int['<SOS>']
    input_seq = [pad_token]*max_len     # List of 40 pad tokens
    input_seq[0] = start_token    # 0th token is updated to <SOS>
    
    input_seq = torch.tensor(input_seq).unsqueeze(0).to(device)     # converted to [1, max_len]
    predicted_sentence = []
    with torch.inference_mode():
        for eval_iter in range(0, max_len-1):
            output, padding_mask = model.forward(img_embed, input_seq) 
            # output -> [seq_len, batch_size(here = 1 (since we're only generating captions for 1 image at a time)), vocab_size] , padding_mask -> [size, size]
            output = output[eval_iter, 0, :]  # Finds the output of the model's current word(current_iter), 0th image in the batch, and across all vocab size
            # So basically output represents the model's output logits for the current word for the 0th (and only) image in our batch
            # output -> [vocab_size]
            
            values = torch.topk(output, K).values.tolist()     # Pick top 'K' values from output
            indices = torch.topk(output, K).indices.tolist()   # Pick indices of top 'K' values from output
            
            next_word_idx = random.choices(indices, values, k=1)[0]   # Select random index from indices based weighted by values
            next_word = int_to_str[next_word_idx]   
            
            input_seq[:, eval_iter+1] = next_word_idx   # Updating the eval_iter+1 th index in the input_seq to be the current word (+1 because initial word is <SOS>)
            
            if next_word == '<EOS>':
                break
            
            predicted_sentence.append(next_word)
    
    predicted_caption = " ".join(predicted_sentence)
    return predicted_caption
            
    
    

