import streamlit as st
import random
from PIL import Image
from torchvision import transforms
from cnn_lstm.cnn_lstm_model import Vocabulary, EncoderDecoder
from cnn_transformer.cnn_transformer_model import ImageCaptioningTransformerArchitecture, PositionalEncoding, generate_caps
from attention_cnn_lstm.attention_cnn_lstm_model import EncoderDecoderUsingAttention
import pickle
import torch
import torchvision.transforms.v2 as T
import asyncio
import sys

if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

device = 'cuda'
print(torch.cuda.is_available())


# LOADING THE CNN + LSTM MODEL
model_vocab = None
with open('cnn_lstm/vocab.pkl', 'rb') as f:
    model_vocab = pickle.load(f) 
# print(model_vocab)
cnn_lstm = EncoderDecoder(embed_size=400, hidden_size=512, num_layers=2, vocab=model_vocab)
cnn_lstm.load_state_dict(torch.load('cnn_lstm/baseline_CNN_LSTM.pth', map_location=device))
cnn_lstm.eval()
# print(cnn_lstm.state_dict)
print(50*'-', 'CNN + LSTM Model Instantiated', 50*'-')


# LOADING THE CNN + TRANSFORMER MODEL
word_dict, int_to_str, str_to_int = None, None, None
with open('cnn_transformer/vocabulary.pkl', 'rb') as f:
    word_dict, int_to_str, str_to_int = pickle.load(f)
# print(word_dict, int_to_str, str_to_int)
cnn_transformer = ImageCaptioningTransformerArchitecture(n_heads=16, n_decoder_layers=4, vocab_size=len(word_dict), embed_size=2048).to(device)
cnn_transformer = torch.load('cnn_transformer/BestModelNew.pth', map_location=device)
cnn_transformer.eval()
# print(cnn_transformer.state_dict)
print(50*'-', 'CNN + Transformer Model Instantiated', 50*'-')


# LOADING THE CNN + LSTM + ATTENTION MECHANISM
dataVocab = None
with open('attention_cnn_lstm/dataVocab.pkl', 'rb') as f:
    dataVocab = pickle.load(f)
# print(dataVocab)
cnn_lstm_attention = EncoderDecoderUsingAttention(embed_size=300, hidden_dim=512, dropout_prob=0.2, encoder_dim=2048, attention_dim=256, num_layers=1, vocab_size=len(dataVocab)).to(device)
cnn_lstm_attention.load_state_dict(torch.load('attention_cnn_lstm/best_modelNew.pth', map_location=device))
cnn_lstm_attention.eval()
# print(cnn_lstm_attention.state_dict)
print(50*'-', 'CNN + LSTM + Attention Model Instantiated', 50*'-')


st.title("Image Caption Generator")
st.write("In this demonstration, we'll be testing out 3 types of architectures for the Image Captioning Task. Checkout the github repository for architecture and training details")
st.write('Github Repository link: LINK GOES HERE')

st.subheader('You can either upload your image for testing or try any random sample image')

img_transforms = T.Compose([
    T.Resize(226),
    T.CenterCrop(224),
    T.ToImage(),
    T.ToDtype(torch.float32, scale=True),
    T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))  # values of normalization as used in ImageNet dataset - so models like ResNet, VGG perform better with this type of normalization rather than just dividing by 255.0. First tuple shows mean of RGB channels and second tuple shows standard deviation of the same
])    

def generate_captions(image):
    transformed_image = img_transforms(image).to(device)   # [num_channels, height, width]
    # print(transformed_image.shape)
    transformed_image = transformed_image.unsqueeze(0).detach()  # [1, num_channels, height, width] -> 1 is for batch compatibility
    image_features = cnn_lstm.encoder(transformed_image) # [batch_size, embed_size]
    out1 = cnn_lstm.decoder.generate_caption(image_features.unsqueeze(1))
    out2 = generate_caps(model=cnn_transformer, img=transformed_image, str_to_int=str_to_int, int_to_str=int_to_str)
    features = cnn_lstm_attention.encoder(transformed_image)
    out3, _ = cnn_lstm_attention.decoder.generate_captions(encoder_features=features, vocab=dataVocab)
    return out1, out2, out3
    
    
sample = st.button('Try with Sample Image')
if sample:
    img_idx = random.randint(0, 17)
    img = 'SampleImages/' + str(img_idx) + '.jpg'
    image = Image.open(img).convert("RGB")
    st.image(image)
    out1, out2, out3 = generate_captions(image)
    # st.image(image)
    tab1, tab2, tab3 = st.tabs(["CNN + LSTM", "CNN + Transformer", "CNN + LSTM with Attention"])
    tab1.image(image, caption=out1)
    tab2.image(image, caption=out2)
    tab3.image(image, caption=out3) 

    
chull = st.file_uploader("Upload an image", type=["jpg"])
if chull:
    image = Image.open(chull).convert("RGB")
    # st.image(image)
    out1, out2, out3 = generate_captions(image)
    # st.image(image)
    tab1, tab2, tab3 = st.tabs(["CNN + LSTM", "CNN + Transformer", "CNN + LSTM with Attention"])
    tab1.image(image, caption=out1)
    tab2.image(image, caption=out2)
    tab3.image(image, caption=out3)