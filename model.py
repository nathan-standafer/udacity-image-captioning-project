import torch
import torch.nn as nn
import torchvision.models as models


class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()
        resnet = models.resnet50(pretrained=True)
        for param in resnet.parameters():
            param.requires_grad_(False)
        
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        print("resnet.fc.in_features: {}".format(resnet.fc.in_features))
        self.embed = nn.Linear(resnet.fc.in_features, embed_size)

    def forward(self, images):
        features = self.resnet(images)
        features = features.view(features.size(0), -1)
        features = self.embed(features)
        return features
    

class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
        super(DecoderRNN, self).__init__()
        
        # embedding layer that turns words into a vector of a specified size
        self.word_embeddings = nn.Embedding(vocab_size, embed_size)
        
        self.lstm = nn.LSTM(embed_size, hidden_size)
        
        # the linear layer that maps the hidden state output dimension 
        # to the number of tags we want as output, tagset_size (in this case this is 3 tags)
        self.hidden2tag = nn.Linear(hidden_size, vocab_size)
        
        # initialize the hidden state (see code below)
        #self.hidden = self.init_hidden()
        
        #pass
    
    def forward(self, features, captions):
        print("DecoderRNN - features.shape: {}".format(features.shape))
        print("DecoderRNN - captions.shape: {}".format(captions.shape))
        
        x = self.word_embeddings(features)
        
               # get the output and hidden state by passing the lstm over our word embeddings
        # the lstm takes in our embeddings and hiddent state
        lstm_out, self.hidden = self.lstm(embeds.view(len(sentence), 1, -1), self.hidden)
        
        #outputs should be a PyTorch tensor with size [batch_size, captions.shape[1], vocab_size]
        #Your output should be designed such that outputs[i,j,k] contains the model's predicted score
        
        # get the scores for the most likely tag for a word
        tag_outputs = self.hidden2tag(lstm_out.view(captions.shape[1], -1))
        tag_scores = F.log_softmax(tag_outputs, dim=1)
        return tag_scores
        

    def sample(self, inputs, states=None, max_len=20):
        " accepts pre-processed image tensor (inputs) and returns predicted sentence (list of tensor ids of length max_len) "
        pass