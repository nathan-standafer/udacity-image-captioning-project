import torch
import torch.nn as nn
import torch.nn.functional as F
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
    def __init__(self, embed_size, hidden_size, vocab_size, sentence_length, num_layers=1):
        super(DecoderRNN, self).__init__()

        print('init DecoderRNN. embed_size: {}, hidden_size: {}, vocab_size: {}, num_layers: {}'.format(embed_size, hidden_size, vocab_size, num_layers))

        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.sentence_length = sentence_length
        self.vocab_size = vocab_size

        # embedding layer that turns words into a vector of a specified size
        self.word_embeddings = nn.Embedding(vocab_size, embed_size)
        
        self.lstm = nn.LSTM(embed_size, hidden_size)
        
        # the linear layer that maps the hidden state output dimension 
        self.hidden2tag = nn.Linear(hidden_size, vocab_size)
        
        # initialize the hidden state (see code below)
        #self.hidden = self.init_hidden()
        
        #pass

    def init_hidden(self, sentence_length):
        ''' Initializes hidden state '''
        # Create two new tensors with sizes n_layers x batch_size x hidden_dim,
        # initialized to zero, for hidden state and cell state of LSTM
        weight = next(self.parameters()).data
        
        hidden = (weight.new(self.num_layers, sentence_length, self.hidden_size).zero_(),
                  weight.new(self.num_layers, sentence_length, self.hidden_size).zero_())

        return hidden

    def forward(self, features, captions, hidden):
        print("DecoderRNN - features.shape: {}".format(features.shape))
        print("DecoderRNN - captions.shape: {}".format(captions.shape))
        
        # ??? should I concatenate the features and captions before feeding into the LSTM?????

        x = self.word_embeddings(captions)
        print("x.shape: {}".format(x.shape))
        
        # get the output and hidden state by passing the lstm over our word embeddings
        # the lstm takes in our embeddings and hiddent state
        sentence_length = captions.shape[1]
        batch_size = captions.shape[0]
        #lstm_out, hidden = self.lstm(x.view(sentence_length, 1, -1), hidden)
        lstm_out, hidden = self.lstm(x, hidden)
        
        #outputs should be a PyTorch tensor with size [batch_size, captions.shape[1], vocab_size]
        #Your output should be designed such that outputs[i,j,k] contains the model's predicted score
        
        # get the scores for the most likely tag for a word
        #??? how to feed outut of RNN into linear layber(s).  Do I just 
        print("lstm_out.shape: {}".format(lstm_out.shape))
        # shape output to be (batch_size*seq_length, hidden_dim)
        lstm_out_contiguous = lstm_out.contiguous().view(-1, self.hidden_size)
        print("lstm_out_contiguous.shape: {}".format(lstm_out_contiguous.shape))

        fc_outputs = self.hidden2tag(lstm_out_contiguous)
        print("fc_outputs.shape: {}".format(fc_outputs.shape))

        outputs = fc_outputs.view(batch_size, -1, self.vocab_size) 
        print("outputs.shape: {}".format(outputs.shape))

        outputs_scores = F.log_softmax(outputs, dim=2)
        return outputs_scores
        

    def sample(self, inputs, states=None, max_len=20):
        " accepts pre-processed image tensor (inputs) and returns predicted sentence (list of tensor ids of length max_len) "
        pass