import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import numpy as np
import random


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

        print('init DecoderRNN. embed_size: {}, hidden_size: {}, vocab_size: {}, num_layers: {}'.format(embed_size, hidden_size, vocab_size, num_layers))

        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size

        # embedding layer that turns words into a vector of a specified size
        self.word_embeddings = nn.Embedding(vocab_size, embed_size)
        
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        
        # the linear layer that maps the hidden state output dimension 
        self.hidden2tag = nn.Linear(hidden_size, vocab_size)

    def init_hidden(self, sentence_length):
        ''' Initializes hidden state '''
        # Create two new tensors with sizes n_layers x batch_size x hidden_dim,
        # initialized to zero, for hidden state and cell state of LSTM
        weight = next(self.parameters()).data
        
        hidden = (weight.new(self.num_layers, sentence_length, self.hidden_size).zero_(),
                  weight.new(self.num_layers, sentence_length, self.hidden_size).zero_())

        return hidden

    def forward(self, features, captions):
        #print("DecoderRNN XXXXX - features.shape: {}".format(features.shape))
        #print("DecoderRNN - captions.shape: {}".format(captions.shape))
        #print("DecoderRNN - captions: {}".format(captions))

        # ??? should I concatenate the features and captions before feeding into the LSTM?????

        x = self.word_embeddings(captions[:,:-1])  #take off the last word (<end>)
        #print("x.shape after passing through word embeddings: {}".format(x.shape))  #torch.Size([10, 13, 256])
        #print("word embeddings out [0]: {}".format(x))
        
        # get the output and hidden state by passing the lstm over our word embeddings
        # the lstm takes in our embeddings and hiddent state
        sentence_length = captions.shape[1]
        batch_size = captions.shape[0]
        #lstm_out, hidden = self.lstm(x.view(sentence_length, 1, -1), hidden)

        #now, I meed to inject the image encodings (features) into the ltsm along with the output from the embeddings.
        #  features.shape: torch.Size([10, 256])
        #  x.shape:        torch.Size([10, 13, 256])
        #features_reshaped = torch.reshape(features, (features.shape[0], 1, features.shape[1]))
        features_reshaped = features.unsqueeze(dim=1)
        #print("DecoderRNN - features_reshaped.shape: {}".format(features_reshaped.shape))
        #print("DecoderRNN - x.shape: {}".format(x.shape))
        #print("feature reshaped[:][0][:] : {}".format(features_reshaped))
        x = torch.cat((features_reshaped, x), dim=1)


        #print("DecoderRNN - x.shape after catenation: {}".format(x.shape))
        #print("catenated x[1] - should match word embeddings out [0]: {}".format(x))
        #print("catenated x[2][0][:] - should match features: {}".format(x[2][0][:]))

        #hidden = self.init_hidden(sentence_length+1)

        lstm_out, _ = self.lstm(x)
        
        #outputs should be a PyTorch tensor with size [batch_size, captions.shape[1], vocab_size]
        #Your output should be designed such that outputs[i,j,k] contains the model's predicted score
        
        # get the scores for the most likely tag for a word
        #??? how to feed outut of RNN into linear layber(s).  Do I just 
        #print("lstm_out.shape: {}".format(lstm_out.shape))
        # shape output to be (batch_size*seq_length, hidden_dim)
        #lstm_out_contiguous = lstm_out.contiguous().view(-1, self.hidden_size)
        #print("lstm_out_contiguous.shape: {}".format(lstm_out_contiguous.shape))

        fc_outputs = self.hidden2tag(lstm_out)
        #print("fc_outputs.shape: {}".format(fc_outputs.shape))

        #outputs = fc_outputs.reshape(batch_size, -1, self.vocab_size) 
        #print("outputs.shape: {}".format(outputs.shape))

        #remove first output word:
        #outputs = outputs[:,1:,:]

        #remove lastt output word:
        #outputs = outputs[:,:-1,:]  #I took off the last word of the incoming caption, so no need to remove it here.

        #outputs_scores = F.log_softmax(outputs, dim=2) #pass raw logits to nn.CrossEntropyLoss,

        # print("outputs.shape: {}".format(outputs.shape))
        # print("outputs_scores.shape: {}".format(outputs_scores.shape))

        #print("pass raw logits to nn.CrossEntropyLoss")
        return fc_outputs
        

    def sample(self, inputs, states=None, max_len=30):
        " accepts pre-processed image tensor (inputs) and returns predicted sentence (list of tensor ids of length max_len) "
        #print("sample inputs.shape: {}".format(inputs.shape))
        #print("sample inputs[0]: {}".format(inputs[0]))

        top_x_words_to_use = 2
        attempts = 10
        best_score = -10000
        output_list_to_return = []
        
        for attempt in range(attempts):
            captions_np = np.zeros([inputs.shape[0], 1])
            captions = torch.from_numpy(captions_np).type(torch.LongTensor).to("cuda")

            output = self.forward(inputs[0], captions)
            output_np = output.cpu().detach().numpy()

            #print("output_np:  {}".format(output_np))
            output_list = []
            score_total = 0

            for i in range(max_len):
                top_five_words = np.argpartition(output_np[0,-1], -top_x_words_to_use)[-top_x_words_to_use:]
                #print("top_five_words: {}".format(top_five_words))
                next_word = top_five_words[random.randint(0, top_x_words_to_use-1)]
                score_total += output_np[0,-1][next_word]

                #next_word = 1 + np.argmax(output_np[0,-1][1:] ) #largest index that isn't index 0
                #next_word = np.argmax( output_np[0, -1] ) #largest index that isn't index 0

                #print("next_word: {}, value: {}".format(next_word, output_np[0,-1][1:][next_word-1]  ))
                #print("next_word: {}, value: {}".format(next_word, output_np[0,-1][1:][next_word-1]  ))

                captions_np = np.append(captions_np, [[next_word]], axis=1)
                captions = torch.from_numpy(captions_np).type(torch.LongTensor).to("cuda")

                #print("\ncurrent captions.shape: {}".format(captions.shape))
                output = self.forward(inputs[0], captions)
                output_np = output.cpu().detach().numpy()
                #print("output_np.shape:  {}".format(output_np.shape))

                output_list.append(int(next_word))

                if next_word == 1:
                    break
            
            if score_total > best_score:
                output_list_to_return = output_list
                best_score = score_total
                #print("best_score: {}".format(best_score))

        #print("best_score: {}".format(best_score))
            
        return output_list_to_return