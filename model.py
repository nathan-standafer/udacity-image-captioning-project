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

    def forward(self, features, captions):
        #sentence_length = captions.shape[1]
        #batch_size = captions.shape[0]
        
        #get word embeddings from input captions
        x = self.word_embeddings(captions[:,:-1])  #take off the last word (<end>)

        #add a dim to the features to allow it to match the embeddings
        features_reshaped = features.unsqueeze(dim=1)
        #concatenate the incoming encoded features of the image (from EncoderCNN) to the word embeddings
        x = torch.cat((features_reshaped, x), dim=1)

        #pass through LSTM network
        lstm_out, _ = self.lstm(x)
        
        #pass through FC network
        fc_outputs = self.hidden2tag(lstm_out)

        #outputs should be a PyTorch tensor with size [batch_size, captions.shape[1], vocab_size]
        #Your output should be designed such that outputs[i,j,k] contains the model's predicted score
        return fc_outputs
        

    def softmax(self, X):
        expo = np.exp(X)
        expo_sum = np.sum(np.exp(X))
        return expo/expo_sum

    def sample_by_probability(self, inputs, states=None, max_len=30):
        " accepts pre-processed image tensor (inputs) and returns predicted sentence (list of tensor ids of length max_len) "

        attempts = 10
        best_score = -10000
        output_list_to_return = []
        
        #generate sentences, keeping score along the way to return the one with the best score.
        for attempt in range(attempts):
            captions_np = np.zeros([inputs.shape[0], 1])
            captions = torch.from_numpy(captions_np).type(torch.LongTensor).to("cuda")

            output = self.forward(inputs[0], captions)
            output_np = output.cpu().detach().numpy()

            output_list = []
            score_total = 0

            #run the sentence through the model, adding the next word each time.
            for i in range(max_len):
                out_softmax = self.softmax(output_np[0,-1])
                next_word_val = np.random.choice(out_softmax, p=out_softmax)
                next_word = np.where(out_softmax == next_word_val)[0][0]

                score_total += next_word_val

                captions_np = np.append(captions_np, [[next_word]], axis=1)
                captions = torch.from_numpy(captions_np).type(torch.LongTensor).to("cuda")

                output = self.forward(inputs[0], captions)
                output_np = output.cpu().detach().numpy()

                output_list.append(int(next_word))

                if next_word == 1: #if we get <end>, stop here
                    break
            
            score_average = score_total / len(output_list)
            if score_average > best_score:
                output_list_to_return = output_list
                best_score = score_average
            
        return output_list_to_return

    def sample(self, inputs, states=None, max_len=30, top_x_words_to_use=2):
        " accepts pre-processed image tensor (inputs) and returns predicted sentence (list of tensor ids of length max_len) "

        attempts = 10
        best_score = -10000
        output_list_to_return = []
        
        #generate sentences, keeping score along the way to return the one with the best score.
        for attempt in range(attempts):
            captions_np = np.zeros([inputs.shape[0], 1])
            captions = torch.from_numpy(captions_np).type(torch.LongTensor).to("cuda")

            output = self.forward(inputs[0], captions)
            output_np = output.cpu().detach().numpy()

            output_list = []
            score_total = 0

            #run the sentence through the model, adding the next word each time.
            for i in range(max_len):
                # #get the x best words from the output for the next word
                top_x_words = np.argpartition(output_np[0,-1], -top_x_words_to_use)[-top_x_words_to_use:]
                
                #randomly choose from the best words
                next_word = top_x_words[random.randint(0, top_x_words_to_use-1)]
                score_total += output_np[0,-1][next_word]

                captions_np = np.append(captions_np, [[next_word]], axis=1)
                captions = torch.from_numpy(captions_np).type(torch.LongTensor).to("cuda")

                output = self.forward(inputs[0], captions)
                output_np = output.cpu().detach().numpy()

                output_list.append(int(next_word))

                if next_word == 1: #if we get <end>, stop here
                    break
            
            score_average = score_total / len(output_list)
            if score_average > best_score:
                output_list_to_return = output_list
                best_score = score_average
                #print("best_score: {}".format(best_score))
            
        return output_list_to_return