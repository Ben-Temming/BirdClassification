import torch 
import torch.nn.functional as F

'''
To do
- class (called BirdClassifier) that can be initialized with a certain model (by passing in the model) 
and then the path to an audio file can be passed into a function called predict_bird_species. 
The class then processes the audio, uses the model to predict the class and looks up the bird name associated to the class id 
and returns the bird name + (not sure) the confidence/percentage of the the prediction

-make is to that a mapping can be loaded (id to bird name) and then return bird name instead of id
'''

#rename to BirdAudioClassifier and rename model BirdAudioClassifierModel or BirdAudioClassifierCNN
class BirdAudioClassifier: 
    def __init__(self, model, confidence_threshold=0.5): 
        #get the device 
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        #move the model to the device and set it to evaluation 
        self.model = model.to(self.device)
        self.model.eval()

        #set the confidence threshold of a prediction 
        self.confidence_threshold = confidence_threshold
        

    def predict_from_spectrogram(self, spectrogram): 
        #reshape the tensor (to add the batch dimension) and move the tensor to the GPU 
        input = spectrogram.unsqueeze(0).to(self.device)

        #use torch.no_grad() for more efficiency by avoiding gradient computations 
        with torch.no_grad(): 
            #make the prediction using the model (this gives the raw logists)
            prediction = self.model(input)
        
        #convert the raw logits to probabilities so that the classifier can give a confidence 
        prediction_probs = F.softmax(prediction, dim=1)

        #get the predicted class (the index with the largest logist value)
        predicted_class = torch.argmax(prediction, dim=1).item() 
        #get the confidence value 
        prediction_confidence = prediction_probs[0, predicted_class].item()

        #if the confidence is to low, the sound is not from a bird 
        if prediction_confidence < self.confidence_threshold: 
            predicted_class = None 
            prediction_confidence = 0
            
        return predicted_class, round(prediction_confidence, 3)


    def predict_from_audio_file(self, file_path): 
        '''
        to do 
        '''
        pass 

                


