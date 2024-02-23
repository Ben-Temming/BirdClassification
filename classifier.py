import os
import torch 
import torch.nn.functional as F

from collections import Counter 
from utils import AudioUtil


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
        

    def predict_class_from_spectrogram(self, spectrogram): 
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


    def predict_class_from_audio_file(self, file_path): 
        #check that the file exists
        if not os.path.exists(file_path): 
            raise Exception(f"The file: {file_path} does not exist")

        #load the waveform 
        waveform, sample_rate = AudioUtil.open_file(file_path)

        #split the audio into 5-second chunks
        waveform_chunks = AudioUtil.split_audio_into_chunks(waveform=waveform, sample_rate=sample_rate, 
                                                          chunk_duration_s = 5.0, padded=True)


        #predict the class for each of the 5-second chunks
        predictions_list= []
        confidence_list = []
        for waveform_chunk in waveform_chunks: 
            #create spectrogram 
            spectrogram = AudioUtil.get_spectrogram(waveform_chunk)
            #predict the class for the spectrogram
            pred_class, pred_confidence = self.predict_class_from_spectrogram(spectrogram=spectrogram)
            predictions_list.append(pred_class)
            confidence_list.append(pred_confidence)

        #find the most common class
        class_counts = Counter(predictions_list)
        most_common_class = class_counts.most_common(1)[0][0]

        #calculate the average confidence of the most common class 
        num_preds = class_counts.most_common(1)[0][1]
        total_confidence = 0
        
        for pred, conf in zip(predictions_list, confidence_list): 
            if pred == most_common_class: 
                total_confidence += conf
        
        average_confidence = total_confidence / num_preds if num_preds > 0 else 0

        #return the most common (and thus likely) prediction and the average confidence for that class
        return most_common_class, round(average_confidence, 3)


    def predict_classes_from_audio_file(self, file_path):
        '''
        adaption of predict_class_from_audio_file that 
        returns multiple bird classes if the confidence 
        for each is high, this could be the case in a real-world recording 
        where multiple birds might be present at the same time
        '''
        pass


                


