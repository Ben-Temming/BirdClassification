import os 
import numpy as np
import torch
from torch.utils.data import Dataset
import pandas as pd
import torchaudio

#creating a audio utility class that can be used to load an audio file 
class AudioUtil: 
    '''
    Load an audio file. Return the signal as a pytorch tensor and the sample rate
    '''
    @staticmethod 
    def open_file(audio_file): 
        waveform, sample_rate = torchaudio.load(audio_file)
        return waveform, sample_rate

    '''
    Create a spectrogram (frequency vs time)
    '''
    @staticmethod
    def get_spectrogram(waveform, n_fft = 800, hop_length=400, win_length=800, center=True): 
        #create transformer 
        transform = torchaudio.transforms.Spectrogram(n_fft=n_fft, hop_length=hop_length, 
                                                      win_length=win_length, center=center)
        #create spectrogram
        spectrogram = transform(waveform)

        #convert the spectrogram to decibel 
        spectrogram = torchaudio.transforms.AmplitudeToDB()(spectrogram)
       
        return spectrogram

    @staticmethod
    def split_audio_into_chunks(waveform, sample_rate, chunk_duration_s, padded=True): 
        #caclualte duration of the signal 
        duration_in_seconds = waveform.size()[1] / sample_rate
        #calculate the number of chunks
        num_chunks = int(np.ceil(duration_in_seconds / chunk_duration_s))
    
        #split the waveform into chunks
        waveform_chunks = []
        for i in range(num_chunks):
            #calculate the start and end index
            start_idx = int(i * chunk_duration_s * sample_rate)
            end_idx = int((i + 1) * chunk_duration_s * sample_rate)
    
            if padded: 
                if i == num_chunks - 1 and end_idx > waveform.size(1):
                    pad_size = end_idx - waveform.size(1)
                    #only pad if the last part fils at least half the time period 
                    if pad_size < ((chunk_duration_s * sample_rate)/2): 
                        padded_chunk = torch.nn.functional.pad(waveform[:, start_idx:], (0, pad_size), 'constant', 0)
                        waveform_chunks.append(padded_chunk)
                else:
                    chunk = waveform[:, start_idx:end_idx]
                    waveform_chunks.append(chunk)
            else:
                chunk = waveform[:, start_idx:end_idx]
                waveform_chunks.append(chunk)
    
        return waveform_chunks


#create a custom dataset 
class AudioDataSet(Dataset): 
    def __init__(self, df_metadata): 
        self.df_metadata = df_metadata

    def __len__(self): 
        return len(self.df_metadata)

    '''
    get an item from the dataset
    '''
    def __getitem__(self, index): 
        #get the path of the item at the specified index
        audio_file_path = self.df_metadata.loc[index, "path"]
        #get the class id 
        class_id = self.df_metadata.loc[index, "class_id"]

        #get the spectrogram 
        waveform, sample_rate = AudioUtil.open_file(audio_file_path)        
        spectrogram = AudioUtil.get_spectrogram(waveform)

        return spectrogram, class_id


