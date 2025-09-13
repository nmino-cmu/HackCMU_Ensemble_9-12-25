import torchaudio

class audiofile:
    def __init__(self, file_name):
        self.file_location = file_name
        self.tensor, self.sample_rate = torchaudio.load(self.file_location)
    
    def get_tensor(self):
        return self.tensor

    def get_sample_rate(self):
        return self.sample_rate
    
    def get_file_location(self):
        return self.file_location
