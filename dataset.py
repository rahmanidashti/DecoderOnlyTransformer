import torch

class Dataset():

    def __init__(self):
        self.token_to_id = {'what': 0, 'is': 1, 'statquest': 2, 'awosome': 3, '<EOS>': 4}
        self.id_to_token = {id: token for token, id in self.token_to_id.items()}

    def get_data(self):
        inputs = torch.tensor([[self.token_to_id['what'], self.token_to_id['is'], self.token_to_id['statquest'], self.token_to_id['<EOS>'], self.token_to_id['awosome']],
                               [self.token_to_id['awosome'], self.token_to_id['is'], self.token_to_id['what'], self.token_to_id['<EOS>'], self.token_to_id['awosome']]])

        labels = torch.tensor([[self.token_to_id['is'], self.token_to_id['statquest'], self.token_to_id['<EOS>'], self.token_to_id['awosome'], self.token_to_id['<EOS>']],
                               [self.token_to_id['is'], self.token_to_id['what'], self.token_to_id['<EOS>'], self.token_to_id['awosome'], self.token_to_id['<EOS>']]])
        
        return inputs, labels