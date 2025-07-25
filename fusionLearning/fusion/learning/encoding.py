from torch import nn



class SinusoidalEncoder(nn.Module): 
    def __init__(self, embedding_dim : int): 
        super().__init__()

        self.register_buffer('position_encoding', self._compute_position_encoding(embedding_dim), persistent=False)
        
    def _compute_position_encoding(self, embedding_dim : int) -> torch.Tensor:
        """
        computes the positional encoding for a single mask in the predicted masks tensor
        """
        num_positions = 224
        position_encodings = torch.zeros((num_positions, embedding_dim))
        for pos in range(num_positions):
            for i in range(embedding_dim):
                if i % 2 == 0:
                    position_encodings[pos, i] = torch.sin(pos / (10000 ** (i // 2)))
                else:
                    position_encodings[pos, i] = torch.cos(pos / (10000 ** ((i-1) // 2)))
        return position_encodings


        