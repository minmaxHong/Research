import torch
import torch.nn as nn

from torchinfo import summary




def main():
    model = CMT()
    summary(model, input_size=[(2, 32, 64, 64), (2, 32, 64, 64)])
    
if __name__ == "__main__":
    main()