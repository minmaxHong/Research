import torch
import torch.nn as nn

from torchinfo import summary

class Own_TF(nn.Module):
    def __init__(self):
        super(Own_TF, self).__init__()

        self.channel_conv_1 = nn.Conv2d(in_channels=32, out_channels=1, kernel_size=1, stride=1, padding=0)
        self.channel_conv_2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=1, stride=1, padding=0)

        self.spatial_conv_1 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=1, stride=1, padding=0)
        self.spatial_conv_2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=1, stride=1, padding=0)
        self.conv11 = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=1, stride=1, padding=0)

    def forward(self, x):

        ######################################## channel
        chn_key = self.channel_conv_1(x)  # H * W * 1
        chn_query = self.channel_conv_2(x)  # H * W * 32

        B, C, H, W = chn_query.size()

        chn_query_unfold = chn_query.view(B, C, H * W)  # HW * 32
        chn_key_unfold = chn_key.view(B, 1, H * W)  # HW * 1

        chn_key_unfold = chn_key_unfold.permute(0, 2, 1)

        chn_query_relevance = torch.bmm(chn_query_unfold, chn_key_unfold)
        chn_query_relevance_ = torch.sigmoid(chn_query_relevance) #softmax?
        chn_query_relevance_ = 1 - chn_query_relevance_ #irrelevance map(channel)
        inv_chn_query_relevance_ = chn_query_relevance_.unsqueeze(2)
        chn_value_final = inv_chn_query_relevance_ * x

        ######################################## spatial
        spa_key = self.spatial_conv_1(x)  # H * W * 32
        spa_query = self.spatial_conv_2(x)  # H * W *32

        B, C, H, W = spa_query.size()

        spa_query_unfold = spa_query.view(B, H * W, C)  # HW * 32
        spa_key_unfold = spa_key.view(B, H * W, C)  # HW * 32

        spa_key_unfold = torch.mean(spa_key_unfold, dim=1)
        spa_key_unfold = spa_key_unfold.unsqueeze(2)

        spa_query_relevance = torch.bmm(spa_query_unfold, spa_key_unfold)
        spa_query_relevance = torch.sigmoid(spa_query_relevance) #softmax?

        inv_spa_query_relevance = 1 - spa_query_relevance #irrelevance map(spatial)
        inv_spa_query_relevance_ = inv_spa_query_relevance.permute(0, 2, 1)
        inv_spa_query_relevance_ = inv_spa_query_relevance_.view(B, 1, H, W)
        spa_value_final = inv_spa_query_relevance_ * x

        key_relevance = torch.cat([chn_value_final, spa_value_final], dim =1)
        key_relevance = self.conv11(key_relevance)

        return key_relevance


    



def main():
    model = Own_TF()
    summary(model, input_size=[(2, 32, 64, 64)])
    
if __name__ == "__main__":
    main()