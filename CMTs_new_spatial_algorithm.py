import torch
import torch.nn as nn

from torchinfo import summary

    
class CMT(nn.Module):
    def __init__(self, vis_flag=False, ir_flag=False):
        super(CMT, self).__init__()
        self.vis_flag = vis_flag
        self.ir_flag = ir_flag
        
        self.sigmoid = nn.Sigmoid()
        
        self.spatial_query_conv = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=1, stride=1, padding=0)
        self.spatial_key_conv = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=1, stride=1, padding=0)
        
        rgb_spatial_3x3 = []
        ir_spatial_3x3 = []
        for _ in range(4): # patch_num * patch_num = 16
            rgb_spatial_3x3.append(nn.Conv2d(in_channels=128, out_channels=32, kernel_size=3, dilation=2, padding=2))
            ir_spatial_3x3.append(nn.Conv2d(in_channels=128, out_channels=32, kernel_size=3, dilation=2, padding=2))
        self.rgb_spatial_3x3 = nn.Sequential(*rgb_spatial_3x3)
        self.ir_spatial_3x3 = nn.Sequential(*ir_spatial_3x3)
    
    def image_to_patches(self, image):
        B, C, H, W = image.shape
        patch_h, patch_w = H // 2, W // 2  # 2x2 패치로 분할

        patches = image.unfold(2, patch_h, patch_h).unfold(3, patch_w, patch_w)
        patches = patches.permute(0, 1, 2, 3, 5, 4).contiguous()
        num_patches = 4
        patches = patches.view(B, C, num_patches, patch_h, patch_w)

        return patches  # (B, C, 4, H/2, W/2)
    
    def forward(self, query, key):
        
        # Spatial Transformer
        
        # Conv 적용
        spa_query = self.spatial_query_conv(query)
        spa_key = self.spatial_key_conv(key)
        ##########
        
        # Patch로 나누고, key는 GAP 적용
        before_spa_query_patches = self.image_to_patches(query)
        spa_query_patches = self.image_to_patches(spa_query)  # (B, C, patches_num, H / 2, W / 2)
        spa_key_patches = self.image_to_patches(spa_key)
        
        B, C, PATCH_NUM, H, W = spa_query_patches.size() # patch의 크기
        
        before_spa_query_patches_unfold = before_spa_query_patches.permute(0, 2, 1, 3, 4)
        spa_query_patches_unfold = spa_query_patches.permute(0, 2, 1, 3, 4) # (B, patches_num, C, H/2, W/2)
        spa_key_patches_unfold = spa_key_patches.permute(0, 2, 1, 3, 4) # (B, patches_num, C, H/2, W/2)
        
        spa_key_patches_unfold = spa_key_patches_unfold.view(B, PATCH_NUM, C, H*W)
        spa_key_patches_unfold = torch.mean(spa_key_patches_unfold, dim=3, keepdim=True) # (B, patches_num, C, 1) <- GAP 적용
        #################################
        
        # 각 patch마다의 attention 값 구하기        
        patches_num = 4
        spatial_reconstructed = []
        for query_i in range(patches_num):
            spa_query_patch_i = spa_query_patches_unfold[:, query_i, :, :, :] # [1, 32, 32, 32](B, C, patch_H, patch_W)
            before_query_patch_i = before_spa_query_patches_unfold[:, query_i, :, :, :]
            i_values = []
            
            for key_j in range(patches_num):
                spa_key_patch_j = spa_key_patches_unfold[:, key_j, :, :]
                spa_key_patch_j = spa_key_patch_j.permute(0, 2, 1)# [1, 32] patch마다의 channel GAP 값
                
                spa_query_patch_i_reshaped = spa_query_patch_i.view(B, C, H*W)
                
                relevance_map_weight_mul = torch.bmm(spa_key_patch_j, spa_query_patch_i_reshaped) # [B, 1, H*W]
                irrelevance_map_weight_mul = 1 - self.sigmoid(relevance_map_weight_mul) 
                
                irrelevance_map_weight_mul_reshaped = irrelevance_map_weight_mul.view(B, 1, H, W)
            
                spa_patches_value = irrelevance_map_weight_mul_reshaped * before_query_patch_i 
                print(spa_patches_value.size())
                
                i_values.append(spa_patches_value) # query_1 * key_{1,2,3,4}를 다 구함
            
            i_values_tensor = torch.concat(i_values, dim=1) # [B, C * 4, H / 2, W / 2]
            
            if self.vis_flag:
                i_values_tensor_3x3 = self.rgb_spatial_3x3[query_i](i_values_tensor)
            else:
                i_values_tensor_3x3 = self.ir_spatial_3x3[query_i](i_values_tensor)
                
            spatial_reconstructed.append(i_values_tensor_3x3)
        
        # B, C, H, W
        spatial_reconstructed_tensor = torch.concat(spatial_reconstructed, dim=1)
        B, C, H, W = spatial_reconstructed_tensor.size()
        spatial_reconstructed_tensor = spatial_reconstructed_tensor.view(B, 32, 4, H, W)
        spatial_reconstructed_tensor = spatial_reconstructed_tensor.view(B, 32, 2, 2, H, W)
        spatial_reconstructed_tensor = spatial_reconstructed_tensor.permute(0, 1, 4, 2, 5, 3).contiguous()
        spatial_final_value = spatial_reconstructed_tensor.view(B, 32, H * 2, W * 2)

        return spatial_final_value

def main():
    model = CMT(vis_flag=True, ir_flag=False)
    summary(model, input_size=[(2, 32, 64, 64), (2, 32, 64, 64)])
    
if __name__ == "__main__":
    main()