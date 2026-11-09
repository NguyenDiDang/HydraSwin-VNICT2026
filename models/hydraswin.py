"""
HydraSwin: Multi-Task Architecture based on Swin Transformer Backbone
Official Implementation for VNICT 2026 Camera Ready Paper.
"""

import torch
import torch.nn as nn
import timm

class HydraSwin(nn.Module):
    """
    HydraSwin: Unified multi-task feature extractor leveraging a pre-trained
    Swin-Tiny backbone with six independent classification heads.
    """
    def __init__(self, num_classes_dict=None, pretrained=True):
        super(HydraSwin, self).__init__()
        
        # Default classes matching Kaggle Fashion Product dataset benchmark
        if num_classes_dict is None:
            num_classes_dict = {
                "gender": 5,
                "master_cat": 7,
                "sub_cat": 45,
                "article_type": 142,
                "base_colour": 46,
                "season": 4
            }
        self.num_classes_dict = num_classes_dict
        
        # Swin-Tiny Backbone (swin_tiny_patch4_window7_224)
        # 768-dimensional latent vector output
        self.backbone = timm.create_model("swin_tiny_patch4_window7_224", pretrained=pretrained, num_classes=0)
        feat_dim = self.backbone.num_features  # 768
        
        # Six specialized parallel classification heads
        self.gender_head = nn.Linear(feat_dim, num_classes_dict['gender'])
        self.master_cat_head = nn.Linear(feat_dim, num_classes_dict['master_cat'])
        self.sub_cat_head = nn.Linear(feat_dim, num_classes_dict['sub_cat'])
        self.article_type_head = nn.Linear(feat_dim, num_classes_dict['article_type'])
        self.base_colour_head = nn.Linear(feat_dim, num_classes_dict['base_colour'])
        self.season_head = nn.Linear(feat_dim, num_classes_dict['season'])

    def forward(self, x, topk=1):
        """
        Forward pass returning both raw task logits and top-k predictions.
        """
        features = self.backbone(x)
        logits = {
            "gender": self.gender_head(features),
            "master_cat": self.master_cat_head(features),
            "sub_cat": self.sub_cat_head(features),
            "article_type": self.article_type_head(features),
            "base_colour": self.base_colour_head(features),
            "season": self.season_head(features)
        }
        
        preds_topk = {
            task: torch.topk(logit, k=min(topk, logit.size(1)), dim=1)
            for task, logit in logits.items()
        }
        return logits, preds_topk
