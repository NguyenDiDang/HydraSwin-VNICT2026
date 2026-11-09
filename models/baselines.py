"""
Comparative Multi-Task Vision Baselines:
MobileNetV3, ConvNeXt-Tiny, ViT-Tiny, and HydraResNet (ResNet-50).
"""

import torch
import torch.nn as nn
import timm

class HydraMultiTaskBaseline(nn.Module):
    """
    Generic multi-task architecture supporting any timm vision backbone.
    Used for comparative empirical benchmarks.
    """
    def __init__(self, backbone_name="mobilenetv3_large_100", num_classes_dict=None, pretrained=True):
        super(HydraMultiTaskBaseline, self).__init__()
        
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
        
        # Initialize backbone with classifier pooling removed
        self.backbone = timm.create_model(backbone_name, pretrained=pretrained, num_classes=0)
        feat_dim = self.backbone.num_features
        
        self.gender_head = nn.Linear(feat_dim, num_classes_dict['gender'])
        self.master_cat_head = nn.Linear(feat_dim, num_classes_dict['master_cat'])
        self.sub_cat_head = nn.Linear(feat_dim, num_classes_dict['sub_cat'])
        self.article_type_head = nn.Linear(feat_dim, num_classes_dict['article_type'])
        self.base_colour_head = nn.Linear(feat_dim, num_classes_dict['base_colour'])
        self.season_head = nn.Linear(feat_dim, num_classes_dict['season'])

    def forward(self, x, topk=1):
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
