from typing import List

import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torchvision.models import get_model


class DogClassification(nn.Module):
    def __init__(self, classes: List = None, num_classes: int = 0, model_type: str = "resnet50", weights="DEFAULT", dropout_ratio: float = 0.5) -> None:
        super(DogClassification, self).__init__()
        if classes and num_classes:
            assert len(classes) == num_classes, f"lenght classes and num_classes don't match {len(classes)} != {num_classes}"
        self.classes = classes
        if self.classes:
            num_classes = len(self.classes)
        
        # Load pretrained
        self.backbone = get_model(model_type, weights=weights)
            
        # Get the classifier/last layer
        name_classifier = list(self.backbone.named_children())[-1][0]
        classifier = getattr(self.backbone, name_classifier)
        
        # Change output dim of the classifier to number of classes
        if isinstance(classifier, nn.Sequential):
            classifier[-1].out_features = num_classes
        elif isinstance(classifier, nn.Linear):
            setattr(
                self.backbone,
                name_classifier,
                nn.Sequential(
                    nn.Linear(classifier.in_features, 256),
                    nn.ReLU(),
                    nn.Dropout(p=dropout_ratio),
                    nn.Linear(256, num_classes))
            )
        else:
            raise Exception(f"Not support {model_type} model")
        
        # Only retrain the classifier, freeze the feature extractor
        self.freeze_except_layers([name_classifier])
    
    def freeze_except_layers(self, except_layers: List[str]):
        for name, child in self.backbone.named_children():
            if name in except_layers:
                for param in child.parameters():
                    param.requires_grad = True
            else:
                for param in child.parameters():
                    param.requires_grad = False
    
    def unfreeze_all_layers(self):
        for param in self.parameters():
            param.requires_grad = True
    
    def forward(self, x):
        return self.backbone(x)
    
    def predict(self, image):
        self.eval()
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        image = transform(image).unsqueeze(0)
        if torch.cuda.is_available():
            image = image.cuda()
            self = self.cuda()
        output = self(image)
        _, predicted_idx = torch.max(output, 1)
        if self.classes:
            return self.classes[predicted_idx]
        else:
            return predicted_idx
