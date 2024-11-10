import torchvision.models as models
import torch.nn as nn

def Deepfake(use_hidden_layer=True, dropout=0.9):
    model = models.resnet18(pretrained=True)

    # Uncomment to freeze pre-trained layers
    # for param in model.parameters():
    #     param.requires_grad = False

    in_features = model.fc.in_features
    print(f'Input feature dim: {in_features}')

    if use_hidden_layer:
        model.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_features, in_features // 2),
            nn.ReLU(),
            nn.BatchNorm1d(in_features // 2),
            nn.Dropout(dropout),
            nn.Linear(in_features // 2, 2)
        )

    else:
        model.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_features, 1)
        )
    return model