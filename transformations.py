from torchvision import datasets, transforms as T

######################
## Transformations ###
######################
# Define the transform for our images - resizes/crops them to 224x224, normalizes (required for ImageNet)
toSizeCenter = T.Compose([
    T.Resize(256),
    T.CenterCrop(224)
])

transformImg = T.Compose([
    T.Resize(256),
    T.CenterCrop(224)
    , T.ToTensor()
    , T.Normalize(mean=[0.485, 0.456, 0.406],
                  std=[0.229, 0.224, 0.225])
])

toTensor = T.Compose([
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225])
])

#################################