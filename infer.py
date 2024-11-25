import torch
import segmentation_models_pytorch as smp
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
import argparse
import cv2
import numpy as np

model = smp.UnetPlusPlus(
    encoder_name="resnet34",
    encoder_weights="imagenet",
    in_channels=3,
    classes=3
)
checkpoint = torch.load('colorization_model.pth')
model.load_state_dict(checkpoint['model'])
device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
model.to(device)

val_transform = A.Compose([
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2(),
])

model.eval()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_path', type=str, required=True, help='Path to input image')
    return parser.parse_args()


def predict_image(image_path, model, transform, device, resize=(256, 256)):
    # Read image
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Resize image to match training size
    image = cv2.resize(image, resize)

    # Apply transforms
    if transform:
        transformed = transform(image=image)
        input_tensor = transformed['image'].unsqueeze(0).to(device)

    # Predict
    with torch.no_grad():
        output = model(input_tensor)
        output = torch.softmax(output, dim=1)
        prediction = output.squeeze().cpu().numpy()

    # Convert prediction to RGB image
    prediction = np.argmax(prediction, axis=0)

    # Map class indices to colors based on the original mask creation
    # Class 0: Background (black)
    # Class 1: Red mask
    # Class 2: Green mask
    color_map = {
        0: [0, 0, 0],  # Background
        1: [255, 0, 0],  # Red mask
        2: [0, 255, 0]  # Green mask
    }

    segmentation = np.zeros((prediction.shape[0], prediction.shape[1], 3), dtype=np.uint8)
    for class_idx, color in color_map.items():
        segmentation[prediction == class_idx] = color

    return segmentation


if __name__ == '__main__':
    args = parse_args()

    # Predict and save
    segmentation = predict_image(args.image_path, model, val_transform, device, resize=(256, 256))
    output_path = 'segmentation_output.png'
    cv2.imwrite(output_path, cv2.cvtColor(segmentation, cv2.COLOR_RGB2BGR))
    print(f"Segmentation saved to {output_path}")
