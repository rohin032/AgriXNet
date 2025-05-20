import torch
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import cv2
from pytorch_grad_cam.utils.image import show_cam_on_image

# --- CAM (Class Activation Map) ---
def generate_cam(model, image_path, target_layer_name='layer4', target_class_index=None):
    model.eval()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    img = Image.open(image_path).convert('RGB')
    input_tensor = transform(img).unsqueeze(0).to(device)
    # Forward pass to get output
    output = model(input_tensor)
    if target_class_index is None:
        target_class_index = output.argmax(dim=1).item()
    # Get the target layer
    layer = dict([*model.named_modules()])[target_layer_name]
    # Register hook to get feature maps
    feature_maps = []
    def hook_fn(module, input, output):
        feature_maps.append(output.detach())
    hook = layer.register_forward_hook(hook_fn)
    # Forward pass again to get feature maps
    _ = model(input_tensor)
    hook.remove()
    fmap = feature_maps[0].cpu().numpy()[0]
    # Get weights from the classifier layer
    params = list(model.parameters())
    weight_softmax = params[-2].cpu().detach().numpy()  # Last FC layer weights
    cam = np.zeros(fmap.shape[1:], dtype=np.float32)
    for i, w in enumerate(weight_softmax[target_class_index]):
        cam += w * fmap[i, :, :]
    cam = np.maximum(cam, 0)
    cam = cv2.resize(cam, (224, 224))
    cam = cam - np.min(cam)
    cam = cam / np.max(cam)
    img_np = np.array(img.resize((224, 224))) / 255.0
    cam_img = show_cam_on_image(img_np, cam, use_rgb=True)
    return cam_img

# --- Saliency Map ---
def generate_saliency_map(model, image_path, target_class_index=None):
    model.eval()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    img = Image.open(image_path).convert('RGB')
    img_tensor = transform(img).unsqueeze(0).to(device)
    img_tensor.requires_grad_()
    output = model(img_tensor)
    if target_class_index is None:
        target_class_index = output.argmax(dim=1).item()
    target_output = output[0, target_class_index]
    target_output.backward()
    saliency = img_tensor.grad.abs().squeeze().cpu().numpy()
    saliency = np.transpose(saliency, (1, 2, 0)).mean(axis=2)
    saliency = (saliency - saliency.min()) / (saliency.max() - saliency.min() + 1e-8)
    saliency = cv2.resize(saliency, (224, 224))
    img_np = np.array(img.resize((224, 224))) / 255.0
    saliency_img = show_cam_on_image(img_np, saliency, use_rgb=True)
    return saliency_img 