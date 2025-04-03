from inference.infer import *

from tqdm import tqdm
import os
from PIL import Image
import numpy as np
import torch
from torchvision import transforms

# Assuming these functions and classes are defined somewhere in your code
# from your_code import BinarySegMetrics, load_deeplab_model, utils

def get_target(mask_path,device):
    mask = Image.open(mask_path)
    mask_array = np.array(mask)
    mask_array = (mask_array > 128).astype(np.uint8)
    mask_array = mask_array * 255
    mask = Image.fromarray(mask_array.astype(np.uint8))
    mask = mask_transform(mask)
    mask = torch.squeeze(mask, 0)
    mask = mask.to(device, dtype=torch.long)
    mask = mask.float()
    return mask.cpu().numpy()

def metrics_from_dataset(checkpoint, dataset_path):
    metrics = BinarySegMetrics()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = load_deeplab_model(checkpoint, device).eval()
    sample_dir = os.path.join(dataset_path, 'samples')
    mask_dir = os.path.join(dataset_path, 'binary_masks')
    
    file_list = os.listdir(sample_dir)
    
    for i in tqdm(file_list, desc="Processing images"):
        img_path = os.path.join(sample_dir, i)
        ext = os.path.basename(img_path).split('.')[-1]
        img_name = os.path.basename(img_path)[:-len(ext) - 1]
        mask_path = os.path.join(mask_dir, img_name + '_binarymask.' + ext)
        
        target = get_target(mask_path,device)
        img = Image.open(img_path)
        
        img_transform = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        img_tensor = img_transform(img).unsqueeze(0).to(device, dtype=torch.float32)
    
        with torch.no_grad():
            output = model(img_tensor)
            output = torch.squeeze(output, dim=1)
            prob = torch.sigmoid(output).detach()
            pred = (prob > 0.5).long().cpu().numpy()[0]
            
        metrics.update(target, pred)
        
        # The following lines are commented out as they appear to be part of additional processing
        # denorm = utils.Denormalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        # img_np = img_tensor[0].detach().cpu().numpy()
        # img_np = (denorm(img_np) * 255).transpose(1, 2, 0).astype(np.uint8)
        # cleaned_mask = apply_dense_crf(img_np, np.array(decode_target(pred)))
        # prob_np = prob[0].cpu().numpy()
        # count = np.sum(prob_np > 0.5)
        # confidence = np.sum(prob_np[prob_np > 0.5]) / count if count != 0 else 0
        
    return metrics.get_results()



class BinarySegMetrics():
    """
    Binary Segmentation
    """
    def __init__(self):
        # two classes (foreground and background)
        self.n_classes = 2
        self.confusion_matrix = np.zeros((2, 2))
        # self.threshold = 0.5  # Threshold for converting probabilities to binary predictions

    def _fast_hist(self, label_true, label_pred):
        # label_pred = label_pred >= self.threshold  # Binarize predictions
        mask = (label_true >= 0) & (label_true < self.n_classes)
        hist = np.bincount(
            2 * label_true[mask].astype(int) + label_pred[mask],
            minlength=self.n_classes ** 2,
        ).reshape(self.n_classes, self.n_classes)
        return hist

    def update(self, label_trues, label_preds):
        for lt, lp in zip(label_trues, label_preds):
            self.confusion_matrix += self._fast_hist(lt.flatten(), lp.flatten())

    def get_results(self):
        """Returns accuracy score evaluation result for binary segmentation."""
        hist = self.confusion_matrix
        tn, fp, fn, tp = hist.ravel()
        
        # Metrics for foreground
        foreground_total = tp + fn
        foreground_acc = tp / foreground_total if foreground_total > 0 else 0
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        iou_foreground = tp / (tp + fn + fp) if (tp + fn + fp) > 0 else 0
        iou_background = tn / (tn + fp + fn) if (tn + fp + fn) > 0 else 0
        mean_iou = (iou_foreground + iou_background) / 2

        #overall_acc = np.diag(hist).sum() / hist.sum()

        return {
            "Foreground Acc": foreground_acc,
            "Precision": precision,
            "Recall": recall,
            "F1 Score": f1_score,
            "IoU Foreground": iou_foreground,
            "IoU Background": iou_background,
            "Mean IoU": mean_iou
        }

    def reset(self):
        self.confusion_matrix = np.zeros((self.n_classes, self.n_classes))

    @staticmethod
    def to_str(results):
        string = "\n"
        for k, v in results.items():
            string += "%s: %f\n" % (k, v)
        return string

def load_deeplab_model(ckpt, device, model_type='deeplabv3plus_mobilenet', num_classes=1, output_stride=16):
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    print("Device: %s" % device)
    
    # Create the model with EXACTLY the same configuration as training
    model = network.modeling.__dict__[model_type](num_classes, output_stride)
    
    # Don't apply separable convolution since it was False during training
    # network.convert_to_separable_conv(model.classifier)  <- COMMENT THIS OUT
    
    utils.set_bn_momentum(model.backbone, momentum=0.01)
    
    # Load the checkpoint
    checkpoint = torch.load(ckpt, map_location=torch.device('cuda'), weights_only=False)
    model.load_state_dict(checkpoint["model_state"])
    
    model = nn.DataParallel(model)
    model.to(device)
    
    print("Resume model from %s" % ckpt)
    del checkpoint
    print(f"Model device: {next(model.parameters()).device}")
    return model
    
kl_path = 'saved_models/kl_check_temp_2_weighted.pth'
#model = load_deeplab_model(kl_path,'cuda')
metrics_from_dataset(kl_path , 'india_sam_dino_annotations_large')