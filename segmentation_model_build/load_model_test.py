from inference.infer import *

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
model = load_deeplab_model(kl_path,'cuda')