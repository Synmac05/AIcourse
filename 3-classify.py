import sys

current_dir = sys.path.insert(0, "./3-SupportingFiles")

import torch
from PIL import Image
from torchvision import transforms
from model import ViolenceClassifier
from pytorch_lightning import LightningModule

class ViolenceClass:
    def __init__(self, model_path):
        # 加载模型、设置参数等
        # 这个函数会根据设备使用合适的Tensor数组，如果后面输入的Tensor进行过设备匹配，这里需要将注释删掉
        #self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = ViolenceClassifier()

        self.model.load_state_dict(torch.load(model_path)['state_dict'])

        #self.model = self.model.to(self.device)
        self.model.eval()

        self.transform = transforms.Compose([
            transforms.ToTensor(),
        ])
    
    def misc(self, others):
        # 其他处理函数
        return None
        
    def classify(self, imgs : torch.Tensor) -> list:
        # 图像分类
        if isinstance(imgs[0], Image.Image):
            imgs = [self.transform(img).unsqueeze(0) for img in imgs]
        with torch.no_grad():
            preds = [self.model(img).argmax().item() for img in imgs]
        return preds
    

# classifier = ViolenceClass('./3-SupportingFiles/train_logs/resnet18_pretrain_test/version_3/checkpoints/resnet18_pretrain_test-epoch=24-val_loss=0.04.ckpt')
# img1 = Image.open('./3-SupportingFiles/testjpg/tester/0_3477.jpg')
# img2 = Image.open('./3-SupportingFiles/testjpg/tester/0_3478.jpg')
# img3 = Image.open('./3-SupportingFiles/testjpg/tester/0_3479.jpg')

# transform = transforms.ToTensor()
# tensor1 = transform(img1).unsqueeze(0)
# tensor2 = transform(img2).unsqueeze(0)
# tensor3 = transform(img3).unsqueeze(0)
# tensors = torch.stack([tensor1, tensor2, tensor3])
# preds = classifier.classify(tensors)

# # preds = classifier.classify([img1, img2])

# print(preds)
