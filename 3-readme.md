## 接口调用实例说明

[接口文件：classify.py](./classify.py)

### 代码结构简介

这个文件定义了一个接口类 `ViolenceClass`，这个接口代码详解如下：

```
import torch
from PIL import Image
from torchvision import transforms
from model import ViolenceClassifier
from pytorch_lightning import LightningModule

class ViolenceClass:
    def __init__(self, model_path):
        # 加载模型、设置参数等

        self.model = ViolenceClassifier()
        self.model.load_state_dict(torch.load(model_path)['state_dict']) #读入模型
        self.model.eval()

        self.transform = transforms.Compose([
            transforms.ToTensor(),	#转化为Tensor类
        ])
  
    def classify(self, imgs : torch.Tensor) -> list:
        # 图像分类
	if isinstance(imgs[0], Image.Image):	#检测图片是否事先转换为了Tensor
        	imgs = [self.transform(img).unsqueeze(0) for img in imgs] #读入图片
        with torch.no_grad():
            preds = [self.model(img).argmax().item() for img in imgs] #进行预测
        return preds
```

函数 `__init__()`：输入使用模型的路径，完成了函数初始化的工作，同时确定对即将应用的模型进行了读入操作。

函数 `classify()`：输入一个图片数组（图片规定为3 * 224 * 224）或者是n * 3 * 224 * 224大小的 Tensor，返回一个数组。数组中位置为 0 则说明模型对该图片的判断为非暴力，1 则是暴力。

### 接口调用示例

运行如下代码

```
#调用相关类，可以按需使用
import torch
from PIL import Image
from classify.py import ViolenceClass
from pytorch_lightning import LightningModule
......

#创建一个类，使用训练好的模型路径如下
classifier = ViolenceClass('./train_logs/resnet18_pretrain_test/version_2/checkpoints/resnet18_pretrain_test-epoch=13-val_loss=0.03.ckpt') 

#通过Image.open打开两个需要判断的图片
img1 = Image.open('violence_224/test/0_0000.jpg')
img2 = Image.open('violence_224/test/1_0000.jpg')

#使用classify函数对图片进行分类，再将预测结果输出
preds = classifier.classify([img1, img2])
print(preds)
```

在终端上使用 python3 运行这段代码，结果如下所示（为了方便起见，我这里直接在接口文件下运行这段代码，可以看到判断结果被输出了）。同样地，在其它文件中使用也可以。

```
(AIcourse) synmac05@synmac05:~/AIAI$ python3 ./classify.py
[0, 1]
```

此外，若将代码改为

```
classifier = ViolenceClass('./train_logs/resnet18_pretrain_test/version_2/checkpoints/resnet18_pretrain_test-epoch=13-val_loss=0.03.ckpt')
img1 = Image.open('violence_224/test/0_0000.jpg')
img2 = Image.open('violence_224/test/1_0000.jpg')

#将Image类转换为tensor类，这里使用的默认的类，如果输入了与device进行匹配的类，那么需要将源代码也作出修改（已注释），否则会出现TypeError
transform = transforms.ToTensor()
tensor1 = transform(img1).unsqueeze(0)
tensor2 = transform(img2).unsqueeze(0)
tensors = torch.stack([tensor1, tensor2])
preds = classifier.classify(tensors)

print(preds)
```

这里没有直接输入 `Image.Image`类而是输入 `Tensor`类，也可以运行，运行结果和之前一致。
