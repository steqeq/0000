# Inception V3 with PyTorch

Pull content from
<https://docs.amd.com/bundle/ROCm-Deep-Learning-Guide-v5.4.1/page/Deep_Learning_Training.html>.
Ignore training description.

# Deep Learning Training

Deep Learning models are designed to capture the complexity of the problem and the underlying data. These models are "deep," comprising multiple component layers. Training is finding the best parameters for each model layer to achieve a well-defined objective.

The training data consists of input features in supervised learning, similar to what the learned model is expected to see during the evaluation or inference phase. The target output is also included, which serves to teach the model. A loss metric is defined as part of training that evaluates the model's performance during the training process.

Training also includes the choice of an optimization algorithm that reduces the loss by adjusting the model's parameters. Training is an iterative process where training data is fed in, usually split into different batches, with the entirety of the training data passed during one training epoch. Training usually is run for multiple epochs.

## Training Phases
Training occurs in multiple phases for every batch of training data. Table 2 provides an explanation of the types of training phases.

||
|:--:|
| **Table 2.  Types of Training Phases**|
||

| Types of Phases |  | 
| ----------- | ----------- | 
| Forward Pass | The input features are fed into the model, whose parameters may be randomly initialized initially. Activations (outputs) of each layer are retained during this pass to help in the loss gradient computation during the backward pass. |
| Loss Computation | The output is compared against the target outputs, and the loss is computed. |
| Backward Pass | The loss is propagated backward, and the model's error gradients are computed and stored for each trainable parameter. |
| Optimization Pass | The optimization algorithm updates the model parameters using the stored error gradients. |

Training is different from inference, particularly from the hardware perspective. Table 3 shows the contrast between training and inference.

||
|:--:|
| **Table 3.  Training vs. Inference**|
||

| Training | Inference | 
| ----------- | ----------- | 
| Training is measured in hours/days. | The inference is measured in minutes. |
| Training is generally run offline in a data center or cloud setting. | The inference is made on edge devices. |
| The memory requirements for training are higher than inference due to storing intermediate data, such as activations and error gradients. | The memory requirements are lower for inference than training. |
| Data for training is available on the disk before the training process and is generally significant. The training performance is measured by how fast the data batches can be processed. | Inference data usually arrive stochastically, which may be batched to improve performance. Inference performance is generally measured in throughput speed to process the batch of data and the delay in responding to the input (latency). |

Different quantization data types are typically chosen between training (FP32, BF16) and inference (FP16, INT8). The computation hardware has different specializations from other datatypes, leading to improvement in performance if a faster datatype can be selected for the corresponding task.

## Case Studies
The following sections contain case studies for the Inception v3 model.

### Inception v3 with PyTorch
Convolution Neural Networks are forms of artificial neural networks commonly used for image processing. One of the core layers of such a network is the convolutional layer, which convolves the input with a weight tensor and passes the result to the next layer. Inception v3 [1] is an architectural development over the ImageNet competition-winning entry, AlexNet, using more profound and broader networks while attempting to meet computational and memory budgets.

The implementation uses PyTorch as a framework. This case study utilizes torchvision [2], a repository of popular datasets and model architectures, for obtaining the model. Torchvision also provides pretrained weights as a starting point to develop new models or fine-tune the model for a new task.

#### Evaluating a Pretrained Model

The Inception v3 model introduces a simple image classification task with the pretrained model. This does not involve training but utilizes an already pretrained model from torchvision.

This example is adapted from the PyTorch research hub page on Inception v3 [3].

Follow these steps:
1. Run the PyTorch ROCm-based Docker image or refer to the section [Installing PyTorch](https://docs.amd.com/bundle/ROCm-Deep-Learning-Guide-v5.4-/page/Frameworks_Installation.html#d1667e113) for setting up a PyTorch environment on ROCm.

```
docker run -it -v $HOME:/data --cap-add=SYS_PTRACE --security-opt seccomp=unconfined --device=/dev/kfd --device=/dev/dri --group-add video --ipc=host --shm-size 8G rocm/pytorch:latest
```

2. Run the Python shell and import packages and libraries for model creation.

```
import torch
import torchvision
```

3. Set the model in evaluation mode. Evaluation mode directs PyTorch not to store intermediate data, which would have been used in training.

```
model = torch.hub.load('pytorch/vision:v0.10.0', 'inception_v3', pretrained=True)
model.eval()
```

4. Download a sample image for inference.

```
import urllib
url, filename = ("https://github.com/pytorch/hub/raw/master/images/dog.jpg", "dog.jpg")
try: urllib.URLopener().retrieve(url, filename)
except: urllib.request.urlretrieve(url, filename)
```

5. Import torchvision and PIL Image support libraries.

```
from PIL import Image
from torchvision import transforms
input_image = Image.open(filename)
```

6. Apply preprocessing and normalization.

```
preprocess = transforms.Compose([
    transforms.Resize(299),
    transforms.CenterCrop(299),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
```

7. Use input tensors and unsqueeze them later.

```
input_tensor = preprocess(input_image)
input_batch = input_tensor.unsqueeze(0)
if torch.cuda.is_available():
    input_batch = input_batch.to('cuda')
    model.to('cuda')
```

8. Find out probabilities.

```
with torch.no_grad():
    output = model(input_batch)
print(output[0])
probabilities = torch.nn.functional.softmax(output[0], dim=0)
print(probabilities)
```

9. To understand the probabilities, download and examine the Imagenet labels.

```
wget https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt
```

10. Read the categories and show the top categories for the image.

```
with open("imagenet_classes.txt", "r") as f:
    categories = [s.strip() for s in f.readlines()]
top5_prob, top5_catid = torch.topk(probabilities, 5)
for i in range(top5_prob.size(0)):
    print(categories[top5_catid[i]], top5_prob[i].item())
```

#### Training Inception v3
The previous section focused on downloading and using the Inception v3 model for a simple image classification task. This section walks through training the model on a new dataset.

Follow these steps:

1. Run the PyTorch ROCm Docker image or refer to the section [Installing PyTorch](https://docs.amd.com/bundle/ROCm-Deep-Learning-Guide-v5.4-/page/Frameworks_Installation.html#d1667e113) for setting up a PyTorch environment on ROCm.

```
docker pull rocm/pytorch:latest
docker run -it --cap-add=SYS_PTRACE --security-opt seccomp=unconfined --device=/dev/kfd --device=/dev/dri --group-add video --ipc=host --shm-size 8G rocm/pytorch:latest
```

2. Download an imagenet database. For this example, the tiny-imagenet-200 [4], a smaller ImageNet variant with 200 image classes and a training dataset with 100,000 images, was downsized to 64x64 color images.

```
wget http://cs231n.stanford.edu/tiny-imagenet-200.zip
```

3. Process the database to set the validation directory to the format expected by PyTorch DataLoader.

4. Run the following script:

```
import io
import glob
import os
from shutil import move
from os.path import join
from os import listdir, rmdir
target_folder = './tiny-imagenet-200/val/'
val_dict = {}
with open('./tiny-imagenet-200/val/val_annotations.txt', 'r') as f:
    for line in f.readlines():
        split_line = line.split('\t')
        val_dict[split_line[0]] = split_line[1]
 
paths = glob.glob('./tiny-imagenet-200/val/images/*')
for path in paths:
    file = path.split('/')[-1]
    folder = val_dict[file]
    if not os.path.exists(target_folder + str(folder)):
        os.mkdir(target_folder + str(folder))
        os.mkdir(target_folder + str(folder) + '/images')
 
for path in paths:
    file = path.split('/')[-1]
    folder = val_dict[file]
    dest = target_folder + str(folder) + '/images/' + str(file)
    move(path, dest)
 
rmdir('./tiny-imagenet-200/val/images')
```

5. Open a Python shell.

6. Import dependencies, including torch, OS, and torchvision.

```
import torch
import os
import torchvision 
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
```

7. Set parameters to guide the training process.

:::{note}
The device is set to "cuda". In PyTorch, "cuda" is a generic keyword to denote a GPU. 
:::

```
device = "cuda"
```

8. Set the data_path to the location of the training and validation data. In this case, the tiny-imagenet-200 is present as a subdirectory to the current directory.

```
data_path = "tiny-imagenet-200"
```

The training image size is cropped for input into Inception v3.

```
train_crop_size = 299
```

9. To smooth the image, use bilinear interpolation, a resampling method that uses the distance weighted average of the four nearest pixel values to estimate a new pixel value.

```
interpolation = "bilinear" 
```

The next parameters control the size to which the validation image is cropped and resized.

```
val_crop_size = 299
val_resize_size = 342
```

The pretrained Inception v3 model is chosen to be downloaded from torchvision.

```
model_name = "inception_v3" 
pretrained = True
```

During each training step, a batch of images is processed to compute the loss gradient and perform the optimization. In the following setting, the size of the batch is determined.

```
batch_size = 32
```

This refers to the number of CPU threads the data loader uses to perform efficient multiprocess data loading.

```
num_workers = 16
```

The PyTorch optim package provides methods to adjust the learning rate as the training progresses. This example uses the StepLR scheduler, which decays the learning rate by lr_gamma at every lr_step_size number of epochs.

```
learning_rate = 0.1
momentum = 0.9
weight_decay = 1e-4
lr_step_size = 30
lr_gamma = 0.1
```

:::{note}
One training epoch is when the neural network passes an entire dataset forward and backward. 
:::

```
epochs = 90
```

 The train and validation directories are determined.

```
train_dir = os.path.join(data_path, "train")
val_dir = os.path.join(data_path, "val")
```

10. Set up the training and testing data loaders.

```
interpolation = InterpolationMode(interpolation)
 
TRAIN_TRANSFORM_IMG = transforms.Compose([
Normalizaing and standardardizing the image    
transforms.RandomResizedCrop(train_crop_size, interpolation=interpolation),
    transforms.PILToTensor(),
    transforms.ConvertImageDtype(torch.float),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225] )
    ])
dataset = torchvision.datasets.ImageFolder(
    train_dir,
    transform=TRAIN_TRANSFORM_IMG
)
TEST_TRANSFORM_IMG = transforms.Compose([
    transforms.Resize(val_resize_size, interpolation=interpolation),
    transforms.CenterCrop(val_crop_size),
    transforms.PILToTensor(),
    transforms.ConvertImageDtype(torch.float),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225] )
    ])
 
dataset_test = torchvision.datasets.ImageFolder( 
    val_dir, 
    transform=TEST_TRANSFORM_IMG
)
 
print("Creating data loaders")
train_sampler = torch.utils.data.RandomSampler(dataset)
test_sampler = torch.utils.data.SequentialSampler(dataset_test)
 
data_loader = torch.utils.data.DataLoader(
    dataset,
    batch_size=batch_size,
    sampler=train_sampler,
    num_workers=num_workers,
    pin_memory=True
)
 
data_loader_test = torch.utils.data.DataLoader(
    dataset_test, batch_size=batch_size, sampler=test_sampler, num_workers=num_workers, pin_memory=True
)
```

:::{note}
Use torchvision to obtain the Inception v3 model. Use the pretrained model weights to speed up training.
:::

```
print("Creating model")
print("Num classes = ", len(dataset.classes))
model = torchvision.models.__dict__[model_name](pretrained=pretrained)
```

11. Adapt Inception v3 for the current dataset. Tiny-imagenet-200 contains only 200 classes, whereas Inception v3 is designed for 1,000-class output. The last layer of Inception v3 is replaced to match the output features required.

```
model.fc = torch.nn.Linear(model.fc.in_features, len(dataset.classes))
model.aux_logits = False
model.AuxLogits = None
```

12. Move the model to the GPU device.

```
model.to(device)
```

13. Set the loss criteria. For this example, Cross Entropy Loss [5] is used.

```
criterion = torch.nn.CrossEntropyLoss()
```

14. Set the optimizer to Stochastic Gradient Descent.

```
optimizer = torch.optim.SGD(
    model.parameters(),
    lr=learning_rate,
    momentum=momentum,
    weight_decay=weight_decay
)
```

15. Set the learning rate scheduler.

```
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=lr_step_size, gamma=lr_gamma)
```

16. Iterate over epochs. Each epoch is a complete pass through the training data.

```
print("Start training")
for epoch in range(epochs):
    model.train()
    epoch_loss = 0
    len_dataset = 0
```

17. Iterate over steps. The data is processed in batches, and each step passes through a full batch.

```
for step, (image, target) in enumerate(data_loader):
```

18. Pass the image and target to the GPU device.

```
image, target = image.to(device), target.to(device)
```

The following is the core training logic:

a. The image is fed into the model.

b. The output is compared with the target in the training data to obtain the loss.

c. This loss is back propagated to all parameters that require optimization.

d. The optimizer updates the parameters based on the selected optimization algorithm.

```
        output = model(image)
        loss = criterion(output, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```

The epoch loss is updated, and the step loss prints.

```
        epoch_loss += output.shape[0] * loss.item()
        len_dataset += output.shape[0];
        if step % 10 == 0:
            print('Epoch: ', epoch, '| step : %d' % step, '| train loss : %0.4f' % loss.item() )
    epoch_loss = epoch_loss / len_dataset
    print('Epoch: ', epoch, '| train loss :  %0.4f' % epoch_loss )
```

The learning rate is updated at the end of each epoch.

```
lr_scheduler.step()
```

After training for the epoch, the model evaluates against the validation dataset. 

```
model.eval()
    with torch.inference_mode():
        running_loss = 0
        for step, (image, target) in enumerate(data_loader_test):
            image, target = image.to(device), target.to(device)
            
            output = model(image)
            loss = criterion(output, target)
 
            running_loss += loss.item()
    running_loss = running_loss / len(data_loader_test)
    print('Epoch: ', epoch, '| test loss : %0.4f' % running_loss )
```

19. Save the model for use in inferencing tasks.

```
# save model
torch.save(model.state_dict(), "trained_inception_v3.pt")
```

Plotting the train and test loss shows both metrics reducing over training epochs. This is demonstrated in Figure 7.

| ![Figure 77](../../data/understand/deep_learning/inception v3.png) |
|:------------------------------------------------------------------:|
| Figure 7. Inception v3 Train and Loss Graph |

