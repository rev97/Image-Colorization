import torch
import random
import os
import csv
import sys
import numpy as np
import glob
import matplotlib.pyplot as plt
import pandas as pd
import torchvision.transforms as transforms
from skimage import io, transform, color
from torch.utils.data import Dataset, DataLoader
from PIL import Image
# Ignore warnings
import warnings
warnings.filterwarnings("ignore")
#setting the default Torch datatype to 32-bit float
torch.set_default_tensor_type(torch.FloatTensor)
import cv2

# For conversion
from skimage.color import lab2rgb, rgb2lab, rgb2gray
from skimage import io
# For everything
import torch
import torch.nn as nn
import torch.nn.functional as F
# For our model
import torchvision.models as models
from torchvision import datasets, transforms
# For utilities
import os, shutil, time

use_gpu = torch.cuda.is_available()

#Loading the Dataset

def load_dataset(path_to_input_folder):

    ##print(cv2.__version__)
    #img_dir contains the file path to image files
    img_dir = "/Users/revanthgottuparthy/Desktop/DL/project2/face_images/"
    img_dir = path_to_input_folder
    #reading all the images from the input folder to list named data using cv2 functions
    data = []
    for images in glob.iglob(f'{img_dir}/*'):
        if (images.endswith(".jpg")):
            img = cv2.imread(images,cv2.IMREAD_COLOR)
            cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            data.append(img)

    # converting data list to numpy array and then converting the array to a tensor
    tensor = torch.from_numpy(np.array(data))
    #shuffling the data in tensor that is defined above by shuffling the indexes and rendexing 
    sampleIndexes = torch.randperm(len(tensor)) 
    tensor = tensor[sampleIndexes]

#method to create annotations file for image lables for the dataset
def create_annotations_file_train(input_path,file_name):
    # Get the list of all files and directories
    #path = "/Users/revanthgottuparthy/Desktop/DL/project2/face_images/"
    dir_list = os.listdir(input_path)
    res =[]
    for i in dir_list:
        if i.endswith(('.jpg', '.png', 'jpeg')):
            res.append([i])
    file = open(os.path.join(input_path,file_name) , 'w+', newline ='')
    
    # writing the data into the file
    with file:   
        write = csv.writer(file)
        write.writerows(res[0:675])

#method to create annotations file for image lables for the dataset
def create_annotations_file_test(input_path,file_name):
    # Get the list of all files and directories
    #path = "/Users/revanthgottuparthy/Desktop/DL/project2/face_images/"
    dir_list = os.listdir(input_path)
    res =[]
    for i in dir_list:
        if i.endswith(('.jpg', '.png', 'jpeg')):
            res.append([i])
    file = open(os.path.join(input_path,file_name) , 'w+', newline ='')
    
    # writing the data into the file
    with file:   
        write = csv.writer(file)
        write.writerows(res[675:750])

#Augmenting the dataset

class Rescale(object):
    """Rescale the image in a sample to a given size.

    Args:
        output_size (tuple or tuple): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        image = sample['image']

        h, w = image.shape[:2]
        #print(h,w)
        if isinstance(self.output_size, int):
            new_h, new_w = self.output_size * h / w, self.output_size
            #new_h, w = self.output_size * h , w
    

        new_h, new_w = int(new_h), int(new_w)

        #transform = T.Resize((new_h, new_w))
        #resized_img = transform(image)
        img = transform.resize(image, (new_h, new_w))

        return {'image': img}

class RandomCrop(object):
    """Crop randomly the image in a sample.

    Args:
        output_size (tuple or int): Desired output size. If int, square crop
            is made.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, sample):
        image = sample['image']

        h, w = image.shape[:2]
        new_h, new_w = self.output_size

        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)

        image = image[top: top + new_h,
                      left: left + new_w]


        return {'image': image}

#LabFaceDataset which we have defined in the previous tasks will be used to give training and validation data for CNN defined
class LabFaceDataset(Dataset):
    """Georgia Tech Face Database."""
    def __init__(self, annotations_file, img_dir,flag = None,scale=None,crop=None,rgb_scale=None,transform=None):
        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.flag = flag
        self.scale = scale
        self.crop = crop
        self.rgb_scale = rgb_scale
        self.transform = transform
        

    def __len__(self):
        return len(self.img_labels)

    def HorizontalFlip(self,sample):
        image = sample['image']
        flippled_image=np.flipud(image)
        return {'image': flippled_image}

    def randomCrop(self,sample,output_size):
        image = sample['image']

        h, w = image.shape[:2]
        new_h, new_w = output_size, output_size

        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)

        image = image[top: top + new_h,
                      left: left + new_w]


        return {'image': image}

    def reScale(self,sample,output_size):
        image = sample['image']

        h, w = image.shape[:2]
        #print(h,w)
        if isinstance(output_size, int):
            new_h, new_w = output_size * h / w, output_size
    

        new_h, new_w = int(new_h), int(new_w)

        img = transform.resize(image, (new_h, new_w))

        return {'image': img}

    def RGBScaling(self,sample):
        image = sample['image']
        h, w = image.shape[:2]
        for i in range(h):
            for j in range(w):
                image[i, j] = image[i, j] * self.rgb_scale



        return {'image': image}


    def ToTensor(self, sample):
        image = sample['image']

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        image = image.transpose((2, 0, 1))
        result = np.where(image<0, 0, image)
        return {'image': torch.from_numpy(result)}

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx,0])
        img = Image.open(img_path)
        #img = io.imread(img_path)
        #img = cv2.imread(img_path,cv2.IMREAD_COLOR)
        # img is the original pic
        # lab is the original pic converted to lab
        img_original = np.asarray(img)
        lab = rgb2lab(img_original) 
        #cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        #lab = color.rgb2lab(img)
        sample = {'image': lab}
        nt_lab = (lab + 128) / 255
        nt_img_ab = nt_lab[:, :, 1:3] #else lab
        nt_img_ab = torch.from_numpy(nt_img_ab.transpose((2, 0, 1))).float()
        nt_img_original = rgb2gray(img_original)
        nt_img_original = torch.from_numpy(nt_img_original).unsqueeze(0).float()

        if self.transform:
            flipped_sample = self.HorizontalFlip(sample)
            cropped_sample = self.randomCrop(flipped_sample,100)
            scaled_sample = self.reScale(cropped_sample,256)
            img_ab = scaled_sample['image'][:, :, 1:3]
            img_ab = torch.from_numpy(img_ab.transpose((2, 0, 1))).float()
            img_original = rgb2gray(img_original)
            img_original = torch.from_numpy(img_original).unsqueeze(0).float()


                
            if self.flag == "Totensor":
                pass
                #return self.ToTensor(flipped_sample)
            else:
                return nt_img_original, nt_img_ab
        else:

            if self.flag == "Totensor":
                return self.ToTensor(sample)
            else:
                pass
                return nt_img_original, nt_img_ab

#Creating Dataloader for training data
def get_train_dataloader(file_path,img_dir):

    transformed_labdataset_tensor = LabFaceDataset(annotations_file=file_path,
                                            img_dir=img_dir,
                                            transform=True)
    #batch_size to the data can be modified here by adjusting the argument batch_size(Here 4)
    dataloader = DataLoader(transformed_labdataset_tensor, batch_size=4,
                            shuffle=True, num_workers=4)

    
    return dataloader

#Creating Dataloader for validation data
def get_val_dataloader(file_path,img_dir):
    transformed_labdataset_tensor = LabFaceDataset(annotations_file=file_path,
                                            img_dir=img_dir,)                                        
    #batch_size to the data can be modified here by adjusting the argument batch_size(Here 4)
    dataloader = DataLoader(transformed_labdataset_tensor, batch_size=4,
                            shuffle=True, num_workers=4)

    
    return dataloader

#methods of class AverageMeter is used to create the mean values for a and b channels
class AverageMeter(object):
  def __init__(self):
    self.reset()
  def reset(self):
    self.val, self.avg, self.sum, self.count = 0, 0, 0, 0
  def update(self, val, n=1):
    self.val = val
    self.sum += val * n
    self.count += n
    self.avg = self.sum / self.count

#colorize method takes L channel from input image and concatenates a,b channels from CNN output inorder to colorize the image
def colorize(L_input, ab_input, path=None, name=None):
  plt.clf() 
  color_image = torch.cat((L_input, ab_input), 0).numpy()
  color_image = color_image.transpose((1, 2, 0))  
  color_image[:, :, 0:1],color_image[:, :, 1:3] = color_image[:, :, 0:1] * 100,color_image[:, :, 1:3] * 255 - 128  
  color_image = lab2rgb(color_image.astype(np.float64))
  L_input = L_input.squeeze().numpy()
  if path is not None and name is not None: 
    plt.imsave(arr=L_input, fname='{}{}'.format(path['gray'], name), cmap='gray')
    plt.imsave(arr=color_image, fname='{}{}'.format(path['colorized'], name))


#ColorizationNet is the CNN that we are going to use to predict the chrominance value 
#for the first layer we are going to use resnet18 model to take single L channel as an input 
class ColorizationNet(nn.Module):

  def __init__(self, input_size=128):
    super(ColorizationNet, self).__init__()
    MIDLEVEL_FEATURE_SIZE = 128

    ## First half: ResNet
    resnet = models.resnet18(num_classes=365) 
    # Changing first convolution layer to accept single L channel as an input
    resnet.conv1.weight = nn.Parameter(resnet.conv1.weight.sum(dim=1).unsqueeze(1)) 
    # Extracting midlevel features from ResNet18 architecture
    self.first_layer = nn.Sequential(*list(resnet.children())[0:6])

    ## Upsampling and Batch normalization is implemented here
    self.second_layer = nn.Sequential(     
      nn.Conv2d(MIDLEVEL_FEATURE_SIZE, 128, kernel_size=3, stride=1, padding=1),
      nn.BatchNorm2d(128),
      nn.ReLU(),
      nn.Upsample(scale_factor=2),
      nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1),
      nn.BatchNorm2d(64),
      nn.ReLU(),
      nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
      nn.BatchNorm2d(64),
      nn.ReLU(),
      nn.Upsample(scale_factor=2),
      nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1),
      nn.BatchNorm2d(32),
      nn.ReLU(),
      nn.Conv2d(32, 2, kernel_size=3, stride=1, padding=1),
      nn.Upsample(scale_factor=2)
    )
  def forward(self, input):

    # Pass input through ResNet-gray to extract features
    out = self.first_layer(input)

    # Upsample to get colors
    output = self.second_layer(out)
    return output

# Even created this model but not going to use this architecture since it is not giving best results in the validation
class ConvNeuralNet(nn.Module):
	#  Determine what layers and their order in CNN object 
    def __init__(self):
        super(ConvNeuralNet, self).__init__()
        
        self.conv_layer1 = nn.Conv2d(in_channels=1, out_channels=2, kernel_size = 3,stride=1, padding=1)
        self.relu1 = nn.ReLU()
        self.conv_layer2 = nn.Conv2d(in_channels=2, out_channels=4, kernel_size = 3,stride=1, padding=1)
        self.relu2 = nn.ReLU()
        self.conv_layer3 = nn.Conv2d(in_channels=4, out_channels=8, kernel_size = 3,stride=1, padding=1)
        self.relu3 = nn.ReLU()
        self.conv_layer4 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size = 3,stride=1, padding=1)
        self.relu4 = nn.ReLU()
        self.conv_layer5 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size = 3,stride=1, padding=1)
        self.relu5 = nn.ReLU()
        self.conv_layer6 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size = 3,stride=1, padding=1)
        self.relu6 = nn.ReLU()
        self.conv_layer7 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size = 3,stride=1, padding=1)
        self.relu7 = nn.ReLU()
        #reducing the channel size
        self.conv_layer8 = nn.Conv2d(in_channels=128, out_channels=2, kernel_size = 3,stride=1, padding=1)
        self.relu7 = nn.ReLU()
    # Progresses data across layers    
    def forward(self, x):
        out = self.conv_layer1(x)
        out = self.conv_layer2(out)
        out = self.conv_layer3(out)
        out = self.conv_layer4(out)
        out = self.conv_layer5(out)
        out = self.conv_layer6(out)
        out = self.conv_layer7(out)
        out = self.conv_layer8(out)
        return out

def train(train_loader, model, criterion, optimizer, epoch):
  print('Starting training for epoch {}'.format(epoch))
  model.train()
  
  # Prepare value counters and timers
  batch_time, data_time, losses = AverageMeter(), AverageMeter(), AverageMeter()

  end = time.time()
  for i, (input_gray, input_ab) in enumerate(train_loader):

    
    # Use GPU if available
    '''
    if use_gpu: 
        input_gray, input_ab = input_gray.cuda(), input_ab.cuda()
    '''
    # Record time to load data (above)
    data_time.update(time.time() - end)

    # Run forward pass
    output_ab = model(input_gray)
    loss = criterion(output_ab, input_ab) 
    losses.update(loss.item(), input_gray.size(0))

    # Compute gradient and optimize
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Record time to do forward and backward passes
    batch_time.update(time.time() - end)
    end = time.time()

    # Print model accuracy -- in the code below, val refers to value, not validation
    if i % 25 == 0:
      print('Epoch: [{0}][{1}/{2}]\t'
            'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
            'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
            'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(
              epoch, i, len(train_loader), batch_time=batch_time,
             data_time=data_time, loss=losses))
  print('Finished training for epoch {}'.format(epoch))

def validate(val_loader, model, criterion, save_images, epoch):
  model.eval()

  # Prepare value counters and timers
  batch_time, data_time, losses = AverageMeter(), AverageMeter(), AverageMeter()

  end = time.time()
  already_saved_images = False
  for i, (input_gray, input_ab) in enumerate(val_loader):
    data_time.update(time.time() - end)

    # Use GPU
    '''
    if use_gpu: 
        input_gray, input_ab = input_gray.cuda(), input_ab.cuda()
    '''

    # Run model and record loss
    output_ab = model(input_gray) # throw away class predictions
    loss = criterion(output_ab, input_ab)
    losses.update(loss.item(), input_gray.size(0))

    # Save images to file
    if save_images and not already_saved_images:
      already_saved_images = True
      for j in range(min(len(output_ab), 10)): # save at most 5 images
        save_path = {'gray': 'outputs/gray/', 'colorized': 'outputs/color/'}
        save_name = 'img-{}-epoch-{}.jpg'.format(i * val_loader.batch_size + j, epoch)
        colorize(input_gray[j].cpu(), ab_input=output_ab[j].detach().cpu(), path=save_path, name=save_name)

    # Record time to do forward passes and save images
    batch_time.update(time.time() - end)
    end = time.time()

    # Print model accuracy -- in the code below, val refers to both value and validation
    if i % 25 == 0:
      print('Validate: [{0}/{1}]\t'
            'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
            'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(
             i, len(val_loader), batch_time=batch_time, loss=losses))

  print('Finished validation.')
  return losses.avg


def run_training(train_loader,val_loader):
    #driver method which runs the CNN model and predicts the mean value of ab values
    '''
    if use_gpu: 
        criterion = criterion.cuda()
        model = model.cuda()
    '''
    #model = ConvNeuralNet()
    model = ColorizationNet()
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2, weight_decay=0.0)

    # Creating the folders for output
    os.makedirs('outputs/color', exist_ok=True)
    os.makedirs('outputs/gray', exist_ok=True)
    os.makedirs('checkpoints', exist_ok=True)

    best_losses = 1e10
    save_images = True
    #Defining number of epochs for training the CNN
    epochs = 10

    # Train model
    for epoch in range(epochs):
    # Training and validationg for each epoch
        train(train_loader, model, criterion, optimizer, epoch)
        with torch.no_grad():
            losses = validate(val_loader, model, criterion, save_images, epoch)
    # Saving checkpoint and replacing old best model if current model is better
    if losses < best_losses:
        best_losses = losses
        torch.save(model.state_dict(), 'checkpoints/model-epoch-{}-losses-{:.3f}.pth'.format(epoch+1,losses))



    
if __name__ == "__main__":
    #reading the system arguments
    #Arg1: Path to input image directory
    #Arg2: Name for Annotations file(.csv format)
    img_dir = sys.argv[1]
    train_file = sys.argv[2]
    test_file = sys.argv[3]
    annotate_file_path_train = os.path.join(img_dir,train_file)
    annotate_file_path_test = os.path.join(img_dir,test_file)
    #loading the facedataset 
    load_dataset(img_dir)
    create_annotations_file_train(img_dir,sys.argv[2])
    create_annotations_file_test(img_dir,sys.argv[3])
    #using the dataloader to get training and validation data 
    train_dataloader = get_train_dataloader(annotate_file_path_train, img_dir)
    val_dataloader = get_val_dataloader(annotate_file_path_test, img_dir)
    #driver method which runs the CNN model, captures the losses and outputs the colorization outputs
    run_training(train_dataloader,val_dataloader)