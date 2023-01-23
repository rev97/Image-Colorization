#Importing all the required packages
import torch
import os
import csv
import sys
import numpy as np
import glob
import matplotlib.pyplot as plt
import pandas as pd
import cv2
import torchvision.transforms as transforms
from torchvision.io import read_image
from torch.utils.data import Dataset, DataLoader
from torchvision import utils
from skimage import io, transform, color
# Ignore warnings
import warnings
warnings.filterwarnings("ignore")
#setting the default Torch datatype to 32-bit float
torch.set_default_tensor_type(torch.FloatTensor)



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
    print(tensor[0])


#Creating the annotations file

def create_annotations_file(input_path,file_name):
    # Get the list of all files and directories
    dir_list = os.listdir(input_path)

    res =[]
    for i in dir_list:
        if i.endswith(('.jpg', '.png', 'jpeg')):
            res.append([i])

    # opening the csv file in 'w+' mode
    file = open(os.path.join(input_path,file_name) , 'w+', newline ='')
    
    # writing the data into the file
    with file:   
        write = csv.writer(file)
        write.writerows(res)


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
        img = transform.resize(image, (new_h, new_w))

        return {'image': img}

#Crating the class for Random Crop for the images
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

#Augmenting the dataset


class FaceDataset(Dataset):
    """Georgia Tech Face Database."""
    def __init__(self, annotations_file, img_dir, flag=None,scale=None,crop=None,rgb_scale=None,transform=None):
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
        image = io.imread(img_path)
        sample = {'image': image}
        if self.transform:
            for i, tsfrm in enumerate([self.scale, self.crop, self.transform]):
                transformed_sample = tsfrm(sample)
                flipped_sample = self.HorizontalFlip(transformed_sample)
                rgbscaled_sample = self.RGBScaling(flipped_sample)
                
            if self.flag == "Totensor":
                return self.ToTensor(rgbscaled_sample)
            else:
                return rgbscaled_sample
            
        else:

            if self.flag == "Totensor":
                return self.ToTensor(sample)
            else:
                return sample
        
#creating a class which converts RGB channels to LAB channels
class LabFaceDataset(Dataset):
    """Georgia Tech Face Database."""
    def __init__(self, annotations_file, img_dir, flag = None,scale=None,crop=None,rgb_scale=None,transform=None):
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
        
    def RGBScaling(self,sample):
        image = sample['image']
        h, w = image.shape[:2]
        for i in range(h):
            for j in range(w):
                image[i, j] = image[i, j] * self.rgb_scale



        return {'image': image}

    def reScale(self,sample,output_size):
        image = sample['image']

        h, w = image.shape[:2]
        if isinstance(output_size, int):
            new_h, w = output_size * h , w


        new_h, new_w = int(new_h), int(new_w)

        #transform = T.Resize((new_h, new_w))
        #resized_img = transform(image)
        img = transform.resize(image, (new_h, new_w))

        return {'image': img}


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
        #image = io.imread(img_path)
        img = cv2.imread(img_path,cv2.IMREAD_COLOR)
        cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        lab = color.rgb2lab(img)
        sample = {'image': lab}
        if self.transform:
            for i, tsfrm in enumerate([self.scale, self.crop, self.transform]):
                transformed_sample = tsfrm(sample)
                flipped_sample = self.HorizontalFlip(transformed_sample)
                
            if self.flag == "Totensor":
                return self.ToTensor(flipped_sample)
            else:
                return flipped_sample,img
        else:

            if self.flag == "Totensor":
                return self.ToTensor(sample)
            else:
                return sample
        

def show_images(image):
    plt.imshow(image)


def show_cv2images(image):
    cv2.imshow('image', image)
    cv2.waitKey(20)
    cv2.destroyAllWindows()

#Function to save the converted RGB To LAB images
def save_labimages(path,labimage,img):
    #image = image.view(image.shape[1], image.shape[2], image.shape[0])
    
    cv2.imshow('image', img)
    #lab = color.rgb2lab(labimage)
    L,a,b=cv2.split(labimage)
    #cv2.imshow('L',L) 
    #cv2.imshow('a',a) 
    #cv2.imshow('b',b)
    cv2.imwrite(os.path.join(path , 'Image.jpg'), img)
    cv2.imwrite(os.path.join(path , 'L.jpg'), L)
    cv2.imwrite(os.path.join(path , 'a.jpg'), a)
    cv2.imwrite(os.path.join(path , 'b.jpg'), b)
    
#Function to test the facedataset for one sample
def test_facedataset_foronesample(file_path,img_dir):
    face_dataset = FaceDataset(annotations_file=file_path,
                                    img_dir=img_dir)

    for i in range(len(face_dataset)):
        sample = face_dataset[i]
        print(sample)
        break
    plt.figure()
    show_images(sample['image'])
    plt.show()


#Function to test the Augmented facedataset for multiple samples
def test_augmented_facedataset(file_path,img_dir):
    import random
    scale = Rescale(256)
    crop = RandomCrop(100)
    transformed_dataset = FaceDataset(annotations_file=file_path,
                                            img_dir=img_dir,
                                                scale = Rescale(256),
                                                crop = RandomCrop(100),
                                                rgb_scale = random.uniform(0.6, 1.0),
                                            transform=transforms.Compose([
                                                scale,
                                                crop,
                                            ]))
    for i in range(len(transformed_dataset)):
        sample = transformed_dataset[i]

        print(i, sample['image'].shape)

        ax = plt.subplot(1, 4, i + 1)
        plt.tight_layout()
        ax.set_title('Sample #{}'.format(i))
        ax.axis('off')
        show_images(**sample)
        #Testing for 3 samples
        if i == 3:
            plt.show()
            break

#Function to test the Augmented LAB facedataset for multiple samples
def test_augmented_labfacedataset(file_path,img_dir):
    import random
    scale = Rescale(256)
    crop = RandomCrop(100)
    lab_dataset = LabFaceDataset(annotations_file=file_path,
                                            img_dir=img_dir,
                                                scale = Rescale(256),
                                                crop = RandomCrop(100),
                                                rgb_scale = random.uniform(0.6, 1.0),
                                            transform=transforms.Compose([
                                                scale,
                                                crop,
                                            ]))


    output_dir = os.path.join(img_dir,'output')
    os.mkdir(output_dir)

    for i in range(3):
        sample,img = lab_dataset[i]

        print(i, sample['image'].shape)
        sample_out_dir = os.path.join(output_dir,str(i))
        os.mkdir(sample_out_dir)
        save_labimages(sample_out_dir,sample['image'],img)

#Function that returns the tensors for each facedataset sample
def get_FaceDataset(file_path,img_dir):
    import random
    scale = Rescale(256)
    crop = RandomCrop(100)
    transformed_dataset_tensor = FaceDataset(annotations_file=file_path,
                                            img_dir=img_dir,
                                            flag = "Totensor",
                                            scale = Rescale(256),
                                            crop = RandomCrop(100),
                                            rgb_scale = random.uniform(0.6, 1.0),
                                            transform=transforms.Compose([
                                            scale,
                                            crop,
                                            ]))
    for i in range(len(transformed_dataset_tensor)):
        print("Getting the tensors for each facedataset sample")
        sample = transformed_dataset_tensor[i]

        print(i, type(sample),sample['image'].size())
        if i == 3:
            break

#Function that returns the tensors for each LAB facedataset sample
def get_LabFaceDataset(file_path,img_dir):
    import random
    scale = Rescale(256)
    crop = RandomCrop(100)
    transformed_labdataset_tensor = LabFaceDataset(annotations_file=file_path,
                                            img_dir=img_dir,
                                            flag = "Totensor",
                                            scale = Rescale(256),
                                            crop = RandomCrop(100),
                                            rgb_scale = random.uniform(0.6, 1.0),
                                            transform=transforms.Compose([
                                            scale,
                                            crop,
                                            ]))
    for i in range(len(transformed_labdataset_tensor)):
        
        print("Getting the tensors for each lab converted facedataset sample")
        sample = transformed_labdataset_tensor[i]
        
        print(i, type(sample),sample['image'].size())
        if i == 3:
            break
        '''
        f_sample,img = transformed_labdataset_tensor[i]
        print(f_sample)
        if i == 1:
            break
        '''
    
#creating the dataloader for LabFaceDataset
def implement_dataloader(file_path,img_dir):
    import random
    scale = Rescale(256)
    crop = RandomCrop(100)
    transformed_labdataset_tensor = LabFaceDataset(annotations_file=file_path,
                                            img_dir=img_dir,
                                            flag = "Totensor",
                                            scale = Rescale(256),
                                            crop = RandomCrop(100),
                                            rgb_scale = random.uniform(0.6, 1.0),
                                            transform=transforms.Compose([
                                            scale,
                                            crop,
                                            ]))

    dataloader = DataLoader(transformed_labdataset_tensor, batch_size=4,
                            shuffle=True, num_workers=4)




    for i_batch, sample_batched in enumerate(dataloader):
        print(i_batch, sample_batched['image'].size())

        # observe 4th batch and stop.
        if i_batch == 3:
            break

if __name__ == "__main__":
    #reading the system arguments
    #Arg1: Path to input image directory
    #Arg2: Name for Annotations file(.csv format)
    img_dir = sys.argv[1]
    annotate_file_path = os.path.join(img_dir,sys.argv[2])
    #load_dataset method creates the tensor of the required size by reading the images from a given folder
    load_dataset(img_dir)
    #create_annotations_file method is called to create a csv file which contains the names of all the images that are present in a directory
    create_annotations_file(img_dir,sys.argv[2])
    #testing the user defined dataset named "facedataset" for one sample of image 
    test_facedataset_foronesample(annotate_file_path,img_dir)
    #testing the augmented facedataset for sample of images
    test_augmented_facedataset(annotate_file_path,img_dir)
    #testing the augmented lab converted facedataset for sample of images
    test_augmented_labfacedataset(annotate_file_path,img_dir)
    #get_FaceDataset method returns the tensors of image data of required size
    get_FaceDataset(annotate_file_path,img_dir)
    #get_LabFaceDataset method returns the tensors of lab converted image data of required size
    get_LabFaceDataset(annotate_file_path,img_dir)
    #creating a dataloader to enumerate through all the images in a directory
    implement_dataloader(annotate_file_path,img_dir)


