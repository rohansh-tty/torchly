# tinyimagenet_dataset.py

from torch.utils.data import Dataset, random_split
from tqdm import tqdm_notebook
from tqdm import *

def create_dataset(dataset_path, id_path, splitRatio = 70, test_transforms = None, train_transforms = None):
  classes = extract_class_id(path = id_path)
  data = TinyImageNet(classes, dataset_path=dataset_path)
  
  dataset_len = len(data)

  train_data_len = dataset_len*splitRatio//100 # 110K * 0.7 = 77K
  test_data_len = dataset_len - train_data_len # 110K - 77K = 33K
  
  train, validation = random_split(data, [train_data_len, test_data_len]) # split the data according to split ratio
  train_dataset = TransformData(train, transform=train_transforms) # Data ready for Loading, passed onto Dataloader func
  test_dataset = TransformData(validation, transform=test_transforms)

  return train_dataset, test_dataset, classes



class TinyImageNet(Dataset):
    def __init__(self, classes, dataset_path):
        
        self.classes = classes
        self.dataset_path = dataset_path
        self.data = []
        self.target = []
        
        
        wnids = open(f"{dataset_path}/wnids.txt", "r")

        # Train Data
        train_data_path = dataset_path+"/train/"
        for cls in notebook.tqdm(wnids, total = 200):
          cls = cls.strip() # strip spaces out of class names

          train_images_path = train_data_path + cls + "/images/"
          
          for i in os.listdir(train_images_path): # this will list nXXXXXXXX Folders containing 500 Images.
            img = Image.open(train_images_path + i)
            npimage = np.asarray(img)
                
            if(len(npimage.shape) == 2): 
              npimage = np.repeat(npimage[:, :, np.newaxis], 3, axis=2) # add a new dim using np.newaxis, if it's a 2D
                
            self.data.append(npimage)  # appending image to data 
            self.target.append(self.classes.index(cls)) # appending corresponding class using self.classes


        # Validation Data
        valdata = open(f"{dataset_path}/val/val_annotations.txt", "r")
        for i in notebook.tqdm(valdata, total =10000):
          img, cls = i.strip().split("\t")[:2] # this will return image name and class ID. Ex: 'val_1.JPEG', 'n04067472'
          img = Image.open(f"{dataset_path}/val/images/{img}")
          npimage = np.asarray(img)
          
          if(len(npimage.shape) == 2):  
                npimage = np.repeat(npimage[:, :, np.newaxis], 3, axis=2) # add a new dim using np.newaxis, if it's a 2D
          
          self.data.append(npimage)  
          self.target.append(self.classes.index(cls))


    def __len__(self):
      """
      returns len of the dataset
      """
      return len(self.data)


    def __getitem__(self, idx):
      image_ = self.data[idx]
      target_ = self.target[idx]

      
      return image_, target_ 

class TransformData(Dataset):
    """
    Helper Class for transforming the images using albumentations.
    """
    def __init__(self, data, transform=None):
        """
        data: Train or Validation Dataset
        transform : List of Transforms that one wants to apply
        """
        self.data = data
        self.transform = transform


    def __getitem__(self, index):
        x, y = self.data[index]
        if self.transform:
            x = self.transform(image=x)
        return x, y

    def __len__(self):
        return len(self.data)




def extract_class_id(path):
    """
    Helps in extracting class ID from wnids file
    """
    IDFile = open(path, "r")
    classes = []

    for line in IDFile:
        classes.append(line.strip())
    return classes


def extract_class_name(path, word_id):
    """
    Helps in extracting class_names for that particular ID from words.txt file
    """
    word_file = open(path, "r")
    class_names = {}
    word_id = extract_class_id(path)

    for line in word_file:
        word_class = line.strip("\n").split("\t")[0] # word_class indicates the nXXXXXXX ID
        if word_class in word_id: 
            class_names[word_class] = line.strip("\n").split("\t")[1]  # Adding ClassName of a particular ID(key) as a value 
    return class_names
