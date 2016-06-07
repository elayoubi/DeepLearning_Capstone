from __future__ import print_function
import scipy.io as sio
from shutil import copyfile
import os
import urllib
import tarfile
import numpy as np
from collections import defaultdict
from collections import namedtuple
from PIL import Image
import matplotlib.image as mpimg
import random


BoundingBox = namedtuple('BoundingBox', ['left', 'top', 'width', 'height'])
Digit = namedtuple('Digit', ['label', 'bounding_box'])
Image_T = namedtuple('Image_T', ['pixels', 'digits'])


def download_and_extract_SVHN(data_kind, outpath, mat_file ):
    '''Download and extract SVHN from its URL. data_kind should be either "test" or "train" 
    Function then produces a directory structure so each image has a text label file describing it'''

    print ("Downloading %s" %data_kind)    
    fname=None
    if data_kind == 'test':
        urllib.urlretrieve("http://ufldl.stanford.edu/housenumbers/test.tar.gz", filename="%s/test.tar.gz" % outpath)
        fname = "%s/test.tar.gz" %outpath
    elif data_kind == 'train':
        urllib.urlretrieve("http://ufldl.stanford.edu/housenumbers/train.tar.gz", filename="%s/train.tar.gz" % outpath)
        fname = "%s/train.tar.gz" %outpath
    else:
        raise ValueError ('Unexpected data_kind=%s' % data_kind)
        
    print ("Finished downloading %s" %data_kind)    

    print ("Extracting %s" %data_kind)    
    with tarfile.open(fname, "r:gz") as f:
        f.extractall("%s/images" % outpath)
    print ("Finished extracting %s" %data_kind)    

    if not os.path.exists("%s/labels/%s" % (outpath, data_kind)):
        os.makedirs("%s/labels/%s" % (outpath, data_kind))
    
    mat_contents = sio.loadmat('%s/digitStruct_v7.mat' % mat_file)

    fname=mat_contents['digitStruct'][0][1][0][0]

    i=0
    count=0

    max_x = 0
    max_y = 0

    for image in mat_contents['digitStruct'][0][:]:

        image_f_name = image[0][0]

        if i % 10000 == 0 :
            print ("Starting image %s " % image_f_name)
        
        count=count+1
        
        raw_label = open ( "%s/labels/%s/%s.label"%(outpath, data_kind, image_f_name.split('.')[0]) , 'w') 
    
        for digitInfo in image[1][0]:
                        
            #get the bounding box and label info
            height=digitInfo[0][0][0]
            left=digitInfo[1][0][0]
            top=digitInfo[2][0][0]
            width=digitInfo[3][0][0]
            label=digitInfo[4][0][0]
        
            print('%d,%d,%d,%d,%d' % (height, left, top, width, label), file=raw_label)
        
        raw_label.close()
            
        i=i+1

    print ("Completed %d images" % count)


def get_list_of_filenames(path, data_kind, batch_size, max_sequence_length=3) :
    '''Get list of numbers with max_sequence_length or less digits'''
    count = 0
    images = np.array([f for f in os.listdir("%s/images/%s" % (path, data_kind)) if 'png' in f])

    if len(images) < batch_size:
        batch_size=len(images)
    result = np.empty(batch_size, dtype=object)
    
    i=0
    while count < batch_size and i < len(images):
        label = images[i].replace('png', 'label')
        num_digits=0
        with open('%s/labels/%s/%s' % (path, data_kind, label)) as f:
            for line in f:
                num_digits += 1
                      
        if num_digits <= max_sequence_length:
            result[count] = images[i]
            count=count+1

        i=i+1    
        
    return result    


def count_images_by_length(image_path, data_kind):
    count = 0
    images = np.array([f  for f in os.listdir("%s/images/%s" % (image_path, data_kind)) if 'png' in f])
    counts = defaultdict(int)
    i=0
    while i < len(images):
        label = images[i].replace('png', 'label')
        num_digits=0
        with open('%s/labels/%s/%s' % (image_path, data_kind, label)) as f:
            for line in f:
                num_digits += 1

        counts[num_digits] += 1

        i=i+1
        
    return counts

def get_outer_bounding_box(png_file_name, label_file_name):
    '''Returns the dimensions of the bounding box of all the digits in the image.'''

    #img = mpimg.imread(png_file_name)
    #plt.imshow(img)
    #ca = plt.gca()

    #bounding coordinates for all digits
    outer_top = 0
    outer_left = 0
    outer_bottom = 0
    outer_right = 0

    with open(label_file_name, 'rb') as f:
        i=-1
        for line in f:
            i=i+1
            height, left, top, width, label = line.split(',')
            h = int(height)
            l = int(left)
            w = int(width)
            t = int(top)

            bottom = t+h
            right = l + w

            label=int(label)

            if i==0:
                outer_top = t
                outer_left = l
                outer_bottom = bottom
                outer_right = right

            else:
                if outer_top > t:
                    outer_top = t
                if outer_left > l : 
                    outer_left = l
                if outer_bottom < bottom :
                    outer_bottom = bottom
                if outer_right < right :
                    outer_right = right


            #ca.add_patch(Rectangle((l, t), w, h, color='blue', fill=False))

        #ca.add_patch(Rectangle((outer_left, outer_top), outer_right - outer_left, outer_bottom - outer_top, color = 'green', fill = False))
        return BoundingBox(left=outer_left, top=outer_top, width=outer_right-outer_left, height=outer_bottom-outer_top)
    

def read_SVHN_label(line):
    height, left, top, width, label = line.split(',')
    h = int(height)
    l = int(left)
    w = int(width)
    t = int(top)
    label=int(label)
    
    if label == 10:
        label = 0
    return l, t, w, h, label


def translate_image_scale_bb(png_file_name, label_file_name, output_dimension):
    '''Slides the image maintaining the numbers in the frame. The resulting bounding box is scaled to match the transformation.'''

    outer_box = get_outer_bounding_box(png_file_name, label_file_name)
        
    h = outer_box.height
    l = outer_box.left
    w = outer_box.width
    t = outer_box.top
    
    dimension = max( int(1.3 * max(w, h)), output_dimension)
    new_width = w / 0.6
    new_height = h / 0.6
    
    tmpfile='.tmp_%s.png' % os.getpid()
  
    pil_img = Image.open("%s"%(png_file_name))
    #pil_img.convert('L').save(tmpfile) #convert to grayscale so there are less features
    
    
    cropped_left = 0
    if l > 0:
        cropped_left = random.randint( max(l+w - int(new_width), 0), l)
    cropped_top = 0
    if t > 0:
        cropped_top = random.randint( max(t+h- int(new_height), 0) , t)

    new_bb_top =  t - cropped_top 
    new_bb_left = l - cropped_left

    #img=mpimg.imread(png_file_name)
    #plt.figure()
    #plt.imshow(img)
    #ca=plt.gca()
    with open(label_file_name, 'rb') as f:
        for line in f:
            d_l, d_t, d_w, d_h, d_label = read_SVHN_label(line)
            
            #ca.add_patch(Rectangle((d_l ,d_t) , d_w, d_h, color='blue', fill=False))
            
    pil_img.crop((cropped_left, cropped_top, int(cropped_left+new_width), int(cropped_top +new_height))).save(tmpfile)
      
    
    cropped_img = Image.open(tmpfile)
    cropped_img.resize((output_dimension, output_dimension), Image.ANTIALIAS).save(tmpfile)
            
    img = mpimg.imread(tmpfile)
    os.remove(tmpfile)
    X =np.array(img, dtype=np.float32)

    
    new_bb = []
    label = []
    #plt.figure()
    #plt.imshow(X)
    #ca=plt.gca()
    with open(label_file_name, 'rb') as f:
        for line in f:
            d_l, d_t, d_w, d_h, d_label = read_SVHN_label(line)
            scaled_left = int((d_l - cropped_left) * output_dimension / new_width)
            scaled_top = int((d_t - cropped_top) * output_dimension / new_height)
            scaled_height = int(d_h * output_dimension / new_height)
            scaled_width = int(d_w * output_dimension / new_width)
            
            #ca.add_patch(Rectangle((scaled_left ,scaled_top) , scaled_width, scaled_height, color='blue', fill=False))
            
            new_bb.append(BoundingBox(left=scaled_left, top=scaled_top, width=scaled_width, height=scaled_height))
            label.append(d_label)
    return X, label, new_bb

def load_svhn_cropped_and_slided(file_name_array, path, dimension=64, n_channels=3,  kind='train'):
    '''Augments the available data by cropping and sliding the image and producing a square image of size dimenxion x dimension'''
    
    number_of_files = len(file_name_array)        
        
    images = np.empty([number_of_files], dtype=Image_T)
    
    i=0
    for png_file in file_name_array:
        label_file = png_file.replace('.png', '.label')
        
        
        pixels, label_array, bb_array = translate_image_scale_bb("%s/images/%s/%s"%(path,kind, png_file), '%s/labels/%s/%s' %(path ,kind, label_file), dimension)
        pixels = (pixels - pixels.mean())
        pixels = pixels.reshape([dimension*dimension*n_channels])
        
        digitsArray =[]
        
        j=0     
        for digit in label_array:
                        
            digitsArray.append(Digit(label=digit, bounding_box=bb_array[j]))
            
            j=j+1
            
        images[i] = Image_T (pixels=pixels, digits=digitsArray)
        
        i=i+1
    
    return images


def extract_vectors(images, number_of_classes=10, input_dimension=64, n_channels=3, max_digit_sequence=3):

    '''Given an array of Image_T named tuples, returns an X, y, and bounding box set of matrices for each image'''

    num_images = len(images)
    
    X = np.empty([num_images, input_dimension * input_dimension * n_channels], dtype=np.float32)
    labels = np.empty([num_images, max_digit_sequence], dtype=np.uint8)
    labels.fill(number_of_classes)
    bb = np.zeros([num_images, max_digit_sequence], dtype=object)
        
    i=0
    for img in images:
        X[i] = img.pixels
        j=0
        for digit in img.digits:
            labels[i][j]  =digit.label          
            bb[i][j]  =digit.bounding_box
            
            j=j+1
    
        i=i+1

    return X, labels, bb

def dense_to_one_hot(labels_dense, num_classes, t=np.uint8):
    """Convert class labels from scalars to one-hot vectors."""
    num_labels = labels_dense.shape[0]
    index_offset = np.arange(num_labels) * num_classes
    labels_one_hot = np.zeros((num_labels, num_classes), dtype=t)
    labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
    return labels_one_hot

def load_batch(list_of_filenames, inpath, data_kind, number_of_classes, image_dimension, n_channels):
    '''Reads a batch of image files from disk. This is done to avoid reading the entire data set in memory, to conserve memory'''
    
    images = load_svhn_cropped_and_slided(list_of_filenames, inpath, image_dimension, n_channels, data_kind)
    X, labels, bb_array = extract_vectors(images, number_of_classes, input_dimension=image_dimension, n_channels=n_channels)

    y1 = dense_to_one_hot(labels[:,0], number_of_classes , np.float32)
    y2 = dense_to_one_hot(labels[:,1], number_of_classes + 1 , np.float32)
    y3 = dense_to_one_hot(labels[:,2], number_of_classes + 1, np.float32)

        
    return X, y1,y2,y3
