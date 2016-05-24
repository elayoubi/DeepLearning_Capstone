from __future__ import print_function
import scipy.io as sio
from shutil import copyfile
import os
import urllib
import tarfile

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

        if i % 1000 == 0 :
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
