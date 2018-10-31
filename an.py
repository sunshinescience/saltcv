import os 
import numpy as np
import scipy.misc
import matplotlib.pyplot as plt

def all_masks(filename):
    '''
    This takes a training set .csv file as an argument. The image names are within the first column of this .csv file. 
    The additional columns contain numbers. Those numbers correspond to the training set of run-length encoded masks.
    Each row in the .csv file is the image (in the first column) and the corresponding run-length encoded mask to that image (in the additional columns).
    This function produces run-length encoded masks of black and white images that match the training set of masks in the .csv file.
    This prints the number of masks that match or do not match the training set. This also prints the row (image_index) in which the 
    mask does not match the train mask.
    '''
    matches_true = []  # A list of the amount of True matches between our masks and the train masks
    matches_not_true = [] # A list of image indexes in which the train masks do not match our masks
    with open(filename) as f:
        lines_list = f.readlines()[1:]
    num_images = len(lines_list) 
    for image_index in range(num_images):
        image_name = lines_list[image_index].strip().split(',')[0]
        f_name_mask = os.path.join('/Users', 'sa4312', 'Dropbox', 'tgs','data_set', 'all', 'train', 'masks', image_name + '.png')
        pict_mask = scipy.misc.imread(f_name_mask, flatten=True)
        our_masks = make_mask(pict_mask)
        train_masks = train_matches(filename, image_index)
        matches = our_masks == train_masks
        if matches == True:
            matches_true.append(matches) # We put the number of matches in a list called matches_true
        elif matches == False:
            matches_not_true.append(image_index)
    print ('The number of our masks that match the train masks is: ', sum(matches_true))
    print ('The number of our masks that do not match the train masks is: ', (num_images - sum(matches_true)))
    print ('The image indexes in which our masks and the train masks do not match are: ', matches_not_true)

def pic_mask(filename, image_index):
    """
    This takes in a file and an image index. This returns an array of pixel values based on an image.
    """
    with open(filename) as f:
        lines_list = f.readlines()[1:]
    image_name = lines_list[image_index].strip().split(',')[0]
    f_name_mask = os.path.join('/Users', 'sa4312', 'Dropbox', 'tgs','data_set', 'all', 'train', 'masks', image_name + '.png')
    picture_mask = scipy.misc.imread(f_name_mask, flatten=True)
    return picture_mask

def make_mask(picture_mask):
    '''
    This takes in an argument, which is an array of pixel values based on an image, and returns a run-length encoded mask.
    '''
    # Flatten the array called 'picture_mask' in order to create a single row of the pixel values
    mask_flattened = np.transpose(picture_mask).flatten()
    mask_flattened_no_salt = np.all(mask_flattened[:] == 0) # The number in 'mask_flattened' is 0 if its a black pixel (i.e., no salt)
    mask_flattened_salt = np.all(mask_flattened[:] == 65535) # The number in 'mask_flattened' is 65535 if it is a white pixel (i.e., salt)
    if mask_flattened_no_salt == True:
        pixel_change_location = np.array([0]) # The array called 'pixel_change_location' is the pixel location of where the pixel changes (e.g., from black to white or vice versa) on the mask image
        Len_last_patch = len(mask_flattened)
    elif mask_flattened_salt == True:
        pixel_change_location = np.array([0])
        Len_last_patch = len(mask_flattened)
    else:
        pixel_change_location = np.where(np.roll(mask_flattened,1) != mask_flattened)[0] 
        Len_last_patch = len(mask_flattened) - pixel_change_location[-1]

    # The differences between consecutive elements (i.e., subtracted consecutive elements) of the array called 'pixel_change_location':
    patch_lengths = np.ediff1d(pixel_change_location, to_end=Len_last_patch, to_begin=None)

    index_list_salt = [] # List of index numbers for our mask
    patch_list_salt = [] # List of patch numbers for our mask
    if mask_flattened[0] == 0: # The index_list_salt and patch_list_salt for the masks in which mask_flattened[0] = 0
        if pixel_change_location[0] != 0:
            for i in pixel_change_location[::2]:
                i = i + 1
                index_list_salt.append(i)
            for j in patch_lengths[::2]:
                patch_list_salt.append(j)
        elif np.all(pixel_change_location[:] == 0): # Account for the case in which pixel_change_location doesn't change (i.e., mask_flattened is all 0 or it is all 65535)
            index_list_salt = []
            patch_list_salt = []
        elif pixel_change_location[0] == 0: 
            for i in pixel_change_location[1::2]:
                i = i + 1
                index_list_salt.append(i)
            for j in patch_lengths[1::2]: 
                patch_list_salt.append(j)
    if mask_flattened[0] == 65535: # The index_list_salt and patch_list_salt for the masks in which mask_flattened[0] = 65535
        if pixel_change_location[0] == 0:
            for k in pixel_change_location[::2]:
                k = k + 1
                index_list_salt.append(k)
            for l in patch_lengths[::2]:
                patch_list_salt.append(l)
        elif pixel_change_location[0] != 0: # We took into account that sometimes the pixel_change_location does not start with a 0
            index_list_start = 1 
            index_list_salt.append(index_list_start)
            patch_list_start = pixel_change_location[0] 
            patch_list_salt.append(patch_list_start)
            for k in pixel_change_location[1::2]:
                k = k + 1
                index_list_salt.append(k)
            for l in patch_lengths[1::2]:
                patch_list_salt.append(l)

    # Returns a run-length encoded mask, which is a list that combines the lists called 'index_list_salt' and 'patch_list_salt'
    our_mask = [None]*(len(index_list_salt)+len(patch_list_salt))
    our_mask[::2] = index_list_salt
    our_mask[1::2] = patch_list_salt
    return our_mask

def train_matches(filename, image_index):
    """
    This takes in a file and an image index. This returns a list, which is the training 
    run-length encoded mask. 
    """
    with open(filename) as f:
        lines_list = f.readlines()[1:]
    train_mask = []
    train_mask_lines = lines_list[image_index].strip().split(',')[1:]
    for str in train_mask_lines:
        if str != '':
            num = int(str)
            train_mask.append(num)
    return train_mask

def mask_matches(train_mask):
    '''
    This checks whether or not the training run-length encoded mask matches our run-length encoded mask. 
    It then prints whether or not these two masks match. 
    '''
    our_mask = make_mask(picture_mask)
    matches = our_mask == train_mask
    if matches == True:
        print ('Our mask matches the train.csv mask')
    else:
        print ('Our mask does not match the train.csv mask')

'''
    # Plot the image in greyscale
    plt.subplot(1,2,1)
    plt.imshow(picture, cmap='gray')
    plt.title('Image {}'.format(image_name))

    # Plot the corresponding mask in greyscale
    plt.subplot(1,2,2)
    plt.imshow(picture_mask, cmap='gray')
    plt.title('Mask {}'.format(image_name))
    plt.tight_layout()
    plt.show()
'''

filename = '/Users/sa4312/Dropbox/tgs/data_set/all/train.csv'

'''
image_index = 3600 # A chosen image index from the train.csv file
picture_mask = pic_mask(filename, image_index)
our_mask = make_mask(picture_mask)
train_mask = train_matches(filename, image_index)
print ('The train mask is: {}'.format(train_mask))
print ('Our mask is: {}'.format(our_mask))
mask_matches(train_mask)
'''

all_masks(filename)

if __name__ == "__main__":
    all_masks(filename)
