import os 
import numpy as np
import scipy.misc
# We installed Pillow
import matplotlib.pyplot as plt


def sunshines_mask(filename):
    """
    This looks at images and provides masks that match a train masks, based on the images.
    """
    
    with open(filename) as f:
        lines_list = f.readlines()[1:]

    # Getting the number of lines in the train mask file
    num_images = len(lines_list)

    matches_true = []  # A list of the amount of True matches between our mask and the train mask
    matches_not_true = [] # A list of image indexes in which the train mask does not match our mask
    # *** Note: if you want to run all of the masks, use the code below ***
    for image_index in range(num_images):
        image_name = lines_list[image_index].strip().split(',')[0]
        f_name = os.path.join('/Users', 'sa4312', 'Dropbox', 'tgs', 'data_set', 'all', 'train', 'images', image_name + '.png')
        print ('Image index {} has the name: {}'.format((image_index), image_name))

        f_name_mask = os.path.join('/Users', 'sa4312', 'Dropbox', 'tgs','data_set', 'all', 'train', 'masks', image_name + '.png')
        picture_mask = scipy.misc.imread(f_name_mask, flatten=True)
        # Flatten the above array called 'picture_mask' in order to create a single row of the pixel numbers
        mask_flattened = np.transpose(picture_mask).flatten()
        # The number in 'mask_flattened' is 65535 if it is a white pixel (salt), but the number is 0 if its a black pixel (no salt)
        mask_flattened_no_salt = np.all(mask_flattened[:] == 0) 
        mask_flattened_salt = np.all(mask_flattened[:] == 65535) 
        if mask_flattened_no_salt == True:
            pixel_change_location = np.array([0])
            Len_last_patch = len(mask_flattened)
        elif mask_flattened_salt == True:
            pixel_change_location = np.array([0])
            Len_last_patch = len(mask_flattened)
        else:
            pixel_change_location = np.where(np.roll(mask_flattened,1) != mask_flattened)[0] # The pixel location of where the pixel changes (e.g., from black to white or vice versa) on the mask image
            Len_last_patch = len(mask_flattened) - pixel_change_location[-1]

        # The differences between consecutive elements (i.e., subtracted consecutive elements) of the array called 'pixel_change_location'
        patch_lengths = np.ediff1d(pixel_change_location, to_end=Len_last_patch, to_begin=None)
        print ('Patch lengths: ', patch_lengths) 

        # list of index numbers for our mask
        index_list_salt = []
        # list of patch numbers for our mask
        patch_list_salt = []
        if mask_flattened[0] == 0:
            if pixel_change_location[0] != 0:
                for i in pixel_change_location[::2]:
                    i = i + 1
                    index_list_salt.append(i)
                for j in patch_lengths[::2]:
                    patch_list_salt.append(j)
            elif np.all(pixel_change_location[:] == 0): # we had to account for the case in which pixel_change_location doesn't change (i.e., mask_flattened is all 0 or it is all 65535)
                index_list_salt = []
                patch_list_salt = []
            elif pixel_change_location[0] == 0: # pixel_change_location begins with 0 for some mask images, so the below list in the for loop needs to start with a 1
                for i in pixel_change_location[1::2]:
                    i = i + 1
                    index_list_salt.append(i)
                for j in patch_lengths[1::2]: 
                    patch_list_salt.append(j)
        # The index_list_salt and patch_list_salt for the masks in which mask_flattened[0] = 65535 (i.e., mask_flattenned begins with a white pixel)
        if mask_flattened[0] == 65535:
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
        print ('index_list_salt: {}'.format(index_list_salt[0:20]))
        print ('patch_list_salt: {}'.format(patch_list_salt[0:20]))

        # Get a list that gives you a mask according to the image index that you specify
        # provide a list that combines the two lists directly above, in order to match the train.csv list
        our_mask = [None]*(len(index_list_salt)+len(patch_list_salt))
        our_mask[::2] = index_list_salt
        our_mask[1::2] = patch_list_salt
        print ('Our mask is: {}'.format(our_mask))

        # Each row of the masks in the train.csv file
        train_mask = []
        train_mask_lines = lines_list[image_index].strip().split(',')[1:]
        for str in train_mask_lines:
            if str != '':
                num = int(str)
                train_mask.append(num)
        print ('The train mask is: {}'.format(train_mask))
        
        matches = our_mask == train_mask
        if matches == True:
            print ('Our mask matches the train.csv mask')
            matches_true.append(matches) # We put the number of matches in a list called matches_true
        elif matches == False:
            print ('Our mask does not match the train.csv mask')
            matches_not_true.append(image_index)

    print ('The number of our masks that match the train masks is: ', sum(matches_true))
    print ('The number of our masks that do not match the train masks is: ', (num_images - sum(matches_true)))
    print ('The image indexes in which our masks and the train masks do not match are: ', matches_not_true)

    '''
    # *** Note: if you want to run the mask individually, use the code below ***
    # Creating a path to an image with the correct file format
    # Enter the image number (as image_index) from the train.csv file (minus 2 digits when you input it here in the code)
    image_index = 3613
    image_name = lines_list[image_index].strip().split(',')[0]
    f_name = os.path.join('/Users', 'sa4312', 'Dropbox', 'tgs', 'data_set', 'all', 'train', 'images', image_name + '.png')
    print ('Image index {} has the name: {}'.format((image_index), image_name))

    f_name_mask = os.path.join('/Users', 'sa4312', 'Dropbox', 'tgs','data_set', 'all', 'train', 'masks', image_name + '.png')
    picture_mask = scipy.misc.imread(f_name_mask, flatten=True)
    # Flatten the above array called 'picture_mask' in order to create a single row of the pixel numbers
    mask_flattened = np.transpose(picture_mask).flatten()
    # The number in 'mask_flattened' is 65535 if it is a white pixel (salt), but the number is 0 if its a black pixel (no salt)
    mask_flattened_no_salt = np.all(mask_flattened[:] == 0) 
    mask_flattened_salt = np.all(mask_flattened[:] == 65535) 
    if mask_flattened_no_salt == True:
        pixel_change_location = np.array([0])
        Len_last_patch = len(mask_flattened)
    elif mask_flattened_salt == True:
        pixel_change_location = np.array([0])
        Len_last_patch = len(mask_flattened)
    else:
        pixel_change_location = np.where(np.roll(mask_flattened,1) != mask_flattened)[0] # The pixel location of where the pixel changes (e.g., from black to white or vice versa) on the mask image
        Len_last_patch = len(mask_flattened) - pixel_change_location[-1]

    # The differences between consecutive elements (i.e., subtracted consecutive elements) of the array called 'pixel_change_location':
    patch_lengths = np.ediff1d(pixel_change_location, to_end=Len_last_patch, to_begin=None)

    # list of index numbers for our mask
    index_list_salt = []
    # list of patch numbers for our mask
    patch_list_salt = []
    if mask_flattened[0] == 0:
        if pixel_change_location[0] != 0:
            for i in pixel_change_location[::2]:
                i = i + 1
                index_list_salt.append(i)
            for j in patch_lengths[::2]:
                patch_list_salt.append(j)
        elif np.all(pixel_change_location[:] == 0): # we had to account for the case in which pixel_change_location doesn't change (i.e., mask_flattened is all 0 or it is all 65535)
            index_list_salt = []
            patch_list_salt = []
        elif pixel_change_location[0] == 0: 
            for i in pixel_change_location[1::2]:
                i = i + 1
                index_list_salt.append(i)
            for j in patch_lengths[1::2]: 
                patch_list_salt.append(j)
    # The index_list_salt and patch_list_salt for the masks in which mask_flattened[0] = 65535
    if mask_flattened[0] == 65535:
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

    # Get a list that gives you a mask according to the image index that you specify
    # provide a list that combines the two lists directly above, in order to match the train.csv list
    our_mask = [None]*(len(index_list_salt)+len(patch_list_salt))
    our_mask[::2] = index_list_salt
    our_mask[1::2] = patch_list_salt
    print ('Our mask is: {}'.format(our_mask))

    # Each row of the masks in the train.csv file
    train_mask = []
    train_mask_lines = lines_list[image_index].strip().split(',')[1:]
    for str in train_mask_lines:
        if str != '':
            num = int(str)
            train_mask.append(num)
    print ('The train mask is: {}'.format(train_mask))

     # Check to see if our mask matches the train mask
    matches = our_mask == train_mask
    if matches == True:
        print ('Our mask matches the train.csv mask')
    else:
        print ('Our mask does not match the train.csv mask')

    '''

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


if __name__ == "__main__":
    sunshines_mask('/Users/sa4312/Dropbox/tgs/data_set/all/train.csv')
