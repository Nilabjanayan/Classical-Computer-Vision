import numpy as np
#### DO NOT IMPORT cv2 

def my_imfilter(image, filter):
    
    """
  Apply a filter to an image. Return the filtered image.

  Args
  - image: numpy nd-array of dim (m, n, c)
  - filter: numpy nd-array of dim (k, k)
  Returns
  - filtered_image: numpy nd-array of dim (m, n, c)

  HINTS:
  - You may not use any libraries that do the work for you. Using numpy to work
   with matrices is fine and encouraged. Using opencv or similar to do the
   filtering for you is not allowed.
  - I encourage you to try implementing this naively first, just be aware that
   it may take an absurdly long time to run. You will need to get a function
   that takes a reasonable amount of time to run so that I can verify
   your code works.
  - Remember these are RGB images, accounting for the final image dimension.
     """

  

  ############################
  ### TODO: YOUR CODE HERE ###

    assert filter.shape[0] % 2 == 1
    assert filter.shape[1] % 2 == 1
    if len(image.shape) == 3:
        img_h,img_w,img_d = image.shape
        fil_h,fil_w = filter.shape
        image_p0 = np.pad(image[:,:,0],((fil_h//2,fil_h//2),(fil_w//2,fil_w//2)),'reflect')
        image_p1 = np.pad(image[:,:,1],((fil_h//2,fil_h//2),(fil_w//2,fil_w//2)),'reflect')
        image_p2 = np.pad(image[:,:,2],((fil_h//2,fil_h//2),(fil_w//2,fil_w//2)),'reflect')
        image_p = np.dstack([image_p0,image_p1,image_p2])
        filtered_image1 = np.zeros((image_p.shape[0],image_p.shape[1],img_d))
        for d in range(img_d):
            for i in range(img_h):
                for j in range(img_w):
                    filtered_image1[i+fil_h//2,j+fil_w//2,d] = np.sum(image_p[i:i+fil_h,j:j+fil_w,d]*filter)
        filtered_image = filtered_image1[fil_h//2:filtered_image1.shape[0]-fil_h//2,fil_w//2:filtered_image1.shape[1]-fil_w//2]
    else:
        img_h,img_w = image.shape
        fil_h,fil_w = filter.shape
        image_p = np.pad(image,((fil_h//2,fil_h//2),(fil_w//2,fil_w//2)),'reflect')
        filtered_image1 = np.zeros((image_p.shape[0],image_p.shape[1]))
        for i in range(img_h):
            for j in range(img_w):
                filtered_image1[i+fil_h//2,j+fil_w//2] = np.sum(image_p[i:i+fil_h,j:j+fil_w]*filter)
        filtered_image = filtered_image1[fil_h//2:filtered_image1.shape[0]-fil_h//2,fil_w//2:filtered_image1.shape[1]-fil_w//2]
        
    
  ### END OF STUDENT CODE ####
  ############################

    return filtered_image

def create_hybrid_image(image1, image2, filter):
    """
  Takes two images and creates a hybrid image. Returns the low
  frequency content of image1, the high frequency content of
  image 2, and the hybrid image.

  Args
  - image1: numpy nd-array of dim (m, n, c)
  - image2: numpy nd-array of dim (m, n, c)
  Returns
  - low_frequencies: numpy nd-array of dim (m, n, c)
  - high_frequencies: numpy nd-array of dim (m, n, c)
  - hybrid_image: numpy nd-array of dim (m, n, c)

  HINTS:
  - You will use your my_imfilter function in this function.
  - You can get just the high frequency content of an image by removing its low
    frequency content. Think about how to do this in mathematical terms.
  - Don't forget to make sure the pixel values are >= 0 and <= 1. This is known
    as 'clipping'.
  - If you want to use images with different dimensions, you should resize them
    in the notebook code.
     """
    
    assert image1.shape[0] == image2.shape[0]
    assert image1.shape[1] == image2.shape[1]
    if (len(image1.shape) == 3 and len(image2.shape) == 3):
        assert image1.shape[2] == image2.shape[2]
    
  ############################
  ### TODO: YOUR CODE HERE ###
    low_frequencies = my_imfilter(image1,filter)
    filtered_image2 = my_imfilter(image2,filter)
    high_frequencies = image2 - filtered_image2
    hybrid_image = low_frequencies + high_frequencies

  ### END OF STUDENT CODE ####
  ############################

    return low_frequencies, high_frequencies, hybrid_image


def myfilter(n,sigma):
    assert n % 2 == 1
    filter = np.zeros((n,n))
    for i in range(n):
        for j in range(n):
            x=i-n//2; y=j-n//2
            filter[i,j] = (1/(2*np.pi*(sigma**2)))*(np.exp((-(x**2 + y**2)/2*(sigma**2))))
    return filter                      
