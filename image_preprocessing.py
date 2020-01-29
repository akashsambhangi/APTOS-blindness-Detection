import numpy as np
import cv2
from tqdm import tqdm

#https://codereview.stackexchange.com/questions/132914/crop-black-border-of-image-using-numpy/132934
def crop_dark_extras(img,tol=7):
    '''
    This function is used to crop out the additions dark areas in the images as this information is not useful
    '''
    if img.ndim ==2:
        mask = img>tol
        return img[np.ix_(mask.any(1),mask.any(0))]
    elif img.ndim==3:
        gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        mask = gray_img>tol
        
        check_shape = img[:,:,0][np.ix_(mask.any(1),mask.any(0))].shape[0]
        if (check_shape == 0): # image is too dark so that we crop out everything,
            return img # return original image
        else:
            img1=img[:,:,0][np.ix_(mask.any(1),mask.any(0))]
            img2=img[:,:,1][np.ix_(mask.any(1),mask.any(0))]
            img3=img[:,:,2][np.ix_(mask.any(1),mask.any(0))]
            
            img = np.stack([img1,img2,img3],axis=-1)
        return img
    



def Preprocessing_images(df, sub_dir, ben=False, sigmaX=10, img_size=224, ext='.png'):
    '''
    This function reads images, converts them to RGB, resizes the image and also enhances the images/
    by blending them with gaussianBluered version of itself
    '''
    rows = df.shape[0]
    df_as_nd_array = np.empty((rows,img_size,img_size,3),dtype=np.uint8)

    for i,id_code in enumerate(tqdm(df)):
        path = "D:/Blindness Detection/"+sub_dir+'/'+ id_code + ext
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = crop_dark_extras(img=img,tol=7)
        img = cv2.resize(img, (img_size,img_size))
        if ben == True:
            #if the kernal size is [0,0] in gaussian blur then sigmaX is used for deciding the kernel size
            # https://www.tutorialkart.com/opencv/python/opencv-python-gaussian-image-smoothing/
            img = cv2.addWeighted(img,4,cv2.GaussianBlur(img,(0,0),sigmaX),-4,128)
        df_as_nd_array[i,:,:,:] = img
        
    return df_as_nd_array