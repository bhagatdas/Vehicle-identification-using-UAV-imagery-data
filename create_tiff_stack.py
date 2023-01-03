import glob
import tifffile as tf
from pystackreg import StackReg
from skimage import io
from matplotlib import pyplot as plt
import numpy as np

with tf.TiffWriter('image2/image2.tif') as stack:
    for filename in glob.glob('tiff_images/*.tif'):
        stack.save(
            tf.imread(filename), 
            photometric='minisblack', 
            contiguous=True
        )
image = tf.imread('image2/image2.tif')
image = np.transpose(image, axes=(1, 2, 0))
tf.imsave('image2/image2.tif',image)

image = tf.imread('image2/image2.tif')
print(image.shape)
# printing all tiff as png........................................
for i in range(image.shape[2]):
  plt.figure(figsize=(12,6))
  plt.imshow(image[:,:,i],cmap = 'gray')
  path ='image2/sample'+str(i)+'.png'
  plt.savefig(path, dpi=300)
  plt.show()

print("DONE PRINTING")