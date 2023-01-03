# Vehicle-identification-using-UAV-imagery-data
The objective is to detect vehicles from UAV imagery data using the deep learning-based  U-NET model.
# Active repository
The process is divided into three sub-processes:
1) pre-processing
2) training
3) testing

My contribution is this repository, which simplifies the process by saving the trained model, predicted result, and plot into individual folders.

Process steps for vehicle identification (Sementic segmentation):
0) Clone the repository using this command: git clone https://github.com/bhagatdas/Vehicle-identification-using-UAV-imagery-data.git
1) Place the UAV or aerial imagery data into the image folder.
2) Place the matching mask into the mask folder.
3) Install all the dependencies:
        pip install patchify
        pip install segmentation-models
        pip install keras-unet
        pip install tifffile
        pip install pystackreg
        pip install tensorflow-addons
        pip install git+https://github.com/JanMarcelKezmann/TensorFlow-Advanced-Segmentation-Models.git
        pip install mlxtend
        pip install Augmentor
4) Now, run train the model from the terminal using this command after changing the folder path.:  python train.py
5) Automatically, the trained model will be saved in the model folder.
6) Now, run test the model from the terminal using this command after changing the folder path.:  python test.py
7) Corresponding evaluation matrices like IOU vs. EPOCH will be saved in the output folder.

# ENJOY 
