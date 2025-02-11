import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import sys

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
while project_root.split("\\")[-1].lower() not in ["computervisionsoc", "computervisionsociety"]:
    project_root = os.path.abspath(os.path.join(project_root, '..'))  # Update relative to the current project_root

sys.path.append(project_root)

os.chdir(project_root)
samples = []

for img in os.listdir("datasets/sample_image"):
    image = cv2.imread("datasets/sample_image/"+img)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (224, 224)) 

    samples.append(image)


edge_detection = np.array([[-1, 0, 1],
                    [-2, 0, 2],
                    [-1, 0, 1]])
blurring = np.array([[1, 2, 1],
                      [2, 4, 2],
                      [1, 2, 1]]) / 16

sharpening = np.array([[ 0, -1,  0],
                       [-1,  5, -1],
                       [ 0, -1,  0]])

embossing = np.array([[-2, -1,  0],
                   [-1,  1,  1],
                   [ 0,  1,  2]])


def apply_filter(image, kernel, padding="same"):
    # Get image and kernel dimensions
    img_height, img_width, num_channels = image.shape
    kernel_height, kernel_width = kernel.shape

    # Determine padding size
    if padding == "same":
        pad_height = ... #TODO
        pad_width = ... #TODO
        padded_image = np.pad(image, ((pad_height, pad_height), (pad_width, pad_width), (0, 0)), mode="constant", constant_values=0)
    else:
        padded_image = image

    output_img = np.zeros_like(image)

    #TODO

    return output_img

rand_img = samples[np.random.randint(0, 20)]
rand_img = rand_img/255
kernel = blurring # Choose the filter


filtered = (apply_filter(rand_img, kernel, "same"))
plt.subplot(1,2,1), plt.imshow(rand_img, cmap='gray'), plt.title("Original")
plt.subplot(1,2,2), plt.imshow(filtered, cmap='gray'), plt.title("Filtered")
plt.show()



def max_pooling(img, pool_size=2, stride=2):
    h, w = img.shape[:2]  # Get the height and width of the image
    out_h = ... #TODO
    out_w = ... #TODO
    pooled_img = np.zeros((out_h, out_w, img.shape[-1]))  # Output image after pooling

    
    return pooled_img

def avg_pooling(img, pool_size=2, stride=2):
    h, w = img.shape[:2]
    out_h = ... #TODO
    out_w = ... #TODO
    pooled_img = np.zeros((out_h, out_w, img.shape[-1]))

    #TODO

    return pooled_img


rand_img = samples[np.random.randint(0, 20)]
rand_img = rand_img/255

pooled = avg_pooling(rand_img, 5, 5) #Choose the pooling 

plt.subplot(1,2,1), plt.imshow(rand_img, cmap='gray'), plt.title("Original")
plt.subplot(1,2,2), plt.imshow(pooled, cmap='gray'), plt.title("Pooled")
plt.show()