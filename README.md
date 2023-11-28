# Galaxy Classification
## Instructions
To run be sure to have the proper libraries installed: h5py, numpy, tensorflow, opencv, itertools, and matplotlib </br>
The file containing my data, Galaxy10_DECals.h5 is too large to upload to GitHub. It can be accessed here: https://astronn.readthedocs.io/en/latest/galaxy10.html </br>
Or it can be loaded with this code: </br>
```python
from astroNN.datasets import load_galaxy10 
from tensorflow.keras import utils
import numpy as np
images, labels = load_galaxy10()
```


I avoided using this in my code since it redownloads the dataset everytime you run the program which took quite a bit of time so I decided to store it locally instead 
<h2>Part 1</h2>
For this project, I plan to implement computer vision algorithms to classify images of galaxies into 10 distinct subsets. I will utilize the extensive dataset from Henrysky's Galaxy10 (https://github.com/henrysky/Galaxy10). This dataset includes 17,736 galaxy images separated into the 10 classes listed below: <br/>
├── Class 0 (1081 images): Disturbed Galaxies <br/>
├── Class 1 (1853 images): Merging Galaxies <br/>
├── Class 2 (2645 images): Round Smooth Galaxies <br/>
├── Class 3 (2027 images): In-between Round Smooth Galaxies <br/>
├── Class 4 ( 334 images): Cigar Shaped Smooth Galaxies <br/>
├── Class 5 (2043 images): Barred Spiral Galaxies <br/>
├── Class 6 (1829 images): Unbarred Tight Spiral Galaxies <br/>
├── Class 7 (2628 images): Unbarred Loose Spiral Galaxies <br/>
├── Class 8 (1423 images): Edge-on Galaxies without Bulge <br/>
└── Class 9 (1873 images): Edge-on Galaxies with Bulge <br/>
I will be developing an algorithm to classify these galaxies based on shape. I'll need to separate the galaxy from the background and determine where the galaxy's edges are. In general, the galaxy will have a bright spot in the center surrounded by different shapes and sizes of dust and stars. I'll have to be able to find the galactic center which will allow me to work outwards to find the edge. I will work out from the center in a circle incrementally increasing in size since no matter the shape, galaxies are usually round. After detecting the edge of the galaxy I'll narrow down the image so that the background is excluded and then I will work to find the shape. This will require me to look for features such as arms, multiple galaxies merging, bulges, etc. To find these features I'll have to classify different key points that are unique to specific galaxy shapes. The color can be mostly ignored as many images of galaxies will be captured through various methods collecting many different wavelengths of light. No matter the method of capture the shape of the galaxy will always be constant and the background of empty space will always be black. Therefore, I'll still be able to separate the galaxy from the background no matter the wavelengths of light being collected. As a member of the graduate portion of this course, I will also indulge in research to further enhance my project. My research plan is, after developing the classical algorithms to classify these galaxies, I'll explore different machine learning techniques. I will examine how different techniques can help improve the accuracy and efficiency of my classification. Some neural network architectures I'll explore are Convolutional Neural Networks, Recurrent Neural Networks, and Transfer Learning. After researching these methods I will try and implement the method I think would work best with my problem. I'll use half of the dataset listed above for training and half for validation. I'll then use new images of galaxies that are being collected by NASA's James Webb Space Telescope to do my final testing.

<h2>Part 2</h2>
I've acquired one large dataset to use for the training, validation, and testing stages of this project.</br>
The dataset is linked here: https://astronn.readthedocs.io/en/latest/galaxy10.html</br>
This is the dataset I discussed before which contains 17,736 galaxy images classified into 10 distinct classes. The images in this dataset are colored and 256x256 pixels. The first 60% of the images in each class will be used for training and will be kept as they were originally found. I will use the next 20% of the images in each class for validation. These images will be converted to grayscale and run through the algorithm I develop. This is to make sure that color isn't a distinguishing feature as the classification of a galaxy should not depend on color at all. Lastly, I will use the final 20% of each class of images for testing. These images were captured in both the visible and infrared spectra using a g (green), r (red), and z (infrared) band filter.

<h2>Part 3</h2>
As my project is focused on developing a convolutional neural network, I'm not following the standard pipeline which I previously discussed with Professor Czajka. Since the preprocessing and classification are done at the same time by the neural network, I'll be describing the design process for my neural network. To begin, I took the code we used for classifying the handwritten numbers and adapted it for my data. At first, I just chose the first 60% of the images in the dataset and trained the model on it. This ended up training quickly to a very high training accuracy, but as I went to test the validation accuracy, I found the model was overfitting to the training data. This model needs to have a balance between identifying specific details and being able to identify the general shape of the galaxies. Some of the 10 classes look very similar, so the model has to be able to pick out the more minor details that distinguish between them. </br> 
As shown here in the images of a Round Galaxy, In-between Round Smooth Galaxy, and Cigar Round Smooth Galaxy, there aren't many apparent details that distinguish between the 3.</br>
<img width="846" alt="Screenshot 2023-11-27 at 8 52 29 PM" src="https://github.com/ty-berg/Galaxy-Classification/assets/70979891/a256dde5-ef69-43c0-b40d-b057a510887b"> </br>
Compare this to these images of an Unbarred Tight Spiral Galaxy, Unbarred Loose Spiral Galaxy, and Edge-on Without Bulge Galaxy, which are very obviously different.</br>
<img width="842" alt="Screenshot 2023-11-27 at 8 58 41 PM" src="https://github.com/ty-berg/Galaxy-Classification/assets/70979891/8d7f70b0-f38b-4e5d-8c57-166283f03ecb"> </br>

To account for this, I did some research to determine how I might adjust for the overfitting of the model while also being able to capture the needed details to distinguish between similar-looking galaxies. Before adjusting the model I made sure that the training data would be representative of the data as a whole. To do this I didn't just take the first 60% of all the data for training but rather took the first 60% of each class for training. I approximated that this would be an appropriate amount of data for each class to be properly trained into the model. Although it would be helpful to have a more even distribution of the classes, I did the best with the data we currently have. After dividing the images properly, I added a couple more convolutional layers and batch normalization. I also experimented with different amounts of dropout layers with different values. Although this helped with the overfitting a bit, it didn't completely solve the problem. I researched more and found that data augmentation could help with the overfitting problem. I rotate, shift the width, shift the height, horizontally flip, and zoom each image in a variety of ways so that the model is able to generalize better. This data augmentation allowed for the model to view the images in many different ways so that it wouldn't just train to the specific image and would be capable of identifying the galaxy type regardless of size and rotation. This seemed to make my model perform better in terms of preventing overfitting but due to the number of images, their size, the amount of augmentation, and layer details, it took a long time to train.</br> 

The training time made it difficult to adjust the model as it would take a while for me to observe if the changes I would make were improving the accuracy and convergence rate. After working on adjusting the model I left it running overnight and at this stage in the project I was able to achieve about 70% accuracy in both my validation and training stages. During this stage of the project I also experimented with different optimizers to see if this sped up convergence or increased accuracy. I found that the Adam optimizer worked best so I stuck with that. For the next stage of the project I decided to focus on how I can improve the convergence rate without sacrificing accuracy.

<h2>Part 4</h2>
For this stage of the project I was focused on improving the training time as it was taking hours for me to be able to observe enough training to evaluate how any changes I made affected the model. First I attempted to use different learning rate schedulers with my current model to see if this could help. I experimented with adaptive learning rate, exponential decay, polynomial decay and others but I didn't see any improvements by using a learning rate scheduler. I also tried to use Global Average Pooling to see how it would affect the model which also didn't improve the training time and decreased training accuracy. I kind of reached an impass here so I did some more research and learned about ResNet in class so I decided to see if using a pretrained model would help increase accuracy and decrease training time. Again I experimented with ResNet50 using the imagenet weights, I tried to experiment with the dense layers and different ResNet parameters but similar to the last few solutions I attempted this did not help with training time or accuracy. I tried to combine different solutions such as ResNet, Global Average Pooling, and learning rate schedulers but still was running into the same issues. Although I was able to achieve a reasonably high accuracy it took a long time which made adjusting the model very difficult. </br>

This then led me to looking more at how I could adjust the data so that the model would have an easier time training. I first attempted to convert the images to grayscale so that the model only had to work with one channel for each pixel. Although this sped up the training time a little bit this seemed to decrease the accuracy. I attempted to adjust the model to see if I could get the same accuracies I was getting with color images. I added more convolutional layers, took away convolutional layers, added more dropout, took away dropout, adjusted dropout values, changed the dense layers, and more. Although the grayscale images were producing fairly accurate results it still wasn't as accurate as I had with the color images. I decided that accuracy was the most important factor for my model so I moved away from working with grayscale images and tried to think of other ways I could adjust the data to help the model.</br>

This led me to my current solution where I decrease the resolution of the images. As I was researching for this project, I stumbled upon some people working with an older dataset of galaxy images which were 69x69 pixels. These images were much lower resolution than my dataset which is 256x256 pixels and because of this the people working with the older dataset separated the galaxies into less specific categories that relied on smaller details. They sorted them into different types of Smooth and Disk galaxies rather than including galaxy types like disturbed, merging, or different versions of spiral galaxies. Although using this older dataset would've been easier I still wanted to use the most up to date survey of our universe as that would be more useful in actual research. Even though I'm not using these lower resolution images, I took some inspiration from them and decided to see how my model would perform if I halved the resolution of my images. This drastically decreased the training time for each epoch from roughly 240 seconds to 45 seconds. This allowed me to experiment more with my layers and model parameters as I was now able to see the changes to the results much more quickly.</br>

Other than small changes to the layers the last big experiment I performed with my model was using different batch sizes. I settled on 32 since it helps with generalizing the model and preventing overfitting. Although 64 seemed to train quicker, the validation accuracies were more variable so I decided against using that. </br>

As I head into the final report one small thing I may work on is finding the best values for the dense layers as this could lead to some improvements. I also will be running the current version overnight for 500 epochs to see what happens to the accuracies. I've currently only run the committed version for 45 epochs but the accuracy made it to 71% while still increasing so I believe that it can get even higher. </br>

I also want to see what the kernels look like in each convolutional layer so I will be working with my code to properly visualize those heading into the final report.





