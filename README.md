# Galaxy Classification
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





