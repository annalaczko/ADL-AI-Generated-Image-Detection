# ADL - AI Generated Image Detection *(Bring your own method)*
 
**Name:** Anna Laczkó
 
**Matriculation number:** 12347556 
 
A project made for my Applied Deep Learning Course at TU Wien

## Assignment 1. Initiate

### Chosen project

My chosen topic is Detecting AI-generated images. Nowadays, my Facebook feed is filled with AI-generated content. I am good at recognising these, but most of my friends and family struggle. It is also quite scary to see all those people in the comments who are victims of the clickbait posts. 

I think using some tools to help these media users recognise fake content would be beneficial. Not only would it help detect fake information and stop the growing distrust of visual arts, but it would also support the artist who is concerned about their livelihood being taken away by the overuse of AI art. I am motivated to see how strong a model I can put together for detecting these images, and I want to familiarise myself with methods to help this process.

### The approach
I want to implement a (hopefully) better neural network for this project, so my chosen project type is **Bring your "own" method**. I would like to take a closer look at the methods and models implemented in the papers and use them. I hope that I can identify strengths and weaknesses and make some adjustments that can improve the results.

My network design will probably be a CNN or a dual-stream network, although I want to leave some room to try out new methods if both aren't working for me. Obviously, my first goal is to implement one of them and, with that, reach the precision mentioned in the corresponding paper.

One thing I am also thinking on, is using an ensemble modell, hoping that a corrected weighted version of it could help me generalise the model, and also make it more precise. I want to use cross-validation to avoid overfitting.

For evaluating the results right now, I am undecided between accuracy, precision, recall, F1, ROC-AUC, and the Confusion Matrix. I want to research these metrics more and find more information about them. Of course, to compare my results to the ones in the papers, I have to use the same metrics as they did.

### The dataset(s)
The three collected papers recommend a wide variety of datasets, and I'm aiming to use those right now.

- Real datasets: CIFAR-10, Flickr-Faces-HQ (FFHQ)
- Fake datasets: CIFAKE

I also found a Kaggle dataset containing CIFAR-10 and CIFAKE. I would like to read through that, and because I have found multiple sources of information about those, I will use those as my starting datasets.

#### CIFAR-10 and CIFAKE: 
CIFAR-10 is a subset of the dataset Tiny Images. It contains 60000 images from 10 classes. 
CIFAKE is a merge of the CIFAR-10 dataset and a similar sized dataset which contains a similar structure of 60000 AI-generated images for the same 10 classes.

### The schedule

Because of my other lecture, I want to do this task in bigger blocks.

| **Task** | **Estimated Time** | **Scheduling** |
|-------------------------------------------|--------------------|-------------------------|
| **Data Preparation and research** | 15-20 hours        | 23-10-2024 - 30-10-2024   |
| **Training the Model(s)** | 20-30 hours        | 23-10-2024 - 30-11-2024   |
| **Evaluate Performance** | 6-8 hours          | 20-11-2024 - 10-12-2024   |
| **Ensemble Model** | 10-12 hours        | 20-11-2024 - 10-12-2024   |
| **Comparing Ensemble to Individual Models**| 4-5 hours         | 05-12-2024 - 12-12-2024   |
| **Buffer time** | -                  | 12-12-2024 - 17-12-2024   |
| **Documentation for Assignment 2** | 8-10 hours        | Simultaneously with every previous step|
| **Building application** | 20-25 hours        | 28-12-2024 - 10-01-2025|
| **Final Report and preparing for presentation** | 10-15 hours        | 10-01-2025 - 15-01-2025|
| **Buffer time** | -                  | 15-01-2025 - 21-01-2025   |


## Assignment 2. Hacking

### Changes from Assignment 1.

I got my feedback for Assignment 1., which mentioned that it would be better to expand the CIFAKE dataset with another one to include images that better represent the state-of-the-art. With this in mind, I deleted CIFAKE because I felt that using those would only distort the results, and I wanted to use only newer images. I tried several datasets, but as I had limited time because I could not start the project in time and also had storage problems on my PC, I chose the [140k Real and Fake Faces](https://www.kaggle.com/datasets/xhlulu/140k-real-and-fake-faces/versions/1/data). 

This, of course, limits the model's capabilities as it will be only trained on portraits. However, I also felt that recognising faces as fake or not is really important because of the many deepfakes that are around nowadays. Also, for this dataset, I could see that some people had pretty good results, so I could rely on that (before, I had found several datasets that were simply not good enough for classification).

### Workflow

I started with only notebooks. I made one for preprocessing and one for the neural network and training. First, I used smaller images, like CIFAKE, to see if I could get even good results using tiny pictures. This showed promise, but I quickly realised that having bigger images dramatically improves the results. I quickly changed the preprocessing to (128,128) pictures.

**What kind of network I made**

The structure is simple; I have four convolutional layers and two fully connected layers. I focused on optimising the layers.  For this, I did some research. The paper on CIFAKE mentioned that most AI images are recognised not by the central object but by the surrounding more minor details. This is why I tried to shape my network to pick up those details:

- I added as many channels as I could
- I added a 5x5 kernel, but I added stride, as I thought that most of the details would be repetitive (and this is also from my own experience), and it would be redundant to go one by one.
- I stayed away from pooling

As I mentioned before, I tried to use smaller pictures. I also thought that if my time allowed it, I would train on three different sizes and then do majority voting. With this, I could even extract different features from the images, but sadly, the lower-resolution pictures were not that effective. 

Another attempt at an Ensemble Model was an idea that I found in [this](https://paperswithcode.com/paper/ai-generated-image-detection-using-a-cross) paper. Here, they tried to make filtered versions of the pictures, highlighting edges and sharpening the images and using that to identify differences between real and AI-generated images. I implemented the filtration in Preprocess.py, but in the end, due to time constraints, I did not use those images for the final training. However, I did some testing with them, and Edge and Sharpen proved to be quite good.

I still think it would be interesting to make an ensemble with them; however, I did not have time to implement that.

After the code worked in the notebooks, I manually transformed them into .py-s and made some tests for them.

### Goals and Results

For this task, I chose the standard accuracy to measure my model's success. My goal was to classify at least 75-80 % of the pictures correctly from the validation set.

In the end, I achieved a 95.02% accuracy, which was a shock (although a positive one). 

For more information, I added a confusion matrix to analyse the accuracy of the two labels.

|           |True 0|True 1|
|-----------|------|------|
|Predicted 0| 18076|   511|
|predicted 1|  2924| 47489|

### Actual Schedule compared to the planned one

| **Task** | **Estimated Time** | **Scheduling** | **Spent time** | **Actual Scheduling** | 
|-------------------------------------------|--------------------|-------------------------|--|--|
| **Data Preparation and research** | 15-20 hours        | 23-10-2024 - 30-10-2024   | 30-35 hours | 04-12-2024 - 15-12-2024|
| **Training the Model(s)** | 20-30 hours        | 23-10-2024 - 30-11-2024   | 35-40 hours | 08-12-2024 - 17-12-2024|
| **Evaluate Performance** | 6-8 hours          | 20-11-2024 - 10-12-2024   | 2 hours | 08-12-2024 - 15-12-2024|
| **Ensemble Model** | 10-12 hours        | 20-11-2024 - 10-12-2024   | 5 hours | 13-12-2024 |
| **Comparing Ensemble to Individual Models** | 4-5 hours         | 05-12-2024 - 12-12-2024   | 1 hours | 13-12-2024 |
| **Buffer time** | -                  | 12-12-2024 - 17-12-2024   | - | - |
| **Documentation for Assignment 2** | 8-10 hours        | Simultaneously with every previous step| 6 hours | 14-12-2024 - 17-12-2024 |
| **Building application** | 20-25 hours        | 28-12-2024 - 10-01-2025| ||
| **Final Report and preparing for presentation** | 10-15 hours        | 10-01-2025 - 15-01-2025|||
| **Buffer time** | -                  | 15-01-2025 - 21-01-2025   |||

As it can be seen, the biggest thing I learned and miscalculated that data collection takes much more time.

### How to Set Up and Run the Project

#### 1. Clone the Repository

git clone https://github.com/your-username/your-repository.git
cd your-repository

#### 2. Download and Extract Data

Download the dataset from [here](https://drive.google.com/drive/folders/1tqMOQfYcUJrGCt6cJPvw_NHCmXyN82Cj?usp=sharing), and extract it to the root of the project directory. This is a restructured version of the [140k Real and Fake Faces](https://www.kaggle.com/datasets/xhlulu/140k-real-and-fake-faces/versions/1/data) on Kaggle.

#### 3. Install Dependencies

Open up console in the root folder and run

``` pip install -r docs/requirements.txt ```

#### 4. Run the notebook

To see the results simply open Notebook.ipynb and run it.

#### 5. Access Documentation

For the .py files the documentation can be found [HERE](https://adl-ai-generated-image-detection.readthedocs.io/en/latest/index.html)

#### 6. Run Tests

To run tests:

```python -m unittest discover .```

### LLM use

I used ChatGPT for the following:

- .ipynb to .py transformation debugging
    - Here, ChatGPT was dead on point and found every missed function and typo made when transferring.
- Test generation
    - As I haven't used testing in Python before, I needed some help understanding the way it works
- Comment generation
    - I commented throughout the whole process. Read the Docs; however, expects the comments to be in another format. Additionally, looking back, I realize that my comments were not enough. So, in the end, during the documentation generation process, I put my whole .py files into ChatGPT and asked to convert my comments into the correct format and extend them if needed.
- General debugging
- Understanding error messages

## References:
- Jordan J. Bird, Ahmad Lotfi **2023.** *CIFAKE: Image Classification and Explainable Identification of AI-Generated Synthetic Images* https://arxiv.org/abs/2303.14126
- Ziyi Xi, Wenmin Huang, Kangkang Wei, Weiqi Luo, Peijia Zheng **2023.** *AI-Generated Image Detection using a Cross-Attention Enhanced Dual-Stream Network* https://paperswithcode.com/paper/ai-generated-image-detection-using-a-cross
- Zeyu Lu, Di Huang, LEI BAI, Jingjing Qu, Chengyue Wu, Xihui Liu, Wanli Ouyang **2023.** *Seeing is not always believing: Benchmarking Human and Model Perception of AI-Generated Images* https://paperswithcode.com/paper/seeing-is-not-always-believing-benchmarking
- Krizhevsky, A., & Hinton, G. (2009). Learning multiple layers of features from tiny images. 
- Bird, J.J. and Lotfi, A., 2024. CIFAKE: Image Classification and Explainable Identification of AI-Generated Synthetic Images. IEEE Access.
- Real images are from Krizhevsky & Hinton (2009), fake images are from Bird & Lotfi (2024). The Bird & Lotfi study is available [here](https://ieeexplore.ieee.org/abstract/document/10409290).
- https://www.kaggle.com/datasets/xhlulu/140k-real-and-fake-faces/versions/1/data
- https://www.kaggle.com/datasets/birdy654/cifake-real-and-ai-generated-synthetic-images


*The text was refined using Grammarly for enhanced English accuracy.*