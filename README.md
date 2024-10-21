# ADL - AI Generated Image Detection *(Bring your own method)*
 **Name:** Anna Laczkó
 
 **Matriculation number:** 12347556 
 
 A project made for my Applied Deep Learning Course at TU Wien

## Chosen project

My chosen topic is Detecting AI-generated images. Nowadays, my Facebook feed is filled with AI-generated content. I am good at recognising these, but most of my friends and family struggle. It is also quite scary to see all those people in the comments who are victims of the clickbait posts. 

I think using some tools to help these media users recognise fake content would be beneficial. Not only would it help detect fake information and stop the growing distrust of visual arts, but it would also support the artist who is concerned about their livelihood being taken away by the overuse of AI art. I am motivated to see how strong a model I can put together for detecting these images, and I want to familiarise myself with methods to help this process.

## The approach
I want to implement a (hopefully) better neural network for this project, so my chosen project type is **Bring your "own" method**. I would like to take a closer look at the methods and models implemented in the papers and use them. I hope that I can identify strengths and weaknesses and make some adjustments that can improve the results.

My network design will probably be a CNN or a dual-stream network, although I want to leave some room to try out new methods if both aren't working for me. Obviously, my first goal is to implement one of them and, with that, reach the precision mentioned in the corresponding paper.

One thing I am also thinking on, is using an ensemble modell, hoping that a corrected weighted version of it could help me generalise the model, and also make it more precise. I want to use cross-validation to avoid overfitting.

For evaluating the results right now, I am undecided between accuracy, precision, recall, F1, ROC-AUC, and the Confusion Matrix. I want to research these metrics more and find more information about them. Of course, to compare my results to the ones in the papers, I have to use the same metrics as they did.

## The dataset(s)
The three collected papers recommend a wide variety of datasets, and I'm aiming to use those right now.

- Real datasets: CIFAR-10, Flickr-Faces-HQ (FFHQ)
- Fake datasets: CIFAKE

I also found a Kaggle dataset containing CIFAR-10 and CIFAKE. I would like to read through that, and because I have found multiple sources of information about those, I will use those as my starting datasets.

## The schedule

Because of my other lecture, I want to do this task in bigger blocks.

| **Task** | **Estimated Time** | **Scheduling** |
|-------------------------------------------|--------------------|-------------------------|
| **Data Preparation and research** | 15-20 hours        | 2024.10.23-2024.10.30   |
| **Training the Model(s)** | 20-30 hours        | 2024.10.23-2024.11.30   |
| **Evaluate Performance** | 6-8 hours          | 2024.11.20-2024.12.10   |
| **Ensemble Model** | 10-12 hours        | 2024.11.20-2024.12.10   |
| **Comparing Ensemble to Individual Models**| 4-5 hours         | 2024.12.05-2024.12.12   |
| **Buffer time** | -                  | 2024.12.12-2024.12.17   |
| **Final Report** | 10-15 hours        | Simultaneously with every other step|


## Research papers:
- Jordan J. Bird, Ahmad Lotfi **2023.** *CIFAKE: Image Classification and Explainable Identification of AI-Generated Synthetic Images* https://arxiv.org/abs/2303.14126
- Ziyi Xi, Wenmin Huang, Kangkang Wei, Weiqi Luo, Peijia Zheng **2023.** *AI-Generated Image Detection using a Cross-Attention Enhanced Dual-Stream Network* https://paperswithcode.com/paper/ai-generated-image-detection-using-a-cross
- Zeyu Lu, Di Huang, LEI BAI, Jingjing Qu, Chengyue Wu, Xihui Liu, Wanli Ouyang **2023.** *Seeing is not always believing: Benchmarking Human and Model Perception of AI-Generated Images* https://paperswithcode.com/paper/seeing-is-not-always-believing-benchmarking

## Other references
- https://www.kaggle.com/datasets/birdy654/cifake-real-and-ai-generated-synthetic-images

*The text was refined using Grammarly for enhanced English accuracy.*