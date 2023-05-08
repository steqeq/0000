# Introduction to Deep Learning Guide

The following sections provide a deeper understanding of Machine Learning (ML) and Deep Learning (DL) as they pertain to ROCm™.

## What Is Machine Learning?

```{figure} ../data/how_to/introduction_to_deep_learning_guide/image.001.png
:name: Artificial-Intelligence-Venn-Diagram
---
align: center
---
Artificial Intelligence Venn Diagram
```

Machine Learning is a field within Artificial Intelligence (AI) that focuses on an algorithm that can learn from experience or data and automatically improve outcomes. These algorithms can adapt to new circumstances, unlike explicitly programmed programs. These algorithms detect patterns in the underlying data and modify an underlying model to extrapolate to new situations. Figure 1 illustrates the interrelated relationship of each algorithm field within AI.

Within ML, this guide primarily focuses on supervised learning algorithms and the training data consisting of the model inputs and desired outputs during the training phase. The training algorithm learns the underlying model parameters through an iterative optimization process. During the evaluation phase, the model topology with the trained model parameters from the learned agent is evaluated based on its performance against new data.

## What Is Deep Learning?

Deep Learning algorithms are a class of ML algorithms inspired by biological neural networks and comprise multiple levels of superficial component layers. The earliest component classifier is the perceptron, a binary classifier comprising a linear, fully connected stage with a non-linear decision stage. Modern DL component layers include widely accepted layers such as:

- Convolution
- Activation
- Fully connected, long short-term memory
- Custom layers designed by application developers

The training of a DL network uses large amounts of data. DL algorithms use Deep Neural Networks to access, explore, and analyze vast sets of information—such as all the music files on streaming platforms that make ongoing suggestions based on the tastes of a specific user. The input—whether an image, a new article, or a song—is evaluated in its raw or untagged form with minimal transformation. This unsupervised training process is sometimes called representation learning. During training, the DL algorithm progressively learns from the data to improve the accuracy of its conclusions (also known as inference). Table 1 provides DL applications and their usage.

#### Table 1. Common Examples of Deep Learning

:::{table} Common Examples of Deep Learning
:name: CommonExamplesofDeepLearning
:widths: auto
| Deep Learning Examples | Usage |
|------------------------|-------|
| Autonomous Driving | Combining deep data (maps, satellite traffic images, weather reports, a user's accumulated preferences), real-time sensor input from the environment (a deer on the road, a swerving driver), and compute power to make decisions (slow down, turn the steering wheel). |
| Medical | Cancer research—for example, learning to detect melanoma in photos. |
| Smart Home | Smart speakers use intelligent personal assistants and voice-recognition algorithms to comprehend and respond to unique users' verbal requests. |

## Why Use AMD GPUs for Deep Learning?

Machine Learning and Deep Learning intelligent applications that respond with human-like reflexes require enormous computer processing power.

The main contributions of AMD to ML and DL systems come from delivering high-performance computing (both CPUs and GPUs) with an open ecosystem for software development. ML and DL applications rely on computer hardware to support the highest processing capabilities (speed, capacity, and organization) to manage complex data sets from multiple input streams simultaneously.

For example, in an autonomous driving scenario, the DL algorithm might be required to recognize an upcoming traffic light changing from green to yellow, nearby pedestrian movement, and water on the pavement from a rainstorm—among a variety of other real-time variables—as well as basic vehicle operations. A trained human driver may take these coordinating reactions for granted. But to simulate the human brain's capabilities, the autonomous driving algorithm needs efficient and accelerated processing to make complex decisions with sufficient speed and high accuracy for passengers and others around them.

The performance of AMD hardware and associated software also offers excellent benefits to developing and testing ML and DL systems. Today, a computing platform built with the latest AMD technologies (AMD EPYC™ CPUs and Radeon Instinct™ GPUs) can create and try a new intelligent application in days or weeks—a process that used to take years.
