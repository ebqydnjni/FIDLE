<a name="top"></a>

[<img width="600px" src="fidle/img/00-Fidle-titre-01.svg"></img>](#top)

<!-- --------------------------------------------------- -->
<!-- To correctly view this README under Jupyter Lab     -->
<!-- Open the notebook: README.ipynb!                    -->
<!-- --------------------------------------------------- -->

## About Fidle

This repository contains all the documents and links of the **Fidle Training** .   
Fidle (for Formation Introduction au Deep Learning) is a 3-day training session co-organized  
by the 3IA MIAI institute, the CNRS, via the Mission for Transversal and Interdisciplinary  
Initiatives (MITI) and the University of Grenoble Alpes (UGA).  

The objectives of this training are :
 - Understanding the **bases of Deep Learning** neural networks
 - Develop a **first experience** through simple and representative examples
 - Understanding **Tensorflow/Keras** and **Jupyter lab** technologies
 - Apprehend the **academic computing environments** Tier-2 or Tier-1 with powerfull GPU

For more information, see **https://fidle.cnrs.fr** :
- **[Fidle site](https://fidle.cnrs.fr)**
- **[Presentation of the training](https://fidle.cnrs.fr/presentation)**
- **[Program 2022/2023](https://fidle.cnrs.fr/programme)**
- [Subscribe to the list](https://fidle.cnrs.fr/listeinfo), to stay informed !
- [Find us on youtube](https://fidle.cnrs.fr/youtube)
- [Corrected notebooks](https://fidle.cnrs.fr/done)

For more information, you can contact us at :  
[<img width="200px" style="vertical-align:middle" src="fidle/img/00-Mail_contact.svg"></img>](#top)

Current Version : <!-- VERSION_BEGIN -->2.2.4<!-- VERSION_END -->


## Course materials

| | | | |
|:--:|:--:|:--:|:--:|
| **[<img width="50px" src="fidle/img/00-Fidle-pdf.svg"></img><br>Course slides](https://fidle.cnrs.fr/supports)**<br>The course in pdf format<br>(12 Mo)| **[<img width="50px" src="fidle/img/00-Notebooks.svg"></img><br>Notebooks](https://fidle.cnrs.fr/notebooks)**<br> &nbsp;&nbsp;&nbsp;&nbsp;Get a Zip or clone this repository &nbsp;&nbsp;&nbsp;&nbsp;<br>(40 Mo)| **[<img width="50px" src="fidle/img/00-Datasets-tar.svg"></img><br>Datasets](https://fidle.cnrs.fr/datasets-fidle.tar)**<br>All the needed datasets<br>(1.2 Go)|**[<img width="50px" src="fidle/img/00-Videos.svg"></img><br>Videos](https://fidle.cnrs.fr/youtube)**<br>&nbsp;&nbsp;&nbsp;&nbsp;Our Youtube channel&nbsp;&nbsp;&nbsp;&nbsp;<br>&nbsp;|

Have a look about **[How to get and install](https://fidle.cnrs.fr/installation)** these notebooks and datasets.


## Jupyter notebooks

<!-- TOC_BEGIN -->
<!-- Automatically generated on : 12/04/23 09:28:57 -->

### Linear and logistic regression
- **[LINR1](LinearReg/01-Linear-Regression.ipynb)** - [Linear regression with direct resolution](LinearReg/01-Linear-Regression.ipynb)  
Low-level implementation, using numpy, of a direct resolution for a linear regression
- **[GRAD1](LinearReg/02-Gradient-descent.ipynb)** - [Linear regression with gradient descent](LinearReg/02-Gradient-descent.ipynb)  
Low level implementation of a solution by gradient descent. Basic and stochastic approach.
- **[POLR1](LinearReg/03-Polynomial-Regression.ipynb)** - [Complexity Syndrome](LinearReg/03-Polynomial-Regression.ipynb)  
Illustration of the problem of complexity with the polynomial regression
- **[LOGR1](LinearReg/04-Logistic-Regression.ipynb)** - [Logistic regression](LinearReg/04-Logistic-Regression.ipynb)  
Simple example of logistic regression with a sklearn solution

### Perceptron Model 1957
- **[PER57](IRIS/01-Simple-Perceptron.ipynb)** - [Perceptron Model 1957](IRIS/01-Simple-Perceptron.ipynb)  
Example of use of a Perceptron, with sklearn and IRIS dataset of 1936 !

### Basic regression using DN
- **[BHPD1](BHPD/01-DNN-Regression.ipynb)** - [Regression with a Dense Network (DNN)](BHPD/01-DNN-Regression.ipynb)  
Simple example of a regression with the dataset Boston Housing Prices Dataset (BHPD)
- **[BHPD2](BHPD/02-DNN-Regression-Premium.ipynb)** - [Regression with a Dense Network (DNN) - Advanced code](BHPD/02-DNN-Regression-Premium.ipynb)  
A more advanced implementation of the precedent example
- **[WINE1](BHPD/03-DNN-Wine-Regression.ipynb)** - [Wine quality prediction with a Dense Network (DNN)](BHPD/03-DNN-Wine-Regression.ipynb)  
Another example of regression, with a wine quality prediction!

### Basic classification using a DN
- **[MNIST1](MNIST/01-DNN-MNIST.ipynb)** - [Simple classification with DNN](MNIST/01-DNN-MNIST.ipynb)  
An example of classification using a dense neural network for the famous MNIST dataset
- **[MNIST2](MNIST/02-CNN-MNIST.ipynb)** - [Simple classification with CNN](MNIST/02-CNN-MNIST.ipynb)  
An example of classification using a convolutional neural network for the famous MNIST dataset

### Images classification with Convolutional Neural Networks (CNN)
- **[GTSRB1](GTSRB/01-Preparation-of-data.ipynb)** - [Dataset analysis and preparation](GTSRB/01-Preparation-of-data.ipynb)  
Episode 1 : Analysis of the GTSRB dataset and creation of an enhanced dataset
- **[GTSRB2](GTSRB/02-First-convolutions.ipynb)** - [First convolutions](GTSRB/02-First-convolutions.ipynb)  
Episode 2 : First convolutions and first classification of our traffic signs
- **[GTSRB3](GTSRB/03-Tracking-and-visualizing.ipynb)** - [Training monitoring](GTSRB/03-Tracking-and-visualizing.ipynb)  
Episode 3 : Monitoring, analysis and check points during a training session
- **[GTSRB4](GTSRB/04-Data-augmentation.ipynb)** - [Data augmentation ](GTSRB/04-Data-augmentation.ipynb)  
Episode 4 : Adding data by data augmentation when we lack it, to improve our results
- **[GTSRB5](GTSRB/05-Full-convolutions.ipynb)** - [Full convolutions](GTSRB/05-Full-convolutions.ipynb)  
Episode 5 : A lot of models, a lot of datasets and a lot of results.
- **[GTSRB6](GTSRB/06-Notebook-as-a-batch.ipynb)** - [Full convolutions as a batch](GTSRB/06-Notebook-as-a-batch.ipynb)  
Episode 6 : To compute bigger, use your notebook in batch mode
- **[GTSRB7](GTSRB/07-Show-report.ipynb)** - [Batch reports](GTSRB/07-Show-report.ipynb)  
Episode 7 : Displaying our jobs report, and the winner is...
- **[GTSRB10](GTSRB/batch_oar.sh)** - [OAR batch script submission](GTSRB/batch_oar.sh)  
Bash script for an OAR batch submission of an ipython code
- **[GTSRB11](GTSRB/batch_slurm.sh)** - [SLURM batch script](GTSRB/batch_slurm.sh)  
Bash script for a Slurm batch submission of an ipython code

### Sentiment analysis with word embeddin
- **[IMDB1](IMDB/01-One-hot-encoding.ipynb)** - [Sentiment analysis with hot-one encoding](IMDB/01-One-hot-encoding.ipynb)  
A basic example of sentiment analysis with sparse encoding, using a dataset from Internet Movie Database (IMDB)
- **[IMDB2](IMDB/02-Keras-embedding.ipynb)** - [Sentiment analysis with text embedding](IMDB/02-Keras-embedding.ipynb)  
A very classical example of word embedding with a dataset from Internet Movie Database (IMDB)
- **[IMDB3](IMDB/03-Prediction.ipynb)** - [Reload and reuse a saved model](IMDB/03-Prediction.ipynb)  
Retrieving a saved model to perform a sentiment analysis (movie review)
- **[IMDB4](IMDB/04-Show-vectors.ipynb)** - [Reload embedded vectors](IMDB/04-Show-vectors.ipynb)  
Retrieving embedded vectors from our trained model
- **[IMDB5](IMDB/05-LSTM-Keras.ipynb)** - [Sentiment analysis with a RNN network](IMDB/05-LSTM-Keras.ipynb)  
Still the same problem, but with a network combining embedding and RNN

### Time series with Recurrent Neural Network (RNN)
- **[LADYB1](SYNOP/LADYB1-Ladybug.ipynb)** - [Prediction of a 2D trajectory via RNN](SYNOP/LADYB1-Ladybug.ipynb)  
Artificial dataset generation and prediction attempt via a recurrent network
- **[SYNOP1](SYNOP/SYNOP1-Preparation-of-data.ipynb)** - [Preparation of data](SYNOP/SYNOP1-Preparation-of-data.ipynb)  
Episode 1 : Data analysis and preparation of a usuable meteorological dataset (SYNOP)
- **[SYNOP2](SYNOP/SYNOP2-First-predictions.ipynb)** - [First predictions at 3h](SYNOP/SYNOP2-First-predictions.ipynb)  
Episode 2 : RNN training session for weather prediction attempt at 3h
- **[SYNOP3](SYNOP/SYNOP3-12h-predictions.ipynb)** - [12h predictions](SYNOP/SYNOP3-12h-predictions.ipynb)  
Episode 3: Attempt to predict in a more longer term 

### Sentiment analysis with transformer
- **[TRANS1](Transformers/01-Distilbert.ipynb)** - [IMDB, Sentiment analysis with Transformers ](Transformers/01-Distilbert.ipynb)  
Using a Tranformer to perform a sentiment analysis (IMDB) - Jean Zay version
- **[TRANS2](Transformers/02-distilbert_colab.ipynb)** - [IMDB, Sentiment analysis with Transformers ](Transformers/02-distilbert_colab.ipynb)  
Using a Tranformer to perform a sentiment analysis (IMDB) - Colab version

### Unsupervised learning with an autoencoder neural network (AE)
- **[AE1](AE/01-Prepare-MNIST-dataset.ipynb)** - [Prepare a noisy MNIST dataset](AE/01-Prepare-MNIST-dataset.ipynb)  
Episode 1: Preparation of a noisy MNIST dataset
- **[AE2](AE/02-AE-with-MNIST.ipynb)** - [Building and training an AE denoiser model](AE/02-AE-with-MNIST.ipynb)  
Episode 1 : Construction of a denoising autoencoder and training of it with a noisy MNIST dataset.
- **[AE3](AE/03-AE-with-MNIST-post.ipynb)** - [Playing with our denoiser model](AE/03-AE-with-MNIST-post.ipynb)  
Episode 2 : Using the previously trained autoencoder to denoise data
- **[AE4](AE/04-ExtAE-with-MNIST.ipynb)** - [Denoiser and classifier model](AE/04-ExtAE-with-MNIST.ipynb)  
Episode 4 : Construction of a denoiser and classifier model
- **[AE5](AE/05-ExtAE-with-MNIST.ipynb)** - [Advanced denoiser and classifier model](AE/05-ExtAE-with-MNIST.ipynb)  
Episode 5 : Construction of an advanced denoiser and classifier model

### Generative network with Variational Autoencoder (VAE)
- **[VAE1](VAE/01-VAE-with-MNIST.ipynb)** - [First VAE, using functional API (MNIST dataset)](VAE/01-VAE-with-MNIST.ipynb)  
Construction and training of a VAE, using functional APPI, with a latent space of small dimension.
- **[VAE2](VAE/02-VAE-with-MNIST.ipynb)** - [VAE, using a custom model class  (MNIST dataset)](VAE/02-VAE-with-MNIST.ipynb)  
Construction and training of a VAE, using model subclass, with a latent space of small dimension.
- **[VAE3](VAE/03-VAE-with-MNIST-post.ipynb)** - [Analysis of the VAE's latent space of MNIST dataset](VAE/03-VAE-with-MNIST-post.ipynb)  
Visualization and analysis of the VAE's latent space of the dataset MNIST

### Generative Adversarial Networks (GANs)
- **[SHEEP1](DCGAN/01-DCGAN-Draw-me-a-sheep.ipynb)** - [A first DCGAN to Draw a Sheep](DCGAN/01-DCGAN-Draw-me-a-sheep.ipynb)  
Episode 1 : Draw me a sheep, revisited with a DCGAN
- **[SHEEP2](DCGAN/02-WGANGP-Draw-me-a-sheep.ipynb)** - [A WGAN-GP to Draw a Sheep](DCGAN/02-WGANGP-Draw-me-a-sheep.ipynb)  
Episode 2 : Draw me a sheep, revisited with a WGAN-GP

### Diffusion Model (DDPM)
- **[DDPM1](DDPM/01-ddpm.ipynb)** - [Fashion MNIST Generation with DDPM](DDPM/01-ddpm.ipynb)  
Diffusion Model example, to generate Fashion MNIST images.
- **[DDPM2](DDPM/model.py)** - [DDPM Python classes](DDPM/model.py)  
Python classes used by DDMP Example

### Training optimization
- **[OPT1](Optimization/01-Apprentissages-rapides-et-Optimisations.ipynb)** - [Training setup optimization](Optimization/01-Apprentissages-rapides-et-Optimisations.ipynb)  
The goal of this notebook is to go through a typical deep learning model training

### Deep Reinforcement Learning (DRL)
- **[DRL1](DRL/FIDLE_DQNfromScratch.ipynb)** - [Solving CartPole with DQN](DRL/FIDLE_DQNfromScratch.ipynb)  
Using a a Deep Q-Network to play CartPole - an inverted pendulum problem (PyTorch)
- **[DRL2](DRL/FIDLE_rl_baselines_zoo.ipynb)** - [RL Baselines3 Zoo: Training in Colab](DRL/FIDLE_rl_baselines_zoo.ipynb)  
Demo of Stable baseline3 with Colab

### Miscellaneous
- **[ACTF1](Misc/Activation-Functions.ipynb)** - [Activation functions](Misc/Activation-Functions.ipynb)  
Some activation functions, with their derivatives.
- **[NP1](Misc/Numpy.ipynb)** - [A short introduction to Numpy](Misc/Numpy.ipynb)  
Numpy is an essential tool for the Scientific Python.
- **[SCRATCH1](Misc/Scratchbook.ipynb)** - [Scratchbook](Misc/Scratchbook.ipynb)  
A scratchbook for small examples
- **[TSB1](Misc/Using-Tensorboard.ipynb)** - [Tensorboard with/from Jupyter ](Misc/Using-Tensorboard.ipynb)  
4 ways to use Tensorboard from the Jupyter environment
- **[PANDAS1](Misc/Using-pandas.ipynb)** - [Quelques exemples avec Pandas](Misc/Using-pandas.ipynb)  
pandas is another essential tool for the Scientific Python.
<!-- TOC_END -->


## Installation

Have a look about **[How to get and install](https://fidle.cnrs.fr/installation)** these notebooks and datasets.

## Licence

[<img width="100px" src="fidle/img/00-fidle-CC BY-NC-SA.svg"></img>](https://creativecommons.org/licenses/by-nc-sa/4.0/)  
\[en\] Attribution - NonCommercial - ShareAlike 4.0 International (CC BY-NC-SA 4.0)  
\[Fr\] Attribution - Pas d’Utilisation Commerciale - Partage dans les Mêmes Conditions 4.0 International  
See [License](https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).  
See [Disclaimer](https://creativecommons.org/licenses/by-nc-sa/4.0/#).  


----
[<img width="80px" src="fidle/img/00-Fidle-logo-01.svg"></img>](#top)
