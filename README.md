# Leaf classification using ANNs
This repository tries to solve the classification task in the kaggle's competition **[leaf_classification](https://www.kaggle.com/c/leaf-classification/overview)** using artificial neural networks. This task consist in classify leaf samples in 99 differents species. There are images and features data for each leaf sample.

### Guideline
You can run each notebook in `Google Colab`. If you are in a local environment  you have to install the `requirements.txt` packages and run the notebook `01 - Get data.ipynb`.

### Content

* **Notebook 01:** Run if you are in a local environment  to download and prepare the datasets necessary in the rest of notebooks.

* **Notebook 02:** A data exploration of leaf_classification dataset. In this notebook you can explore the distributions and correlation of the data.

* **Notebook 03:** A set of functions to split the data between train and test sets keeping a class balance in each set.

* **Notebook 04:** ML basic model tested in the leaf_classification problem. The performance of this model can be compared with the deep learning approaches.

* **Notebook 05:** A one-dimensional convolutional neural network model is used to classify leaves by features you should get a good precision result after training the model with the data and contracting it with the test data.

* **Notebook 06:** A three-layer convolutional neural network model is used to try to classify the leaves by means of flattened images, you get several graphs one of the losses with respect to the steps and also other of the weights in layers 1 and 3 where you notice that the weights you are gained information, but without satisfactory results at the end.

* **Notebook 07:** A convolutional neural network model with several layers is used, starting with a convolutional layer of two dimensions, then with several dense layers, in addition it was also tested with batch normalization and Dropout layers to try to classify the leaves by means of binary images more quickly and efficiently, in the end you get the accuracy and confusion matrices but without very good results at the end.

* **Notebook 08-09:** Multimodal neural network tested in the leaf_classification problem. The images and features data are used to train and test a multimodal neural network. The multimodal network is build with Denses and Convolutional layers.

* **Notebook 10-11:** Generative adversarial networks tested in the generation of leaf samples. A GAN is build with convolutional layers to generate new leaf samples.

### Conclusions

* Baseline is the best way to solve this classification problem.
* Images in the deep learning models apply noise in models training.
* Use center or resize images is not relevant for the models.

### Authors
* Andrés F. Pérez
* Daniel C. Patiño 


