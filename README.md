# GANs

## [DCGAN](/DCGAN)
DCGAN implementation using Tensorflow + Keras. <br/>
Original paper: https://arxiv.org/pdf/1511.06434 <br/>
### Faces dataset results after 100 epochs. <br/>
This dataset consisted of people sitting in front of a book shelf. The GAN did a good job of getting the bookshelf, books, shirt, and general structure of the person; however, it struggled in generating a face. I think with another 100 epochs of training the generator will learn how to create faces better. <br/>o
![](DCGAN/FACES/results.png) <br/>
### CIFAR10 results after 300 epochs <br/>
The GAN was able to get the general style of this dataset. The results could definitely be improved with more hyperparameter tuning; however, other GAN architectures such as Conditional GAN have been proven to produce better results on this dataset. <br/>
![](DCGAN/CIFAR10/results.png) <br/>
### MNIST results <br/>
![](DCGAN/MNIST/results/download(1).png)
![](DCGAN/MNIST/results/download(2).png)
![](DCGAN/MNIST/results/download(3).png) <br/>
![](DCGAN/MNIST/results/download(4).png)
![](DCGAN/MNIST/results/download(5).png)
![](DCGAN/MNIST/results/download(6).png) <br/>
