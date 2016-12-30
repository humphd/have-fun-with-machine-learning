# Have Fun with Machine Learning: A Guide for Beginners

##Preface

This is a hands-on guide to machine learning for programmers with
*no background* in AI. Our goal will be to write a program that can
predict, with a high degree of certainty, whether the images in
[data/untrained-samples](data/untrained-samples) are of **dolphins**
or **seahorses** using only the images themselves.

![A dolphin](data/untrained-samples/dolphin1.jpg?raw=true "Dolphin")
![A seahorse](data/untrained-samples/seahorse1.jpg?raw=true "Seahorse")

To do that we’re going to train and use a [Convolutional Neural Network (CNN)](https://en.wikipedia.org/wiki/Convolutional_neural_network).
We’re going to approach this from the point of view of a practitioner vs.
from first principles. There is so much excitement about AI right now,
but much of what’s being written feels a bit like being taught to do
tricks on your bike by a physics professor at a chalkboard instead
of your friends in the park.

If you’re like me, you’re never going to be be good enough at the underlying
math and theory to advance the state of AI.  But using a neural network doesn’t
require a PhD, and you don’t need to be a data scientist at Google or Uber.
It turns out that you don’t need to be the person who makes the next breakthrough
in AI in order to use what exists today.  What we have now is already breathtaking,
and highly usable.  Play with this stuff like you would any other open source technology.

Speaking of open source, I’ve decided to write this on Github vs. as a blog post
because I’m sure that some of what I’ve written below is misleading, naive, or
just plain wrong.  I’m still learning myself, and I’ve found the lack of solid
beginner documentation an obstacle.  If you see me making a mistake or missing
important details, please send a pull request. 

With all of that out the way, let me show you how to do some tricks on your bike!

##Overview

Here’s what we’re going to explore:

* Setup and use existing, open source machine learning technologies, specifically [Caffe](http://caffe.berkeleyvision.org/) and [DIGITS](https://developer.nvidia.com/digits)
* Create a dataset of images
* Train a neural network from scratch
* Test our neural network on images it has never seen before
* Improve our neural network’s accuracy by fine tuning existing neural networks
* Deploy and use our neural network in a simple Python script

This guide won’t teach you how neural networks are designed, cover much theory,
or use a single mathematical expression.  I don’t pretend to understand most of
what I’m going to show you.  Instead, we’re going to use existing things in
interesting ways to solve a hard problem.

> Q: I know you said we won’t talk about the theory of neural networks, but I’m
> feeling like I’d at least like an overview before we get going.  Where should I start?

There are literally hundreds of introductions to this, from short posts to full
online courses.  If you want a good intro to what we’re going to be doing that
won’t get overly mathmatical, I’d recommend [this video](https://www.youtube.com/watch?v=FmpDIaiMIeA)
by [Brandon Rohrer](https://www.youtube.com/channel/UCsBKTrp45lTfHa_p49I2AEQ).
If you’d rather have a bit more theory, I’d recommend [this online book](http://neuralnetworksanddeeplearning.com/chap1.html)
by [Michael Nielsen](http://michaelnielsen.org/).

##Setup

###Installing Caffe

First, we’re going to be using the [Caffe deep learning framework](http://caffe.berkeleyvision.org/)
from the Berkely Vision and Learning Center (BSD licensed).

> Q: “Wait a minute, why Caffe? Why not use something like TensorFlow,
> which everyone is talking about these days…”  

There are a lot of great choices available, and you should look at all the
options.  [TensorFlow](https://www.tensorflow.org/) is great, and you should
play with it.  However, I’m using Caffe for a number of reasons:

* It’s tailormade for computer vision problems
* It has support for C++, Python, (with [node.js support](https://github.com/silklabs/node-caffe))
* It’s fast and stable

But the **number one reason** I’m using Caffe is that you don’t need to
write any code to work with it.  You can do everything declaratively
(Caffe uses structured text files to define the network architecture) and using
command-line tools.  Also, you can use some nice front-ends for Caffe to make
training and validating your network a lot easier.  We’ll be using
[nVidia’s DIGITS](https://developer.nvidia.com/digits) tool below for just this purpose.

Caffe can be a bit of work to get installed.  There are [installation instructions](http://caffe.berkeleyvision.org/installation.html)
for various platforms, including some prebuilt Docker or AWS configurations.

On a Mac it can be frustrating to get working, with version issues halting
your progress at various steps in the build.  It took me a couple of days
of trial and error.  There are a dozen guides I followed, each with slightly
different problems.  In the end I found [this one](https://gist.github.com/doctorpangloss/f8463bddce2a91b949639522ea1dcbe4) to be the closest:

I’d also recommend [this post](https://eddiesmo.wordpress.com/2016/12/20/how-to-set-up-caffe-environment-and-pycaffe-on-os-x-10-12-sierra/),
which is quite recent and links to many of the same discussions I saw.

Getting Caffe installed is by far the hardest thing we'll do, which is pretty
neat, since you’d assume the AI aspects would be harder!  Don’t give up if you have
issues, it’s worth the pain.  If I was doing this again, I’d probably use an Ubuntu VM
instead of trying to do it on Mac directly.  There's also a [Caffe Users](https://groups.google.com/forum/#!forum/caffe-users) group, if you need answers.

> Q: “Do I need powerful hardware to train a neural network? What if I don’t have
> access to fancy GPUs?”

It’s true, deep neural networks require a lot of computing power and energy to
train...if you’re training them from scratch and using massive datasets.
We aren’t going to do that.  The secret is to use a pretrained network that someone
else has already invested hundreds of hours of compute time training, and then to fine
tune it to your particular dataset.  We’ll look at how to do this below, but suffice
it to say that everything I’m going to show you, I’m doing on a year old MacBook
Pro without a fancy GPU.

As an aside, because I have an integrated Intel graphics card vs. an nVidia GPU,
I decided to use the [OpenCL Caffe branch](https://github.com/BVLC/caffe/tree/opencl),
and it’s worked great on my laptop.

When you’re done installing Caffe, you should have, or be able to do all of the following:

* A directory that contains your built caffe.  If you did this in the standard way,
there will be a `build/` dir which contains everything you need to run caffe,
the Python bindings, etc.  The parent dir that contains `build/` will be your
`CAFFE_ROOT` (we’ll need this later).
* Running `make test && make runtest` should pass
* After installing all the Python deps (doing `for req in $(cat requirements.txt); do pip install $req; done` in `python/`),
running `make pycaffe && make pytest` should pass
* You should also run `make distribute` in order to create a distributable version of caffe with all necessary headers, binaries, etc. in `distribute/`.

On my machine, with Caffe fully built, I’ve got the following basic layout in my CAFFE_ROOT dir:

```
caffe/
    build/
        python/
        lib/
        tools/
            caffe ← this is our main binary 
    distribute/
        python/
        lib/
        include/
        bin/
        proto/
```

At this point, we have everything we need to train, test, and program with neural
networks.  In the next section we’ll add a user-friendly, web-based front end to
Caffe called DIGITS, which will make training and testing our networks much easier.

###Installing DIGITS

nVidia’s [Deep Learning GPU Training System, or DIGITS](https://github.com/NVIDIA/DIGITS),
is BSD-licensed Python web app for training neural networks.  While it’s
possible to do everything DIGITS does in Caffe at the command-line, or with code,
using DIGITS makes it a lot easier to get started.  I also found it more fun, due
to the great visualizations, real-time charts, and other graphical features.
Since you’re experimenting and trying to learn, I highly recommend beginning with DIGITS.

There are quite a few good docs at https://github.com/NVIDIA/DIGITS/tree/master/docs,
including a few [Installation](https://github.com/NVIDIA/DIGITS/blob/master/docs/BuildDigits.md),
[Configuration](https://github.com/NVIDIA/DIGITS/blob/master/docs/Configuration.md),
and [Getting Started](https://github.com/NVIDIA/DIGITS/blob/master/docs/GettingStarted.md)
pages.  I’d recommend reading through everything there before you continue, as I’m not
an expert on everything you can do with DIGITS.

There are various ways to install and run DIGITS, from Docker to pre-baked packages
on Linux, or you can build it from source. I’m on a Mac, so I built it from source.
Because it’s just a bunch of Python scripts, it was fairly painless to get working.
The one thing you need to do is tell DIGITS where your `CAFFE_ROOT` is by setting
an environment variable before starting the server:

```bash
export CAFFE_ROOT=/path/to/caffe
./digits-devserver
```

NOTE: on Mac I had issues with the server scripts assuming my Python binary was
called `python2`, where I only have `python2.7`.  You can symlink it in `/usr/bin`
or modify the DIGITS startup script(s) to use the proper binary on your system.

Once the server is started, you can do everything else via your web browser at http://localhost:5000, which is what I'll do below.

##Training a Neural Network

Training a neural network involves a few steps:

1. Assemble and prepare a dataset of categorized images
2. Define the network’s architecture
3. Train and Validate this network using the prepared dataset

We’re going to do this 3 different ways, in order to show the difference
between starting from scratch and using a pretrained network, and also to show
how to work with two popular pretrained networks (AlexNet, GoogLeNet) that are
commonly used with Caffe and DIGITs.

For our training attempts, we’ll use a small dataset of Dolphins and Seahorses.
I’ve put the images I used in [data/dolphins-and-seahorses](data/dolphins-and-seahorses).
You need at least 2 categories, but could have many more (some of the networks
we’ll use were trained on 1000+ image categories).  Our goal is to be able to
give an image to our network and have it tell us whether it’s a Dolphin or a Seahorse.

###Prepare the Dataset

The easiest way to begin is to divide your images into a categorized directory layout:

```
dolphins-and-seahorses/
    dolphin/
        image_0001.jpg
        image_0002.jpg
        image_0003.jpg
        ...
    seahorse/
        image_0001.jpg
        image_0002.jpg
        image_0003.jpg
        ...
```

Here each directory is a category we want to classify, and each image within
that category dir an example we’ll use for training and validation. 

> Q: “Do my images have to be the same size?  What about the filenames, do they matter?”

No to both. The images sizes will be normalized before we feed them into
the network.  We’ll eventually want colour images of 256 x 256 pixels, but
DIGITS will crop or squash (we'll squash) our images automatically in a moment.
The filenames are irrelevant--it’s only important which category they are contained
within.

> Q: “Can I do more complex segmentation of my categories?”

Yes. See https://github.com/NVIDIA/DIGITS/blob/digits-4.0/docs/ImageFolderFormat.md.

We want to use these images on disk to create a **New Dataset**, and specifically,
a **Classification Dataset**.

![Create New Dataset](images/create-new-dataset.png?raw=true "Create New Dataset")

We’ll use the defaults DIGITS gives us, and point **Training Images** at the path
to our [data/dolphins-and-seahorses](data/dolphins-and-seahorses) folder.
DIGITS will use the categories (`dolphin` and `seahorse`) to create a database
of squashed, 256 x 256 Training (75%) and Testing (25%) images.

Give your Dataset a name,`dolphins-and-seahorses`, and click **Create**.

![New Image Classification Dataset](images/new-image-classification-dataset.png?raw=true "New Image Classification Dataset")

This will create our dataset, which took only 4s on my laptop.  In the end I
have 92 Training images (49 dolphin, 43 seahorse) in 2 categories, with 30
Validation images (16 dolphin, 14 seahorse).  It’s a really small dataset, but perfect
for our experimentation and learning purposes, because it won’t take forever to train
and validate a network that uses it. 

You can **Explore the db** if you want to see the images after they have been squashed. 

![Explore the db](images/explore-dataset.png?raw=true "Explore the db")

### Training: Attempt 1 from Scratch

Back in the DIGITS Home screen, we need to create a new **Classification Model**:

![Create Classification Model](images/create-classification-model.png?raw=true "Create Classification Model")

We’ll start by training a model that uses our `dolphins-and-seahorses` dataset,
and the default settings DIGITS provides.  For our first network, we’ll choose to
use one of the standard network architectures, [AlexNet (pdf)](http://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf). [AlexNet’s design](http://vision.stanford.edu/teaching/cs231b_spring1415/slides/alexnet_tugce_kyunghee.pdf)
won a major computer vision competition called ImageNet in 2012.  The competition
required categorizing 1000+ image categories across 1.2 million images.
 
Caffe uses structured text files to define network architectures.  These text files
are based on [Google’s Protocol Buffers](https://developers.google.com/protocol-buffers/).
You can read the [full schema](https://github.com/BVLC/caffe/blob/master/src/caffe/proto/caffe.proto) Caffe uses.
For the most part we’re not going to work with this, but it’s good to be aware of their
existence, since we’ll have to modify them in later steps.  The AlexNet prototxt file
looks like this, for example: https://github.com/BVLC/caffe/blob/master/models/bvlc_alexnet/train_val.prototxt. 

We’ll train our network for **30 epochs**, which means that it will learn (with our
training images) then test itself (using our validation images), and adjust the
network’s weights depending on how well it’s doing, and repeat this process 30 times.
Each time it completes a cycle we’ll get info about **Accuracy** (0% to 100%,
where higher is better) and what our **Loss** is (the sum of all the mistakes that were
made, where lower is better).  Ideally we want a network that is able to predict with
high accuracy, and with few errors (small loss).

Initially, our network’s accuracy is a bit below 50%.  This makes sense, because at first it’s
just “guessing” between two categories using randomly assigned weights.  Over time
it’s able to achieve 87.5% accuracy, with a loss of 0.37.  The entire 30 epoch run
took me just under 6 minutes.

![Model Attempt 1](images/model-attempt1.png?raw=true "Model Attempt 1")

We can test our model using an image we upload or a URL to an image on the web.
Let’s test it on a few examples that weren’t in our training/validation dataset:

![Model 1 Classify 1](images/model-attempt1-classify1.png?raw=true "Model 1 Classify 1")

![Model 1 Classify 2](images/model-attempt1-classify2.png?raw=true "Model 1 Classify 2")

It almost seems perfect, until we try another:

![Model 1 Classify 3](images/model-attempt1-classify3.png?raw=true "Model 1 Classify 3")

Here it falls down completely, and confuses a seahorse for a dolphin, and worse,
does so with a high degree of confidence.

The reality is that our dataset is too small to be useful for training a really good
neural network.  We really need 10s or 100s of thousands of images, and with that, a
lot of computing power to process everything.

### Training: Attempt 2, Fine Tuning AlexNet

####How Fine Tuning works

Designing a neural network from scratch, collecting data sufficient to train
it (e.g., millions of images), and accessing GPUs for weeks to complete the
training is beyond the reach of most of us.  To make it practical for smaller amounts
of data to be used, we employ a technique called **Transfer Learning**, or **Fine Tuning**.
Fine tuning takes advantage of the layout of deep neural networks, and uses
pretrained networks to do the hard work of initial object detection.

Imagine using a neural network to be like looking at something far away with a 
pair of binoculars.  You first put the binoculars to your eyes, and everything is
blurry.  As you adjust the focus, you start to see colours, lines, shapes, and eventually
you are able to pick out the shape of a bird, then with some more adjustment you can
identify the species of bird.

In a multi-layered network, the initial layers extract features (e.g., edges), with
later layers using these features to detect shapes (e.g., a wheel, an eye), which are
then feed into final classification layers that detect items based on accumulated 
characteristics from previous layers (e.g., a cat vs. a dog).  A network has to be 
able to go from pixels to circles to eyes to two eyes placed in a particular orientation, 
and so on up to being able to finally conclude that an image depicts a cat.

What we’d like to do is to specialize an existing, pretrained network for classifying 
a new set of image classes instead of the ones on which it was initially trained. Because
the network already knows how to “see” features in images, we’d like to retrain 
it to “see” our particular image types.  We don’t need to start from scratch with the 
majority of the layers--we want to transfer the learning already done in these layers 
to our new classification task.  Unlike our previous attempt, which used random weights, 
we’ll use the existing weights of the final network in our training.  However, we’ll 
throw away the final classification layer(s) and retrain the network with *our* image 
dataset, fine tuning it to our image classes.

For this to work, we need a pretrained network that is similar enough to our own data
that the learned weights will be useful.  Luckily, the networks we’ll use below were 
trained on millions of natural images from [ImageNet](http://image-net.org/), which 
is useful across a broad range of classification tasks.

This technique has been used to do interesting things like screening for eye diseases 
from medical imagery, identifying plankton species from microscopic images collected at 
sea, to categorizing the artistic style of Flickr images.

Doing this perfectly, like all of machine learning, requires you to understand the
data and network architecture--you have to be careful with overfitting of the data, 
might need to fix some of the layers, might need to insert new layers, etc.  
However, my experience is that it “Just Works” much of the time, and it’s worth 
you simply doing an experiment to see what you can achieve using our naive approach.

####Uploading Pretrained Networks

In our first attempt, we used AlexNet’s architecture, but started with random
weights in the network’s layers.  What we’d like to do is download and use a
version of AlexNet that has already been trained on a massive dataset.

Thankfully we can do exactly this.  A snapshot of AlexNet is available for download: https://github.com/BVLC/caffe/tree/master/models/bvlc_alexnet.
We need the binary `.caffemodel` file, which is what contains the weights, and it’s
available for download at http://dl.caffe.berkeleyvision.org/bvlc_alexnet.caffemodel.

While you’re downloading pretrained models, let’s get one more at the same time.
In 2014, Google won the same ImageNet competition with [GoogLeNet](https://research.google.com/pubs/pub43022.html) (codenamed Inception):
a 22-layer neural network. A snapshot of GoogLeNet is available for download
as well, see https://github.com/BVLC/caffe/tree/master/models/bvlc_googlenet.
Again, we’ll need the `.caffemodel` file with all the pretrained weights,
which is available for download at http://dl.caffe.berkeleyvision.org/bvlc_googlenet.caffemodel. 

With these `.caffemodel` files in hand, we can upload them into DIGITs.  Go to
the **Pretrained Models** tab in DIGITs home page and choose **Upload Pretrained Model**:

![Load Pretrained Model](images/load-pretrained-model.png?raw=true "Load Pretrained Model")

For both of these pretrained models, we can use the defaults DIGITs provides
(i.e., colour, squashed images of 256 x 256).  We just need to provide the 
`Weights (**.caffemodel)` and `Model Definition (original.prototxt)`.
Click each of those buttons to select a file.

For the model definitions we can use https://github.com/BVLC/caffe/blob/master/models/bvlc_googlenet/train_val.prototxt
for GoogLeNet and https://github.com/BVLC/caffe/blob/master/models/bvlc_alexnet/train_val.prototxt
for AlexNet.  We aren’t going to use the classification labels of these networks,
so we’ll skip adding a `labels.txt` file:
 
![Upload Pretrained Model](images/upload-pretrained-model.png?raw=true "Upload Pretrained Model")

Repeat this process for both AlexNet and GoogLeNet, as we’ll use them both in the coming steps.

####Fine Tuning AlexNet for Dolphins and Seahorses

Training a network using a pretrained Caffe Model is similar to starting from scratch,
though we have to make a few adjustments.  First, we’ll adjust the **Base Learning Rate**
to 0.001 from 0.01, since we don’t need to make such large jumps (i.e., we’re fine tuning).
We’ll also use a **Pretrained Network**, and **Customize** it.

![New Image Classification](images/new-image-classification-model-attempt2.png?raw=true "New Image Classification")

In the pretrained model’s definition (i.e., prototext), we need to rename all
references to the final **Fully Connected Layer** (where the end result classifications
happen).  We do this because we want the model to re-learn new categories from
our dataset vs. its original training data (i.e., we want to throw away the current
final layer).  We have to rename the last fully connected layer from “fc8” to
something else, “fc9” for example.  Finally, we also need to adjust the number
of categories from `1000` to `2`, by changing `num_output` to `2`.

Here are the changes we need to make:

```diff
@@ -332,8 +332,8 @@
 }
 layer {
-  name: "fc8"
+  name: "fc9"
   type: "InnerProduct"
   bottom: "fc7"
-  top: "fc8"
+  top: "fc9"
   param {
     lr_mult: 1
@@ -345,5 +345,5 @@
   }
   inner_product_param {
-    num_output: 1000
+    num_output: 2
     weight_filler {
       type: "gaussian"
@@ -359,5 +359,5 @@
   name: "accuracy"
   type: "Accuracy"
-  bottom: "fc8"
+  bottom: "fc9"
   bottom: "label"
   top: "accuracy"
@@ -367,5 +367,5 @@
   name: "loss"
   type: "SoftmaxWithLoss"
-  bottom: "fc8"
+  bottom: "fc9"
   bottom: "label"
   top: "loss"
@@ -375,5 +375,5 @@
   name: "softmax"
   type: "Softmax"
-  bottom: "fc8"
+  bottom: "fc9"
   top: "softmax"
   include { stage: "deploy" }
```

I’ve included the fully modified file I’m using in [src/alexnet-customized.prototxt](src/alexnet-customized.prototxt).

This time our accuracy starts at ~60% and climbs right away to 87.5%, then to 96%
and all the way up to 100%, with the Loss steadily decreasing. After 5 minutes we
end up with an accuracy of 100% and a loss of 0.0009.

![Model Attempt 2](images/model-attempt2.png?raw=true "Model Attempt 2")

Testing the same seahorse image our previous network got wrong, we see a complete
reversal: 100% seahorse.

![Model 2 Classify 1](images/model-attempt2-classify1.png?raw=true "Model 2 Classify 1")

Even a children’s drawing of a seahorse works:

![Model 2 Classify 2](images/model-attempt2-classify2.png?raw=true "Model 2 Classify 2")

The same goes for a dolphin:

![Model 2 Classify 3](images/model-attempt2-classify3.png?raw=true "Model 2 Classify 3")

Even with images that you think might be hard, like this one that has multiple dolphins
close together, and with their bodies mostly underwater, it does the right thing:

![Model 2 Classify 4](images/model-attempt2-classify4.png?raw=true "Model 2 Classify 4")

### Training: Attempt 3, Fine Tuning GoogLeNet

Like the previous AlexNet model we used for fine tuning, we can use GoogLeNet as well.
Modifying the network is a bit trickier, since you have to redefine three fully
connected layers instead of just one.

To fine tune GoogLeNet for our use case, we need to once again create a
new **Classification Model**:

![New Classification Model](images/new-image-classification-model-attempt3.png?raw=true "New Classification Model")

We rename all references to the three fully connected classification layers,
`loss1/classifier`, `loss2/classifier`, and `loss3/classifier`, and redefine
the number of categories (`num_output: 2`).  Here are the changes we need to make
in order to rename the 3 classifier layers, as well as to change from 1000 to 2 categories:

```diff
@@ -917,10 +917,10 @@
   exclude { stage: "deploy" }
 }
 layer {
-  name: "loss1/classifier"
+  name: "loss1a/classifier"
   type: "InnerProduct"
   bottom: "loss1/fc"
-  top: "loss1/classifier"
+  top: "loss1a/classifier"
   param {
     lr_mult: 1
     decay_mult: 1
@@ -930,7 +930,7 @@
     decay_mult: 0
   }
   inner_product_param {
-    num_output: 1000
+    num_output: 2
     weight_filler {
       type: "xavier"
       std: 0.0009765625
@@ -945,7 +945,7 @@
 layer {
   name: "loss1/loss"
   type: "SoftmaxWithLoss"
-  bottom: "loss1/classifier"
+  bottom: "loss1a/classifier"
   bottom: "label"
   top: "loss1/loss"
   loss_weight: 0.3
@@ -954,7 +954,7 @@
 layer {
   name: "loss1/top-1"
   type: "Accuracy"
-  bottom: "loss1/classifier"
+  bottom: "loss1a/classifier"
   bottom: "label"
   top: "loss1/accuracy"
   include { stage: "val" }
@@ -962,7 +962,7 @@
 layer {
   name: "loss1/top-5"
   type: "Accuracy"
-  bottom: "loss1/classifier"
+  bottom: "loss1a/classifier"
   bottom: "label"
   top: "loss1/accuracy-top5"
   include { stage: "val" }
@@ -1705,10 +1705,10 @@
   exclude { stage: "deploy" }
 }
 layer {
-  name: "loss2/classifier"
+  name: "loss2a/classifier"
   type: "InnerProduct"
   bottom: "loss2/fc"
-  top: "loss2/classifier"
+  top: "loss2a/classifier"
   param {
     lr_mult: 1
     decay_mult: 1
@@ -1718,7 +1718,7 @@
     decay_mult: 0
   }
   inner_product_param {
-    num_output: 1000
+    num_output: 2
     weight_filler {
       type: "xavier"
       std: 0.0009765625
@@ -1733,7 +1733,7 @@
 layer {
   name: "loss2/loss"
   type: "SoftmaxWithLoss"
-  bottom: "loss2/classifier"
+  bottom: "loss2a/classifier"
   bottom: "label"
   top: "loss2/loss"
   loss_weight: 0.3
@@ -1742,7 +1742,7 @@
 layer {
   name: "loss2/top-1"
   type: "Accuracy"
-  bottom: "loss2/classifier"
+  bottom: "loss2a/classifier"
   bottom: "label"
   top: "loss2/accuracy"
   include { stage: "val" }
@@ -1750,7 +1750,7 @@
 layer {
   name: "loss2/top-5"
   type: "Accuracy"
-  bottom: "loss2/classifier"
+  bottom: "loss2a/classifier"
   bottom: "label"
   top: "loss2/accuracy-top5"
   include { stage: "val" }
@@ -2435,10 +2435,10 @@
   }
 }
 layer {
-  name: "loss3/classifier"
+  name: "loss3a/classifier"
   type: "InnerProduct"
   bottom: "pool5/7x7_s1"
-  top: "loss3/classifier"
+  top: "loss3a/classifier"
   param {
     lr_mult: 1
     decay_mult: 1
@@ -2448,7 +2448,7 @@
     decay_mult: 0
   }
   inner_product_param {
-    num_output: 1000
+    num_output: 2
     weight_filler {
       type: "xavier"
     }
@@ -2461,7 +2461,7 @@
 layer {
   name: "loss3/loss"
   type: "SoftmaxWithLoss"
-  bottom: "loss3/classifier"
+  bottom: "loss3a/classifier"
   bottom: "label"
   top: "loss"
   loss_weight: 1
@@ -2470,7 +2470,7 @@
 layer {
   name: "loss3/top-1"
   type: "Accuracy"
-  bottom: "loss3/classifier"
+  bottom: "loss3a/classifier"
   bottom: "label"
   top: "accuracy"
   include { stage: "val" }
@@ -2478,7 +2478,7 @@
 layer {
   name: "loss3/top-5"
   type: "Accuracy"
-  bottom: "loss3/classifier"
+  bottom: "loss3a/classifier"
   bottom: "label"
   top: "accuracy-top5"
   include { stage: "val" }
@@ -2489,7 +2489,7 @@
 layer {
   name: "softmax"
   type: "Softmax"
-  bottom: "loss3/classifier"
+  bottom: "loss3a/classifier"
   top: "softmax"
   include { stage: "deploy" }
 }
```

I’ve put the complete file in [src/googlenet-customized.prototxt](src/googlenet-customized.prototxt).

Like we did with fine tuning AlexNet, we also reduce the learning rate by
10% from `0.01` to `0.001`.  GoogLeNet has a more complicated architecture
than AlexNet, and fine tuning it requires more time.  On my laptop, it takes
10 minutes to retrain GoogLeNet with our dataset, achieving 100% accuracy and
a loss of 0.0070:

![Model Attempt 3](images/model-attempt3.png?raw=true "Model Attempt 3")

Just as we saw with the fine tuned version of AlexNet, our modified GoogLeNet
performs amazing well--the best so far:

![Model Attempt 3 Classify 1](images/model-attempt3-classify1.png?raw=true "Model Attempt 3 Classify 1")

![Model Attempt 3 Classify 2](images/model-attempt3-classify2.png?raw=true "Model Attempt 3 Classify 2")

![Model Attempt 3 Classify 3](images/model-attempt3-classify3.png?raw=true "Model Attempt 3 Classify 3")

## Results

At the beginning we said that our goal was to write a program that used a neural network to
correctly classify all of the images in [data/untrained-samples](data/untrained-samples).
These are images of dolphins and seahorses that were never used in the training or validation
data:

### Untrained Dolphin Images

![Dolphin 1](data/untrained-samples/dolphin1.jpg?raw=true "Dolphin 1")
![Dolphin 2](data/untrained-samples/dolphin2.jpg?raw=true "Dolphin 2")
![Dolphin 3](data/untrained-samples/dolphin3.jpg?raw=true "Dolphin 3")

### Untrained Seahorse Images

![Seahorse 1](data/untrained-samples/seahorse1.jpg?raw=true "Seahorse 1")
![Seahorse 2](data/untrained-samples/seahorse2.jpg?raw=true "Seahorse 2")
![Seahorse 3](data/untrained-samples/seahorse3.jpg?raw=true "Seahorse 3")

Let's look at how each of our three attempts did with this challenge.

### Model Attempt 1: AlexNet from Scratch (3rd Place)

| Image | Dolphin | Seahorse | Result | 
|-------|---------|----------|--------|
|[data/untrained-samples/dolphin1.jpg](data/untrained-samples/dolphin1.jpg)| 71.11% | 28.89% | :expressionless: |
|[data/untrained-samples/dolphin2.jpg](data/untrained-samples/dolphin2.jpg)| 99.2% | 0.8% | :sunglasses: |
|[data/untrained-samples/dolphin3.jpg](data/untrained-samples/dolphin3.jpg)| 63.3% | 36.7% | :confused: |
|[data/untrained-samples/seahorse1.jpg](data/untrained-samples/seahorse1.jpg)| 95.04% | 4.96% | :disappointed: |
|[data/untrained-samples/seahorse2.jpg](data/untrained-samples/seahorse2.jpg)| 56.64% | 43.36 |  :confused: |
|[data/untrained-samples/seahorse3.jpg](data/untrained-samples/seahorse3.jpg)| 7.06% | 92.94% |  :grin: |

### Model Attempt 2: Fine Tuned AlexNet (2nd Place)

| Image | Dolphin | Seahorse | Result | 
|-------|---------|----------|--------|
|[data/untrained-samples/dolphin1.jpg](data/untrained-samples/dolphin1.jpg)| 99.1% | 0.09% |  :sunglasses: |
|[data/untrained-samples/dolphin2.jpg](data/untrained-samples/dolphin2.jpg)| 99.5% | 0.05% |  :sunglasses: |
|[data/untrained-samples/dolphin3.jpg](data/untrained-samples/dolphin3.jpg)| 91.48% | 8.52% |  :grin: |
|[data/untrained-samples/seahorse1.jpg](data/untrained-samples/seahorse1.jpg)| 0% | 100% |  :sunglasses: |
|[data/untrained-samples/seahorse2.jpg](data/untrained-samples/seahorse2.jpg)| 0% | 100% |  :sunglasses: |
|[data/untrained-samples/seahorse3.jpg](data/untrained-samples/seahorse3.jpg)| 0% | 100% |  :sunglasses: |

### Model Attempt 3: Fine Tuned GoogLeNet (1st Place)

| Image | Dolphin | Seahorse | Result | 
|-------|---------|----------|--------|
|[data/untrained-samples/dolphin1.jpg](data/untrained-samples/dolphin1.jpg)| 99.86% | 0.14% |  :sunglasses: |
|[data/untrained-samples/dolphin2.jpg](data/untrained-samples/dolphin2.jpg)| 100% | 0% |  :sunglasses: |
|[data/untrained-samples/dolphin3.jpg](data/untrained-samples/dolphin3.jpg)| 100% | 0% |  :sunglasses: |
|[data/untrained-samples/seahorse1.jpg](data/untrained-samples/seahorse1.jpg)| 0.5% | 99.5% |  :sunglasses: |
|[data/untrained-samples/seahorse2.jpg](data/untrained-samples/seahorse2.jpg)| 0% | 1000% |  :sunglasses: |
|[data/untrained-samples/seahorse3.jpg](data/untrained-samples/seahorse3.jpg)| 0.02% | 99.98% |  :sunglasses: |
