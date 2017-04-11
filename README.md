# Have Fun with Machine Learning: A Guide for Beginners
Also available in [Chinese (Traditional)](README_zh-tw.md).

## Preface

This is a **hands-on guide** to machine learning for programmers with *no background* in
AI. Using a neural network doesn’t require a PhD, and you don’t need to be the person who
makes the next breakthrough in AI in order to *use* what exists today.  What we have now
is already breathtaking, and highly usable.  I believe that more of us need to play with
this stuff like we would any other open source technology, instead of treating it like a
research topic.

In this guide our goal will be to write a program that uses machine learning to predict, with a
high degree of certainty, whether the images in [data/untrained-samples](data/untrained-samples)
are of **dolphins** or **seahorses** using only the images themselves, and without
having seen them before.  Here are two example images we'll use:

![A dolphin](data/untrained-samples/dolphin1.jpg?raw=true "Dolphin")
![A seahorse](data/untrained-samples/seahorse1.jpg?raw=true "Seahorse")

To do that we’re going to train and use a [Convolutional Neural Network (CNN)](https://en.wikipedia.org/wiki/Convolutional_neural_network).
We’re going to approach this from the point of view of a practitioner vs.
from first principles. There is so much excitement about AI right now,
but much of what’s being written feels like being taught to do
tricks on your bike by a physics professor at a chalkboard instead
of your friends in the park.

I’ve decided to write this on Github vs. as a blog post
because I’m sure that some of what I’ve written below is misleading, naive, or
just plain wrong.  I’m still learning myself, and I’ve found the lack of solid
beginner documentation an obstacle.  If you see me making a mistake or missing
important details, please send a pull request. 

With all of that out the way, let me show you how to do some tricks on your bike!

## Overview

Here’s what we’re going to explore:

* Setup and use existing, open source machine learning technologies, specifically [Caffe](http://caffe.berkeleyvision.org/) and [DIGITS](https://developer.nvidia.com/digits)
* Create a dataset of images
* Train a neural network from scratch
* Test our neural network on images it has never seen before
* Improve our neural network’s accuracy by fine tuning existing neural networks (AlexNet and GoogLeNet)
* Deploy and use our neural network

This guide won’t teach you how neural networks are designed, cover much theory,
or use a single mathematical expression.  I don’t pretend to understand most of
what I’m going to show you.  Instead, we’re going to use existing things in
interesting ways to solve a hard problem.

> Q: "I know you said we won’t talk about the theory of neural networks, but I’m
> feeling like I’d at least like an overview before we get going.  Where should I start?"

There are literally hundreds of introductions to this, from short posts to full
online courses.  Depending on how you like to learn, here are three options
for a good starting point:

* This fantastic [blog post](https://jalammar.github.io/visual-interactive-guide-basics-neural-networks/) by J Alammar,
which introduces the concepts of neural networks using intuitive examples.
* Similarly, [this video](https://www.youtube.com/watch?v=FmpDIaiMIeA) introduction by [Brandon Rohrer](https://www.youtube.com/channel/UCsBKTrp45lTfHa_p49I2AEQ) is a really good intro to
Convolutional Neural Networks like we'll be using
* If you’d rather have a bit more theory, I’d recommend [this online book](http://neuralnetworksanddeeplearning.com/chap1.html) by [Michael Nielsen](http://michaelnielsen.org/).

## Setup

Installing the software we'll use (Caffe and DIGITS) can be frustrating, depending on your platform
and OS version.  By far the easiest way to do it is using Docker.  Below we examine how to do it with Docker,
as well as how to do it natively.

### Option 1a: Installing Caffe Natively

First, we’re going to be using the [Caffe deep learning framework](http://caffe.berkeleyvision.org/)
from the Berkely Vision and Learning Center (BSD licensed).

> Q: “Wait a minute, why Caffe? Why not use something like TensorFlow,
> which everyone is talking about these days…”  

There are a lot of great choices available, and you should look at all the
options.  [TensorFlow](https://www.tensorflow.org/) is great, and you should
play with it.  However, I’m using Caffe for a number of reasons:

* It’s tailormade for computer vision problems
* It has support for C++, Python, (with [node.js support](https://github.com/silklabs/node-caffe) coming)
* It’s fast and stable

But the **number one reason** I’m using Caffe is that you **don’t need to write any code** to work
with it.  You can do everything declaratively (Caffe uses structured text files to define the
network architecture) and using command-line tools.  Also, you can use some nice front-ends for Caffe to make
training and validating your network a lot easier.  We’ll be using
[nVidia’s DIGITS](https://developer.nvidia.com/digits) tool below for just this purpose.

Caffe can be a bit of work to get installed.  There are [installation instructions](http://caffe.berkeleyvision.org/installation.html)
for various platforms, including some prebuilt Docker or AWS configurations.  

**NOTE:** when making my walkthrough, I used the following non-released version of Caffe from their Github repo:
https://github.com/BVLC/caffe/commit/5a201dd960840c319cefd9fa9e2a40d2c76ddd73

On a Mac it can be frustrating to get working, with version issues halting
your progress at various steps in the build.  It took me a couple of days
of trial and error.  There are a dozen guides I followed, each with slightly
different problems.  In the end I found [this one](https://gist.github.com/doctorpangloss/f8463bddce2a91b949639522ea1dcbe4) to be the closest.
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
* After installing all the Python deps (doing `pip install -r requirements.txt` in `python/`),
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

### Option 1b: Installing DIGITS Natively

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
an expert on everything you can do with DIGITS.  There's also a public [DIGITS User Group](https://groups.google.com/forum/#!forum/digits-users) if you have questions you need to ask.

There are various ways to install and run DIGITS, from Docker to pre-baked packages
on Linux, or you can build it from source. I’m on a Mac, so I built it from source.

**NOTE:** In my walkthrough I've used the following non-released version of DIGITS
from their Github repo: https://github.com/NVIDIA/DIGITS/commit/81be5131821ade454eb47352477015d7c09753d9

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

### Option 2: Caffe and DIGITS using Docker

Install [Docker](https://www.docker.com/), if not already installed, then run the following command
in order to pull and run a full Caffe + Digits container.  A few things to note:
* make sure port 8080 isn't allocated by another program. If so, change it to any other port you want.
* change `/path/to/this/repository` to the location of this cloned repo, and `/data/repo` within the container
will be bound to this directory.  This is useful for accessing the images discussed below.

```bash
docker run --name digits -d -p 8080:5000 -v /path/to/this/repository:/data/repo kaixhin/digits
```

Now that we have our container running you can open up your web browser and open `http://localhost:8080`. Everything in the repository is now in the container directory `/data/repo`.  That's it. You've now got Caffe and DIGITS working.

If you need shell access, use the following command:

```bash
docker exec -it digits /bin/bash
```

## Training a Neural Network

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

### Prepare the Dataset

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

### Training: Attempt 1, from Scratch

Back in the DIGITS Home screen, we need to create a new **Classification Model**:

![Create Classification Model](images/create-classification-model.png?raw=true "Create Classification Model")

We’ll start by training a model that uses our `dolphins-and-seahorses` dataset,
and the default settings DIGITS provides.  For our first network, we’ll choose to
use one of the standard network architectures, [AlexNet (pdf)](http://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf). [AlexNet’s design](http://vision.stanford.edu/teaching/cs231b_spring1415/slides/alexnet_tugce_kyunghee.pdf)
won a major computer vision competition called ImageNet in 2012.  The competition
required categorizing 1000+ image categories across 1.2 million images.
 
![New Classification Model 1](images/new-image-classification-model-attempt1.png?raw=true "Model 1")

Caffe uses structured text files to define network architectures.  These text files
are based on [Google’s Protocol Buffers](https://developers.google.com/protocol-buffers/).
You can read the [full schema](https://github.com/BVLC/caffe/blob/master/src/caffe/proto/caffe.proto) Caffe uses.
For the most part we’re not going to work with these, but it’s good to be aware of their
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

#### How Fine Tuning works

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
might need to fix some of the layers, might need to insert new layers, etc. However,
my experience is that it “Just Works” much of the time, and it’s worth you simply doing
an experiment to see what you can achieve using our naive approach.

#### Uploading Pretrained Networks

In our first attempt, we used AlexNet’s architecture, but started with random
weights in the network’s layers.  What we’d like to do is download and use a
version of AlexNet that has already been trained on a massive dataset.

Thankfully we can do exactly this.  A snapshot of AlexNet is available for download: https://github.com/BVLC/caffe/tree/master/models/bvlc_alexnet.
We need the binary `.caffemodel` file, which is what contains the trained weights, and it’s
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

> Q: "Are there other networks that would be good as a basis for fine tuning?"

The [Caffe Model Zoo](http://caffe.berkeleyvision.org/model_zoo.html) has quite a few other
pretrained networks that could be used, see https://github.com/BVLC/caffe/wiki/Model-Zoo.

#### Fine Tuning AlexNet for Dolphins and Seahorses

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

> Q: "What about changes to the prototext definitions of these networks?
> We changed the fully connected layer name(s), and the number of categories.
> What else could, or should be changed, and in what circumstances?"

Great question, and it's something I'm wondering, too.  For example, I know that we can
["fix" certain layers](https://github.com/BVLC/caffe/wiki/Fine-Tuning-or-Training-Certain-Layers-Exclusively)
so the weights don't change.  Doing other things involves understanding how the layers work,
which is beyond this guide, and also beyond its author at present!

Like we did with fine tuning AlexNet, we also reduce the learning rate by
10% from `0.01` to `0.001`.

> Q: "What other changes would make sense when fine tuning these networks?
> What about different numbers of epochs, batch sizes, solver types (Adam, AdaDelta, AdaGrad, etc),
> learning rates, policies (Exponential Decay, Inverse Decay, Sigmoid Decay, etc),
> step sizes, and gamma values?"

Great question, and one that I wonder about as well.  I only have a vague understanding of these
and it’s likely that there are improvements we can make if you know how to alter these
values when training.  This is something that needs better documentation.

Because GoogLeNet has a more complicated architecture than AlexNet, fine tuning it requires
more time.  On my laptop, it takes 10 minutes to retrain GoogLeNet with our dataset,
achieving 100% accuracy and a loss of 0.0070:

![Model Attempt 3](images/model-attempt3.png?raw=true "Model Attempt 3")

Just as we saw with the fine tuned version of AlexNet, our modified GoogLeNet
performs amazing well--the best so far:

![Model Attempt 3 Classify 1](images/model-attempt3-classify1.png?raw=true "Model Attempt 3 Classify 1")

![Model Attempt 3 Classify 2](images/model-attempt3-classify2.png?raw=true "Model Attempt 3 Classify 2")

![Model Attempt 3 Classify 3](images/model-attempt3-classify3.png?raw=true "Model Attempt 3 Classify 3")

## Using our Model

With our network trained and tested, it’s time to download and use it.  Each of the models
we trained in DIGITS has a **Download Model** button, as well as a way to select different
snapshots within our training run (e.g., `Epoch #30`):

![Trained Models](images/trained-models.png?raw=true “Trained Models”)

Clicking **Download Model** downloads a `tar.gz` archive containing the following files:

```
deploy.prototxt
mean.binaryproto
solver.prototxt
info.json
original.prototxt
labels.txt
snapshot_iter_90.caffemodel
train_val.prototxt
```

There’s a [nice description](https://github.com/BVLC/caffe/wiki/Using-a-Trained-Network:-Deploy) in
the Caffe documentation about how to use the model we just built.  It says:

> A network is defined by its design (.prototxt), and its weights (.caffemodel). As a network is
> being trained, the current state of that network's weights are stored in a .caffemodel. With both
> of these we can move from the train/test phase into the production phase.
>
> In its current state, the design of the network is not designed for deployment. Before we can
> release our network as a product, we often need to alter it in a few ways:
>
> 1. Remove the data layer that was used for training, as for in the case of classification we are no longer providing labels for our data.
> 2. Remove any layer that is dependent upon data labels.
> 3. Set the network up to accept data.
> 4. Have the network output the result.

DIGITS has already done the work for us, separating out the different versions of our `prototxt` files.
The files we’ll care about when using this network are:

* `deploy.prototxt` - the definition of our network, ready for accepting image input data
* `mean.binaryproto` - our model will need us to subtract the image mean from each image that it processes, and this is the mean image.
* `labels.txt` - a list of our labels (`dolphin`, `seahorse`) in case we want to print them vs. just the category number
* `snapshot_iter_90.caffemodel` - these are the trained weights for our network

We can use these files in a number of ways to classify new images.  For example, in our
`CAFFE_ROOT` we can use `build/examples/cpp_classification/classification.bin` to classify one image:

```bash
$ cd $CAFFE_ROOT/build/examples/cpp_classification
$ ./classification.bin deploy.prototxt snapshot_iter_90.caffemodel mean.binaryproto labels.txt dolphin1.jpg
```

This will spit out a bunch of debug text, followed by the predictions for each of our two categories:

```
0.9997 - “dolphin”
0.0003 - “seahorse”
```

You can read the [complete C++ source](https://github.com/BVLC/caffe/tree/master/examples/cpp_classification)
for this in the [Caffe examples](https://github.com/BVLC/caffe/tree/master/examples).

For a classification version that uses the Python interface, DIGITS includes a [nice example](https://github.com/NVIDIA/DIGITS/tree/master/examples/classification).  There's also a fairly
[well documented Python walkthrough](https://github.com/BVLC/caffe/blob/master/examples/00-classification.ipynb) in the Caffe examples.

### Python example

Let's write a program that uses our fine-tuned GoogLeNet model to classify the untrained images
we have in [data/untrained-samples](data/untrained-samples).  I've cobbled this together based on
the examples above, as well as the `caffe` [Python module's source](https://github.com/BVLC/caffe/tree/master/python),
which you should prefer to anything I'm about to say.

A full version of what I'm going to discuss is available in [src/classify-samples.py](src/classify-samples.py).
Let's begin!

First, we'll need the [NumPy](http://www.numpy.org/) module.  In a moment we'll be using [NumPy](http://www.numpy.org/)
to work with [`ndarray`s](https://docs.scipy.org/doc/numpy/reference/generated/numpy.ndarray.html), which Caffe uses a lot.
If you haven't used them before, as I had not, you'd do well to begin by reading this
[Quickstart tutorial](https://docs.scipy.org/doc/numpy-dev/user/quickstart.html).

Second, we'll need to load the `caffe` module from our `CAFFE_ROOT` dir.  If it's not already included
in your Python environment, you can force it to load by adding it manually. Along with it we'll
also import caffe's protobuf module:

```python
import numpy as np

caffe_root = '/path/to/your/caffe_root'
sys.path.insert(0, os.path.join(caffe_root, 'python'))
import caffe
from caffe.proto import caffe_pb2
```

Next we need to tell Caffe whether to [use the CPU or GPU](https://github.com/BVLC/caffe/blob/61944afd4e948a4e2b4ef553919a886a8a8b8246/python/caffe/_caffe.cpp#L50-L52).
For our experiments, the CPU is fine:

```python
caffe.set_mode_cpu()
```

Now we can use `caffe` to load our trained network.  To do so, we'll need some of the files we downloaded
from DIGITS, namely:

* `deploy.prototxt` - our "network file", the description of the network.
* `snapshot_iter_90.caffemodel` - our trained "weights"

We obviously need to provide the full path, and I'll assume that my files are in a dir called `model/`:

```python
model_dir = 'model'
deploy_file = os.path.join(model_dir, 'deploy.prototxt')
weights_file = os.path.join(model_dir, 'snapshot_iter_90.caffemodel')
net = caffe.Net(deploy_file, caffe.TEST, weights=weights_file)
```

The `caffe.Net()` [constructor](https://github.com/BVLC/caffe/blob/61944afd4e948a4e2b4ef553919a886a8a8b8246/python/caffe/_caffe.cpp#L91-L117)
takes a network file, a phase (`caffe.TEST` or `caffe.TRAIN`), as well as an optional weights filename.  When
we provide a weights file, the `Net` will automatically load them for us. The `Net` has a number of
[methods and attributes](https://github.com/BVLC/caffe/blob/master/python/caffe/pycaffe.py) you can use.

**Note:** There is also a [deprecated version of this constructor](https://github.com/BVLC/caffe/blob/61944afd4e948a4e2b4ef553919a886a8a8b8246/python/caffe/_caffe.cpp#L119-L134),
which seems to get used often in sample code on the web. It looks like this, in case you encounter it:

```python
net = caffe.Net(str(deploy_file), str(model_file), caffe.TEST)
```

We're interested in loading images of various sizes into our network for testing. As a result,
we'll need to *transform* them into a shape that our network can use (i.e., colour, 256x256).
Caffe provides the [`Transformer` class](https://github.com/BVLC/caffe/blob/61944afd4e948a4e2b4ef553919a886a8a8b8246/python/caffe/io.py#L98)
for this purpose.  We'll use it to create a transformation appropriate for our images/network:

```python
transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
# set_transpose: https://github.com/BVLC/caffe/blob/61944afd4e948a4e2b4ef553919a886a8a8b8246/python/caffe/io.py#L187
transformer.set_transpose('data', (2, 0, 1))
# set_raw_scale: https://github.com/BVLC/caffe/blob/61944afd4e948a4e2b4ef553919a886a8a8b8246/python/caffe/io.py#L221
transformer.set_raw_scale('data', 255)
# set_channel_swap: https://github.com/BVLC/caffe/blob/61944afd4e948a4e2b4ef553919a886a8a8b8246/python/caffe/io.py#L203
transformer.set_channel_swap('data', (2, 1, 0))
```

We can also use the `mean.binaryproto` file DIGITS gave us to set our transformer's mean:

```python
# This code for setting the mean from https://github.com/NVIDIA/DIGITS/tree/master/examples/classification
mean_file = os.path.join(model_dir, 'mean.binaryproto')
with open(mean_file, 'rb') as infile:
    blob = caffe_pb2.BlobProto()
    blob.MergeFromString(infile.read())
    if blob.HasField('shape'):
        blob_dims = blob.shape
        assert len(blob_dims) == 4, 'Shape should have 4 dimensions - shape is %s' % blob.shape
    elif blob.HasField('num') and blob.HasField('channels') and \
            blob.HasField('height') and blob.HasField('width'):
        blob_dims = (blob.num, blob.channels, blob.height, blob.width)
    else:
        raise ValueError('blob does not provide shape or 4d dimensions')
    pixel = np.reshape(blob.data, blob_dims[1:]).mean(1).mean(1)
    transformer.set_mean('data', pixel)
```

If we had a lot of labels, we might also choose to read in our labels file, which we can use
later by looking up the label for a probability using its position (e.g., 0=dolphin, 1=seahorse):

```python
labels_file = os.path.join(model_dir, 'labels.txt')
labels = np.loadtxt(labels_file, str, delimiter='\n')
``` 

Now we're ready to classify an image.  We'll use [`caffe.io.load_image()`](https://github.com/BVLC/caffe/blob/61944afd4e948a4e2b4ef553919a886a8a8b8246/python/caffe/io.py#L279)
to read our image file, then use our transformer to reshape it and set it as our network's data layer:

```python
# Load the image from disk using caffe's built-in I/O module
image = caffe.io.load_image(fullpath)
# Preprocess the image into the proper format for feeding into the model
net.blobs['data'].data[...] = transformer.preprocess('data', image)
```

> Q: "How could I use images (i.e., frames) from a camera or video stream instead of files?"

Great question, here's a skeleton to get you started:

```python
import cv2
...
# Get the shape of our input data layer, so we can resize the image
input_shape = net.blobs['data'].data.shape
...
webCamCap = cv2.VideoCapture(0) # could also be a URL, filename
if webCamCap.isOpened():
    rval, frame = webCamCap.read()
else:
    rval = False

while rval:
    rval, frame = webCamCap.read()
    net.blobs['data'].data[...] = transformer.preprocess('data', frame)
    ...

webCamCap.release()
```

Back to our problem, we next need to run the image data through our network and read out
the probabilities from our network's final `'softmax'` layer, which will be in order by label category:

```python
# Run the image's pixel data through the network
out = net.forward()
# Extract the probabilities of our two categories from the final layer
softmax_layer = out['softmax']
# Here we're converting to Python types from ndarray floats
dolphin_prob = softmax_layer.item(0)
seahorse_prob = softmax_layer.item(1)

# Print the results. I'm using labels just to show how it's done
label = labels[0] if dolphin_prob > seahorse_prob else labels[1]
filename = os.path.basename(fullpath)
print '%s is a %s dolphin=%.3f%% seahorse=%.3f%%' % (filename, label, dolphin_prob*100, seahorse_prob*100)
```

Running the full version of this (see [src/classify-samples.py](src/classify-samples.py)) using our
fine-tuned GoogLeNet network on our [data/untrained-samples](data/untrained-samples) images gives
me the following output:

```
[...truncated caffe network output...]
dolphin1.jpg is a dolphin dolphin=99.968% seahorse=0.032%
dolphin2.jpg is a dolphin dolphin=99.997% seahorse=0.003%
dolphin3.jpg is a dolphin dolphin=99.943% seahorse=0.057%
seahorse1.jpg is a seahorse dolphin=0.365% seahorse=99.635%
seahorse2.jpg is a seahorse dolphin=0.000% seahorse=100.000%
seahorse3.jpg is a seahorse dolphin=0.014% seahorse=99.986%
```

I'm still trying to learn all the best practices for working with models in code. I wish I had more
and better documented code examples, APIs, premade modules, etc to show you here. To be honest,
most of the code examples I’ve found are terse, and poorly documented--Caffe’s
documentation is spotty, and assumes a lot.

It seems to me like there’s an opportunity for someone to build higher-level tools on top of the
Caffe interfaces for beginners and basic workflows like we've done here.  It would be great if
there were more simple modules in high-level languages that I could point you at that “did the
right thing” with our model; someone could/should take this on, and make *using* Caffe
models as easy as DIGITS makes *training* them.  I’d love to have something I could use in node.js,
for example.  Ideally one shouldn’t be required to know so much about the internals of the model or Caffe.
I haven’t used it yet, but [DeepDetect](https://deepdetect.com/) looks interesting on this front,
and there are likely many other tools I don’t know about.

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

Let's look at how each of our three attempts did with this challenge:

### Model Attempt 1: AlexNet from Scratch (3rd Place)

| Image | Dolphin | Seahorse | Result | 
|-------|---------|----------|--------|
|[dolphin1.jpg](data/untrained-samples/dolphin1.jpg)| 71.11% | 28.89% | :expressionless: |
|[dolphin2.jpg](data/untrained-samples/dolphin2.jpg)| 99.2% | 0.8% | :sunglasses: |
|[dolphin3.jpg](data/untrained-samples/dolphin3.jpg)| 63.3% | 36.7% | :confused: |
|[seahorse1.jpg](data/untrained-samples/seahorse1.jpg)| 95.04% | 4.96% | :disappointed: |
|[seahorse2.jpg](data/untrained-samples/seahorse2.jpg)| 56.64% | 43.36 |  :confused: |
|[seahorse3.jpg](data/untrained-samples/seahorse3.jpg)| 7.06% | 92.94% |  :grin: |

### Model Attempt 2: Fine Tuned AlexNet (2nd Place)

| Image | Dolphin | Seahorse | Result | 
|-------|---------|----------|--------|
|[dolphin1.jpg](data/untrained-samples/dolphin1.jpg)| 99.1% | 0.09% |  :sunglasses: |
|[dolphin2.jpg](data/untrained-samples/dolphin2.jpg)| 99.5% | 0.05% |  :sunglasses: |
|[dolphin3.jpg](data/untrained-samples/dolphin3.jpg)| 91.48% | 8.52% |  :grin: |
|[seahorse1.jpg](data/untrained-samples/seahorse1.jpg)| 0% | 100% |  :sunglasses: |
|[seahorse2.jpg](data/untrained-samples/seahorse2.jpg)| 0% | 100% |  :sunglasses: |
|[seahorse3.jpg](data/untrained-samples/seahorse3.jpg)| 0% | 100% |  :sunglasses: |

### Model Attempt 3: Fine Tuned GoogLeNet (1st Place)

| Image | Dolphin | Seahorse | Result | 
|-------|---------|----------|--------|
|[dolphin1.jpg](data/untrained-samples/dolphin1.jpg)| 99.86% | 0.14% |  :sunglasses: |
|[dolphin2.jpg](data/untrained-samples/dolphin2.jpg)| 100% | 0% |  :sunglasses: |
|[dolphin3.jpg](data/untrained-samples/dolphin3.jpg)| 100% | 0% |  :sunglasses: |
|[seahorse1.jpg](data/untrained-samples/seahorse1.jpg)| 0.5% | 99.5% |  :sunglasses: |
|[seahorse2.jpg](data/untrained-samples/seahorse2.jpg)| 0% | 100% |  :sunglasses: |
|[seahorse3.jpg](data/untrained-samples/seahorse3.jpg)| 0.02% | 99.98% |  :sunglasses: |

## Conclusion

It’s amazing how well our model works, and what’s possible by fine tuning a pretrained network.
Obviously our dolphin vs. seahorse example is contrived, and the dataset overly limited--we really
do want more and better data if we want our network to be robust.  But since our goal was to examine
the tools and workflows of neural networks, it’s turned out to be an ideal case, especially since it
didn’t require expensive equipment or massive amounts of time.

Above all I hope that this experience helps to remove the overwhelming fear of getting started.
Deciding whether or not it’s worth investing time in learning the theories of machine learning and
neural networks is easier when you’ve been able to see it work in a small way.  Now that you’ve got
a setup and a working approach, you can try doing other sorts of classifications.  You might also look
at the other types of things you can do with Caffe and DIGITS, for example, finding objects within an
image, or doing segmentation.

Have fun with machine learning!
