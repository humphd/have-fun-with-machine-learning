#Preface

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

#Overview

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

#Setup

##Installing Caffe

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

##Installing DIGITS

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

#Training a Neural Network

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

##Prepare the Dataset

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


