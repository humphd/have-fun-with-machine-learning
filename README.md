#Preface

This is a hands-on guide to machine learning for programmers with
*no background* in AI. Our goal will be to write a program that can
predict, with a high degree of certainty, whether the images in
[data/untrained-samples](data/untrained-samples) are of **dolphins**
or **seahorses** using only the images themselves.

![Alt text](relative/path/to/img.jpg?raw=true "Title")

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

