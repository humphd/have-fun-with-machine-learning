# 機器學習動手玩：給新手的教學

> 原作者：[David Humphrey](https://github.com/humphd)  
  中文（繁體）語系譯者：[Birkhoff Lee](https://fb.me/birkhofflee)。歡迎改進翻譯，譯者還只是個國中生。

##序言

這是一個提供給**無人工智慧背景知識**程式員的機器學習**實作教學**。使用類神經網絡事實上並不需要博士學位，而且你也不需要成為下一個在人工智慧領域有極大突破的人，而且我們現在的成就已經十分驚人，且具有高可用性。我相信我們這些人之中是想玩這個東西——就跟我們玩開源軟體一樣——而不是將它視為一個研究議題。

在這篇教學中，我們的目標是寫一個程式，能夠使用機器學習來進行高度確定的判定 —— 僅僅依該圖片來判斷在 [data/untrained-samples](data/untrained-samples) 中的陌生圖案是**海豚**還是**海馬**。 以下是兩個我們將會用到的範例圖案：

![一隻海豚](data/untrained-samples/dolphin1.jpg?raw=true "Dolphin")
![一隻海馬](data/untrained-samples/seahorse1.jpg?raw=true "Seahorse")

為了進行判定，我們將訓練一個[卷積神經網絡](https://zh.wikipedia.org/wiki/%E5%8D%B7%E7%A7%AF%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C)。我們將以實踐者的角度與第一原理* 的觀點來解決這個問題。人工智慧現在十分令人興奮，但是下面將寫到的內容將讓你感覺像是一個物理教授在黑板上像你講解如何在腳踏車上玩特技，而不是與你的朋友在公園內練習。

我決定不在部落格寫這篇文，而是在 GitHub 上。因為我確定以下我所寫的內容有些會誤導讀者，或根本是錯的。我還在自學這一方面的知識，而且我發現一些新手教學會成為障礙。 如果你發現我哪裡寫錯了、或是缺少了什麼重要的細節，請建立一個 pull request。

把那些拋諸腦後吧！讓我來教你如何使用這些東西。

> \* 譯者按：「第一原理」是指不必經過驗證，即已明白的原理，即是理由已經存在於原理之中，也是自證原理。
  就範圍大小區分，第一原理可以是解釋所有事件的終極真理，也可以視為一個系統一致性、連貫性的
  一種根源性的解釋。

##概覽

以下是我們將要探索的內容：

* 設定並且使用現有且開放原始碼的機器學習技術，特別是 [Caffe](http://caffe.berkeleyvision.org/) 與 [DIGITS](https://developer.nvidia.com/digits)。
* 建立一個圖像資料集
* 從頭開始訓練一個類神經網絡
* 用我們的類神經網絡測試判別它從沒見過的圖案
* 對現有的類神經網絡（AlexNet 與 GoogLeNet）進行微調以改進我們類神經網絡的判別準確度
* 部署並使用我們的類神經網絡

這個教學將不會教你這些類神經網絡是如何設計的、其背後的理論，也不會給你什麼數學表達式。我不假裝全然理解我接下來將教你的。相反地，我們將以有趣的方式來用現有的東西解決一個困難的問題。

> 問：「我知道你說了我們不會討論類神經網絡背後的理論，但是我還是覺得在我們開始之前我至少需要一些概覽。我該從何開始？」

網路上大概有上百個關於這個東西的介紹。其中不乏短文、甚至完整的線上課程應有盡有。看你希望如何學習，這裡有三個不錯的選項供你參考：

* 這個奇妙的[部落格文章](https://jalammar.github.io/visual-interactive-guide-basics-neural-networks/)，由 J Alammar 所著。它以直觀的例子介紹了類神經網絡的概念。
* 類似的，這部由 [Brandon Rohrer](https://www.youtube.com/channel/UCsBKTrp45lTfHa_p49I2AEQ) 所拍攝的 [介紹影片](https://www.youtube.com/watch?v=FmpDIaiMIeA) 是個很不錯的關於卷積神經網絡（我們將會使用它）的介紹。
* 如果你想了解更多其背後的理論，我會推薦你[這本書](http://neuralnetworksanddeeplearning.com/chap1.html)，它由 [Michael Nielsen](http://michaelnielsen.org/) 所著。

##設定

###安裝 Caffe

首先，我們將會使用到來自 Berkely Vision 及 Learning Center 的 [Caffe 深度學習框架](http://caffe.berkeleyvision.org/)（BSD 協議）。

> 問：「等一下，為什麼要用 Caffe？為什麼不使用最近很多人都在討論的 TensorFlow?」

我們有很多很棒的選擇，而且你應該都稍微了解一下他們。[TensorFlow](https://www.tensorflow.org/) 是很不錯，而且你也應該玩玩看。不過，由於以下這些原因我選擇使用 Caffe：

* 它專門用來解決電腦視覺相關的問題
* 它支援 C++、Python，及[即將到來的 Node.js 支援]((https://github.com/silklabs/node-caffe)
* 它又快又穩定

不過最首要的原因是**你不需要寫任何程式碼**來使用它。你可以用說明的方式做任何事，或是用命令行工具（Caffe 使用結構化文字檔來定義網絡架構）。還有，你可以透過一些不錯的前端介面來使用 Caffe 以訓練及驗證你的網絡，這將變得十分簡單。我們將會使用 [nVidia 公司的 DIGITS](https://developer.nvidia.com/digits) 來當做我們的前端介面。

要安裝好 Caffe 有點費力。這裡有對於某些平台的[安裝指引](http://caffe.berkeleyvision.org/installation.html)，及一些已經預先編譯好的 Docker 或 AWS 配置。

**注意：** 當我在寫這篇時，我使用了這個非正式版本的 Caffe：https://github.com/BVLC/caffe/commit/5a201dd960840c319cefd9fa9e2a40d2c76ddd73

想在 Mac 上安裝它很容易讓你感到挫敗，在編譯 Caffe 時會有很多版本問題，我試了很多天。我跟隨了很多教學，每次都有一些稍微不一樣的問題。最後我找到了最接近的[這篇](https://gist.github.com/doctorpangloss/f8463bddce2a91b949639522ea1dcbe4)。我也推薦[最近的這篇文章](https://eddiesmo.wordpress.com/2016/12/20/how-to-set-up-caffe-environment-and-pycaffe-on-os-x-10-12-sierra/)，裡面也連結到很多我看到的討論串。對於中文讀者來說，[BirkhoffLee](https://github.com/BirkhoffLee) 也推薦了他的[完整中文版教學](https://blog.birkhoff.me/macos-sierra-10-12-2-build-caffe)，教你如何在 macOS Sierra 上編譯 Caffe。

將 Caffe 安裝好是到目前為止我們將做的最難的事情，這很好，因為你可能會認為人工智慧方面的問題會更難。如果你遇到問題，千萬不要放棄，這是值得的。如果要讓我再做一次，我不會直接在 Mac 上安裝它，而是在一台 Ubuntu 虛擬機器上安裝。如果你有問題，這裡是 [Caffe 使用者群組](https://groups.google.com/forum/#!forum/caffe-users)，你可以在此提問。

> 問：「訓練一個類神經網絡需不需要很好的硬體？如果我沒有很棒的 GPU 呢？」

事實上沒錯。訓練一個深度類神經網絡需要非常大量的預算能力和精力...前提是你要用非常大量的資料集從頭開始訓練。我們不會這樣做。我們的秘訣是用一個別人事先以上百小時訓練好的類神經網絡，然後我們再針對我們的資料集進行微調。下面的教學將會教你如何這樣做。簡單來說，下面我所做的事情，都是我在一臺一歲的 MacBook Pro 上做的（這台沒有很好的 GPU）。

順便說一下，因為我的 MacBook Pro 只有 Intel 整合繪圖處理器（即內建顯示核心），它沒有 nVidia 的 GPU，所以我決定使用 [Caffe 的 OpenCL 版本](https://github.com/BVLC/caffe/tree/opencl)，而且它在我的筆電上跑的很不錯。

當你把 Caffe 搞定之後，你應該有，或能做這些東西：

* 一個資料夾，裡面有你編譯好的 Caffe。如果你用了標準的方法來編譯它，裡面會有一個叫做「`build/`」的資料夾，它裡面有你跑 Caffe 所需要的所有東西，像是 Python 的綁定什麼的。那個包含 `build/` 的資料夾就是你的「`CAFFE_ROOT`」（我們等一下會用到這個）。
* 執行 `make test && make runtest` 要能通過測試
* 安裝完所有 Python 相依性套件之後（在 `python/` 內執行 `for req in $(cat requirements.txt); do pip install $req; done`），執行 `make pycaffe && make pytest` 要能通過測試
* 你也應該執行 `make distribute` 以建立一個含有所有必須的 header、binary 之類東西的可散佈版的 Caffe。

在我的機器上，我已經完整的編譯好 Caffe 了。我的 CAFFE_ROOT 裡面的基本結構看起來長這樣：

```
caffe/
    build/
        python/
        lib/
        tools/
            caffe ← 這是我們主要使用的二進位檔案
    distribute/
        python/
        lib/
        include/
        bin/
        proto/
```

現在，我們已經萬事俱全，可以訓練、測試我們的神經網絡以及為它編寫程式了。在下一節我們將為 Caffe 添加一個十分友好的網頁介面——「DIGITS」，這樣我們訓練及測試我們的類神經網絡時將變得更簡單。

###安裝 DIGITS

nVidia 的[深度學習 GPU 訓練系統（DIGITS）](https://github.com/NVIDIA/DIGITS)是個 BSD 協議的 Python 網頁應用程式，專門拿來訓練類神經網絡用的。雖然我們可以在命令行（或是自己寫程式）完成任何 DIGITS 對 Caffe 做的事，但是用 DIGITS 將讓我們更容易上手。我發現 DIGITS 的視覺化資料、即時圖表和其他類似的功能讓這一切都變得更有趣了。因為你還在實驗及試著學習，我非常推薦以 DIGITS 上手。

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
might need to fix some of the layers, might need to insert new layers, etc. However,
my experience is that it “Just Works” much of the time, and it’s worth you simply doing
an experiment to see what you can achieve using our naive approach.

####Uploading Pretrained Networks

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

##Using our Model

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

###Python example

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

##Conclusion

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
