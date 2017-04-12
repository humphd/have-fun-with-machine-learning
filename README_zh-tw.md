# 機器學習動手玩：給新手的教學

> Author: David Humphrey (original [English version](README.md))  
  中文（繁體）語系譯者：[Birkhoff Lee](https://fb.me/birkhofflee)

### 譯者有話要說 Translator's Note

各位好。這是一篇很棒的教學，小弟希望能夠幫助到中文讀者，於是利用自己課後的時間（我還是個國中生）來翻譯這篇文章。在這篇文章裡有很多專業的術語，而有些是我不曾聽聞的——例如「Transfer Learning」——我遇到這些我不清楚的術語時，我使用 Google 來搜尋相關的中文文獻以期得到該術語現有的翻譯。還有一些內容是無法直接從英文翻到中文的，必須重建語境來翻譯，因此我會盡可能地不偏離原文的意思。如果您發現哪裡的翻譯有問題或是可以翻譯地更好，請開一個 issue 或是直接發一個 pull request 來協助修正翻譯，謝謝。

## 序言

這是一個提供給**無人工智慧背景知識**程式員的機器學習**實作教學**。使用類神經網絡事實上並不需要什麼博士學位，而且你也不需要成為下一個在人工智慧領域有極大突破的人，而且我們現在的成就已經十分驚人，且可用性極高。我相信大多數人是想玩玩看這個東西——就跟我們玩開源軟體一樣——而不是將它視為一個研究議題。

在這篇教學中，我們的目標是寫一個程式，能夠使用機器學習來進行精確的判定——僅僅依該圖片來判斷在 [data/untrained-samples](data/untrained-samples) 中的陌生圖案是**海豚**還是**海馬**。以下是兩個我們將會用到的範例圖案：

![一隻海豚](data/untrained-samples/dolphin1.jpg?raw=true "海豚")
![一隻海馬](data/untrained-samples/seahorse1.jpg?raw=true "海馬")

為了進行判定，我們將訓練一個[卷積神經網絡](https://zh.wikipedia.org/wiki/%E5%8D%B7%E7%A7%AF%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C)。我們將以實踐者的角度（而不是以第一原理\* 的觀點）來解決這個問題。人工智慧現在十分令人興奮，不過現在大多數關於 AI 的文章就像是物理學家在黑板上傳授你騎腳踏車技巧一樣，不過你應該與你的朋友在公園內練習才對不是嗎？

我決定在 GitHub 上發表這篇文章，而不是在我的部落格上。因為我確定以下我所寫的內容有些會誤導讀者，或根本是錯的。我還在自學這一方面的知識，而且我發現一些新手教學會成為障礙。如果你發現我哪裡寫錯了、或是缺少了什麼重要的細節，請建立一個 pull request。

把那些拋諸腦後吧！讓我來教你如何使用這些東西。

> \* 譯者按：「第一原理」是指不必經過驗證，即已明白的原理，即是理由已經存在於原理之中，也是自證原理。
  就範圍大小區分，第一原理可以是解釋所有事件的終極真理，也可以視為一個系統一致性、連貫性的
  一種根源性的解釋。

## 概覽

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

## 設定

根據你所使用的平台與作業系統版本，安裝我們將使用到的軟體（Caffe 與 DIGITS）可能會讓你感到十分挫敗。目前為止最簡單的方式是使用 Docker。以下我們將示範如何使用 Docker 來設定，以及如何用原生的方式來設定它。

### 方法 1a：原生安裝 Caffe

首先，我們將會使用到來自 Berkely Vision 及 Learning Center 的 [Caffe 深度學習框架](http://caffe.berkeleyvision.org/)（BSD 協議）。

> 問：「等一下，為什麼要用 Caffe？為什麼不使用最近很多人都在討論的 TensorFlow?」

我們有很多很棒的選擇，而且你應該都稍微了解一下他們。[TensorFlow](https://www.tensorflow.org/) 是很不錯，而且你也應該玩玩看。不過，由於以下這些原因我選擇使用 Caffe：

* 它專門用來解決電腦視覺相關的問題
* 它支援 C++、Python，及[即將到來的 Node.js 支援](https://github.com/silklabs/node-caffe)
* 它既快又穩定

不過最首要的原因是**你不需要寫任何程式碼**來使用它。你可以用說明的方式做任何事，或是用命令行工具（Caffe 使用結構化文字檔來定義網絡架構）。還有，你可以透過一些不錯的前端介面來使用 Caffe 以訓練及驗證你的網絡，這將變得十分簡單。我們將會使用 [nVidia 公司的 DIGITS](https://developer.nvidia.com/digits) 來當做我們的前端介面。

要安裝好 Caffe 有點費力。這裡有對於某些平台的[安裝指引](http://caffe.berkeleyvision.org/installation.html)，及一些已經預先編譯好的 Docker 或 AWS 配置。

**注意：** 當我在寫這篇時，我使用了這個非正式版本的 Caffe：https://github.com/BVLC/caffe/commit/5a201dd960840c319cefd9fa9e2a40d2c76ddd73

想在 Mac 上安裝它很容易讓你感到挫敗，在編譯 Caffe 時會有很多版本問題，我試了很多天，也找了很多教學，每次都有略微不同的問題。最後我找到了最接近的[這篇](https://gist.github.com/doctorpangloss/f8463bddce2a91b949639522ea1dcbe4)。我也推薦[最近發表的這篇](https://eddiesmo.wordpress.com/2016/12/20/how-to-set-up-caffe-environment-and-pycaffe-on-os-x-10-12-sierra/)文章，裡面也提到很多我看到的討論串。對於中文讀者來說，[BirkhoffLee](https://github.com/BirkhoffLee) 也推薦了他完整的[中文版教學](https://blog.birkhoff.me/macos-sierra-10-12-2-build-caffe)，教你如何在 macOS Sierra 上編譯 Caffe。

將 Caffe 安裝好是到目前為止我們將做的最難的事情，這很好，因為你可能會認為人工智慧方面的問題會更難。如果你遇到問題，千萬不要放棄，這是值得的。如果要讓我再做一次，我不會直接在 Mac 上安裝它，而是在一台 Ubuntu 虛擬機器上安裝。如果你有問題，這裡是 [Caffe 使用者群組](https://groups.google.com/forum/#!forum/caffe-users)，你可以在此提問。

> 問：「訓練一個類神經網絡需不需要很好的硬體？如果我沒有很棒的 GPU 呢？」

事實上沒錯。訓練一個深層類神經網絡需要非常大量的預算能力和精力...前提是你要用非常大量的資料集從頭開始訓練。我們不會這樣做。我們的秘訣是用一個別人事先以上百小時訓練好的類神經網絡，然後我們再針對我們的資料集進行微調。下面的教學將會教你如何這樣做。簡單來說，下面我所做的事情，都是我在一臺一歲的 MacBook Pro 上做的（這台沒有很好的 GPU）。

順便說一下，因為我的 MacBook Pro 只有 Intel 整合繪圖處理器（即內建顯示核心），它沒有 nVidia 的 GPU，所以我決定使用 [Caffe 的 OpenCL 版本](https://github.com/BVLC/caffe/tree/opencl)，而且它在我的筆電上跑的很不錯。

當你把 Caffe 搞定之後，你應該有，或能做這些東西：

* 一個資料夾，裡面有你編譯好的 Caffe。如果你用了標準的方法來編譯它，裡面會有一個叫做「`build/`」的資料夾，它裡面有你跑 Caffe 所需要的所有東西，像是 Python 的綁定什麼的。那個包含 `build/` 的資料夾就是你的「`CAFFE_ROOT`」（我們等一下會用到這個）。
* 執行 `make test && make runtest` 要能通過測試
* 安裝完所有 Python 相依性套件之後（在 `python/` 內執行 `pip install -r requirements.txt`），執行 `make pycaffe && make pytest` 要能通過測試
* 你也應該執行 `make distribute` 以建立一個含有所有必須的 header、binary 之類東西的可散佈版的 Caffe。

在我的機器上，我已經完整的編譯好 Caffe 了。我的 CAFFE_ROOT 裡面的基本結構看起來長這樣：

```
caffe/
    build/
        python/
        lib/
        tools/
            caffe ← 這是我們主要使用的執行檔
    distribute/
        python/
        lib/
        include/
        bin/
        proto/
```

現在，我們已經萬事俱全，可以訓練、測試我們的網絡以及為它編寫程式了。在下一節我們將為 Caffe 添加一個十分友好的網頁介面——「DIGITS」，這樣我們訓練及測試我們的網絡時將變得更簡單。

### 方法 1b：原生安裝 DIGITS

nVidia 的[深度學習 GPU 訓練系統（DIGITS）](https://github.com/NVIDIA/DIGITS)是個 BSD 協議的 Python 網頁應用程式，專門用來訓練類神經網絡。雖然我們可以在命令行（或是自己寫程式）完成任何 DIGITS 對 Caffe 做的事，但是用 DIGITS 將讓我們更容易上手。我發現 DIGITS 的視覺化資料、即時圖表和其他類似的功能讓這一切都變得更有趣了。因為你還在實驗及嘗試學習，我非常推薦以 DIGITS 上手。

https://github.com/NVIDIA/DIGITS/tree/master/docs 有一些十分不錯的文檔供你參考，裡面也有[安裝](https://github.com/NVIDIA/DIGITS/blob/master/docs/BuildDigits.md)、[設定](https://github.com/NVIDIA/DIGITS/blob/master/docs/Configuration.md)及[供你上手](https://github.com/NVIDIA/DIGITS/blob/master/docs/GettingStarted.md)的資料。我建議在你開始之前，先把所有東西都稍微看一遍，因為我並不是 DIGITS 的專家——我並不知道它能做的所有事情。如果你有什麼問題想問，公開的 [DIGITS 使用者群組](https://groups.google.com/forum/#!forum/digits-users)是一個不錯的地方。

要安裝且執行 DIGITS 有很多方法，有 Docker image、預先包裝好的 Linux 套件，或者你也可以自行編譯它。我使用的是 Mac，所以我選擇自行編譯它。

**注意：** 當我在寫這篇時，我使用了這個非正式版本的 DIGITS：https://github.com/NVIDIA/DIGITS/commit/81be5131821ade454eb47352477015d7c09753d9

DIGITS 很容易安裝，因為他就只是一堆 Python 腳本。你唯一需要告訴 DIGITS 的一件事就是你的 `CAFFE_ROOT` 在哪裡。你可以用環境變數搞定這件事，然後就可以啟動伺服器了：

```bash
export CAFFE_ROOT=/path/to/caffe
./digits-devserver
```

注意：在 Mac 上我在啟動伺服器時發生了一些問題——啟動伺服器的腳本直接默認了我的 Python 執行檔叫做 `python2`，但是我只有 `python2.7`。你可以建立一個到 `/usr/bin` 的符號連結或是修改 DIGITS 的啟動腳本來使用正確的 Python 執行檔。

當你啟動了伺服器之後，你可以透過你的網頁瀏覽器在這個網址做所有其他的事情（我們等下會做的事）了：http://localhost:5000。

###方法 2：用 Docker 執行 Caffe 與 DIGITS
如果你還沒安裝 [Docker](https://www.docker.com/) 請先安裝它，接著執行以下指令來拉取與執行一個完整的 Caffe + DIGITS 容器。一些需要注意的事項：  
* 確認 8080 連線埠沒有被其他程式佔用，如果被佔用了你也可以將它改為其他的連線埠號碼。
* 將這個 repository clone 下來，然後將 `/path/to/this/repository` 改為你 clone 的位置，容器內的 `/data/repo` 會被綁定到這個資料夾上。

```bash
docker run --name digits -d -p 8080:5000 -v /path/to/this/repository:/data/repo /kaixhin/digits
```

這樣容器就開始執行了，你可以打開你的瀏覽器然後打開 `http://localhost:8080`。所有在這個 repository 的資料都在容器內的 `/data/repo` 了。就這樣。你已經把 Caffe 與 DIGITS 搞定了。

如果你需要 shell access，請使用以下指令：

```bash
docker exec -it digits /bin/bash
```

## 訓練類神經網絡

訓練一個類神經網絡涉及到這些步驟：

1. 組合及準備一個分類好的照片的資料集  
2. 定義這個類神經網絡的架構  
3. 用準備好的資料集訓練及驗證這個網絡

我們將用三種方法做這件事以體現出從頭開始訓練與使用一個預先訓練好的網絡之間的差別，順便了解如何使用 AlexNet 與 GoogLeNet 這兩個相當受歡迎的預先訓練好的網絡，他們常常與 Caffe 和 DIGITS 搭配使用。

我們將使用一個包含了海豚與海馬的小資料集來嘗試訓練。我已經把我使用的照片放在了 [data/dolphins-and-seahorses](data/dolphins-and-seahorses)。你需要最少兩個分類，不過你可以有更多（有些你將會用到的網絡是以一千多個影像分類訓練而成的）。我們的目標是當我們給我們的網絡一個圖片，它能告訴我們他是隻海豚還是海馬。

### 準備資料集

要開始，最簡單的方法是將你的圖片分成這個分類好的資料夾樣式：

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

這裡的每個資料夾都是一個我們想分類的類別（category），在裡面的每個圖片都將被我們用來訓練及驗證我們的網絡。

> 問：「照片都要一樣的大小嗎？那檔案名稱呢？」

兩個都不用管他。在我們餵食網絡之前，圖片的大小都會被一般化。我們會希望我們的照片尺寸是 256 x 256 像素，DIGITS 等一下會自動裁切或縮放（這裡選擇縮放）我們的圖片。那些檔案名稱你要怎麼取根本沒差——重要的是它們是在什麼分類裡。

> 問：「我可以再細分我的分類嗎？」

可以。詳閱 https://github.com/NVIDIA/DIGITS/blob/digits-4.0/docs/ImageFolderFormat.md 。

我們將使用這些在硬碟上的照片來建立一個**新的資料集**，而且是一個**分類用資料集**。

![建立一個資料集](images/create-new-dataset.png?raw=true "建立一個資料集")

我們將使用 DIGITS 的默認設定，然後將 **Training Images** 指向我們 [data/dolphins-and-seahorses](data/dolphins-and-seahorses) 的資料夾。DIGITS 將會以 `dolphin` 與 `seahorse` 這兩個分類來建立一個縮放好（256 x 256）的資料集，其中的 75% 用來訓練，另外的 25% 用來測試。

給你的資料集取個名字：`dolphins-and-seahorses`，然後點選 **Create**。

![新的影像辨識資料集](images/new-image-classification-dataset.png?raw=true "新的影像辨識資料集")

這會建立我們的資料集，在我的筆電上只用了 4 秒就跑完了。最後我在兩個類別裡共有 92 個訓練用圖片（49 個海豚和 43 個海馬）和 30 個驗證用圖片（16 個海豚和 43 海馬）。這是個十分小的資料集，不過對於我們的實驗和學習用途十分完美——訓練及驗證一個用這個資料集的網絡不會花我們一輩子的時間。

如果你想看看縮放之後的圖片，你可以**瀏覽資料庫**。

![Explore the db](images/explore-dataset.png?raw=true "Explore the db")

### 訓練：第一次嘗試，從頭開始訓練

回到 DIGITS 的主畫面，我們需要先建立一個新的**分類用模型**：

![建立分類用模型](images/create-classification-model.png?raw=true "建立分類用模型")

我們將從訓練一個使用我們 `dolphins-and-seahorses` 資料集的模型開始，我們將以 DIGITS 給的默認設定值來訓練它。這是我們的第一個網絡，我們選擇使用一個標準的網絡架構——「[AlexNet (pdf)](http://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf)」。[AlexNet 的設計](http://vision.stanford.edu/teaching/cs231b_spring1415/slides/alexnet_tugce_kyunghee.pdf) 在 2012 年贏得了一個大型的電腦視覺比賽——ImageNet。這個比賽要求將一百二十萬個圖像分類到一千多種不同的分類中。

![新的分類用模型 1](images/new-image-classification-model-attempt1.png?raw=true "新的分類用模型 1")

Caffe 使用結構化的文字檔案來定義網絡架構。這些檔案使用的是 [Google 的 Protocol Buffers](https://developers.google.com/protocol-buffers/)。你可以閱讀 Caffe 使用的[整個架構](https://github.com/BVLC/caffe/blob/master/src/caffe/proto/caffe.proto)。
這不是我們主要要處理的部分，不過他們的存在值得我們注意，因為我們等會要修改他們。AlexNet 的 prototxt 檔案長這樣，例如：https://github.com/BVLC/caffe/blob/master/models/bvlc_alexnet/train_val.prototxt 。

我們將會訓練我們的網絡 **30 個循環週期**。這表示網絡會使用我們的訓練圖片來學習，接著使用驗證圖片來測試他自己，然後根據結果來調整網絡的權重，然後重複這整個過程三十遍。當它每次完成一個循環之後我們會得到它的**準確度（_accuracy_）**（0% ~ 100%，越高越好）以及**損失（_loss_）**（所有錯誤的總和，值越低越好）。對於一個網路而言，最理想的狀況是有高準確度與最低的損失。

一開始，我們網路的準確度大致低於 50%。這十分合理，因為它一開始只是在用隨機的權重值來在兩個分類之間進行猜測。隨著訓練的時間增加，它的準確度可以達到 87.5%，且損失為 0.37。我的電腦用了不到六分鐘的時間就跑完這 30 個循環週期了。

![模型 嘗試 1](images/model-attempt1.png?raw=true "模型 嘗試 1")

我們可以上傳一張照片或一個圖片的 URL 來測試我們的模型。讓我們使用一些不在我們資料集的圖片來測試看看：

![模型 1 分類 1](images/model-attempt1-classify1.png?raw=true "模型 1 分類 1")

![模型 1 分類 2](images/model-attempt1-classify2.png?raw=true "模型 1 分類 2")

看起來還蠻不錯的，但是這一張的話 ...：

![模型 1 分類 3](images/model-attempt1-classify3.png?raw=true "模型 1 分類 3")

我們的網絡在這張圖上完全失敗了，而且將海馬混淆成了海豚，更糟糕的是，它十分有自信地認為這張圖是海豚。

事實上其實是我們的資料集太小了，沒辦法訓練一個很好的類神經網絡。我們很需要上萬甚至數十萬張照片來訓練，如果要這樣，我們還會需要十分強大的運算能力來處理這些照片。

### 訓練：第二次嘗試，微調 AlexNet

#### 微調背後的原理 

> 譯者有話要說（Translator's Note）：本段的內容較為複雜（很多專業術語），因此我的翻譯可能沒有很好。如果你願意，你可以看看原文，並希望你能順便幫忙改進翻譯，謝謝。

從頭開始設計一個類神經網絡、取得足夠的資料來訓練它（例如上百萬張照片）以及用好幾周的時間使用 GPU 來運算已經超出我們大多數人的能力範圍了。如果要使用較少的資料來訓練，我們會採用一個叫做「**遷移學習（_Transfer Learning_）**」的技術，也有人稱之為「**微調（_Fine Tuning_）**」。「微調」利用深層類神經網絡的架構及事先訓練好的網絡來達成一開始的物件偵測。

想像一下你一拿起望遠鏡要看很遠很遠的東西的時候，你會先將望遠鏡貼近你的眼睛，接著你看到的一切都是模糊的。隨著望遠鏡的焦距的調整，你會慢慢開始看見顏色、線條、形狀…… 慢慢地你就能看清楚一隻鳥的形狀。再稍微調整一下，你就能辨識這隻鳥的種類了。這就是使用一個類神經網絡的過程。

在一個多層網絡中，初始層（_initial layer_）提取一些特徵（例如：邊緣），接下來的層使用這些特徵來偵測形狀（例如輪子與眼睛），然後送到以在之前的層累積的特徵來偵測物件的最終分類層（例如一隻貓跟一隻狗）。一個網絡必須能夠從像素點開始掃描，到圓形、到眼睛、到朝著特定方向的兩個眼睛等等，直到最終能夠斷定這個照片內描繪的是一隻貓。

我們想做的是讓一個現有的、事先訓練好的網絡能夠專門來分類一些全新的影像分類，而不是讓它來分類當初用來訓練這個網絡的圖形。之所以這樣做是因為這種網絡已經知道如何「看見」圖形中的特徵，然後我們要重新訓練它來讓它能「看見」我們要他分類的特殊圖形。我們不需要從頭開始設定大多數的層——我們想要轉移這些已經學習好的層到我們的新分類任務。不像我們之前的訓練嘗試使用的是隨機的權重，我們這次要使用最終網絡中已有的權重來進行訓練。總而言之，我們將把最終的分類層丟掉，然後用**我們自己**的影像資料集來重新訓練它，將他微調到我們自己的影像分類。

如果要這樣做，我們需要一個與所需資料足夠相似的現有網絡，這樣它學習到的權重對我們來說才會有用處。幸運的是我們接下來將使用的網絡是曾使用上百萬個來自 [ImageNet](http://image-net.org/) 大自然的照片來進行訓練的網絡，因此它對非常多種不同的分類任務都十分的有用處。

這項技術常被用來做有趣的事情，例如從自醫學圖像中掃描是否有眼部疾病、識別從海上採集的浮游生物顯微圖像，到分類 Flickr 網站圖片的藝術風格。

跟所有的機器學習一樣，如果你想做到完美，你需要了解你的資料以及網絡架構——你必須注意這些資料是否會造成過度學習（_overfitting_）、你可能需要修復其中幾層，或是加入新的幾層，諸如此類。總之，我的經驗是它在大多數的時候是可行的，你值得試試看，看你能用我們的方法做得如何。

#### 上傳已事先訓練好的網絡

在我們第一次的嘗試中，我們使用了 AlexNet 的架構，但是在該網絡的層中我們以隨機的權重來開始訓練。我們現在希望能夠下載並使用一個已經使用龐大的資料集來訓練過的 AlexNet 版本。

令人感激的是我們完全可以這樣做。一個 AlexNet 快照（_snapshot_）可以在這裡下載：https://github.com/BVLC/caffe/tree/master/models/bvlc_alexnet 。
我們需要 `.caffemodel` 檔案，它裡面包含了已經訓練過的權重。我們可以在此下載它：http://dl.caffe.berkeleyvision.org/bvlc_alexnet.caffemodel 。

當你在下載他們的時候，我們再順便多下載一個吧。在 2014 年，Google 使用了一個 22 層的類神經網絡 [GoogLeNet](https://research.google.com/pubs/pub43022.html) (代號為「Inception」）贏了同一個 ImageNet 比賽：
GoogLeNet 也有個快照可以下載，在這裡：https://github.com/BVLC/caffe/tree/master/models/bvlc_googlenet 。跟上次一樣，我們會需要含有已訓練過權重的 `.caffemodel` 檔案，你可以在這裡下載它：http://dl.caffe.berkeleyvision.org/bvlc_googlenet.caffemodel 。

有了這些 `.caffemodel` 檔案，我們就可以把他們上傳到 DIGITS 裡了。在 DIGITS 的首頁選擇「**Pretrained Models**」然後選擇 「**Upload Pretrained Model**」：

![載入事先訓練好的模型](images/load-pretrained-model.png?raw=true "載入事先訓練好的模型")

這兩個模型我們都使用 DIGITS 提供的預設設定。我們只需要提供 `Weights (**.caffemodel)` ，即權重值檔案 `.caffemodel` 以及 `Model Definition (original.prototxt)` 模型定義檔案 `original.prototxt`。點一下對應的按鈕並選擇你的檔案就可以上傳。

GoogLeNet 的模型定義檔案我們使用 https://github.com/BVLC/caffe/blob/master/models/bvlc_googlenet/train_val.prototxt ，AlexNet 的我們使用 https://github.com/BVLC/caffe/blob/master/models/bvlc_alexnet/train_val.prototxt 。我們不會使用到分類標籤（_classification labels_），所以我們將跳過 `labels.txt`。

![上傳事先訓練好的模型](images/upload-pretrained-model.png?raw=true "上傳事先訓練好的模型")

記得兩個網絡（AlexNet 與 GoogLeNet）都要上傳，兩個網絡我們下面都會用到。

> 問：「有其他可以拿來微調的網絡嗎？」

[Caffe Model Zoo](http://caffe.berkeleyvision.org/model_zoo.html) 還有蠻多可以用的已訓練好的網絡，詳閱 https://github.com/BVLC/caffe/wiki/Model-Zoo 。

#### 針對海豚與海馬來微調 AlexNet

用一個已訓練好的 Caffe 模型來訓練一個網絡還蠻像是從頭開始訓練的，只不過我們需要做一些細微的調整。首先，我們將調整**基礎學習速率**（_**Base Learning Rate**_），因為我們不需要很大的變動（我們在微調），因此我們將把它從 0.01 改為 0.001。接下來選取下面的「**Pretrained Network**（**事先訓練好的網絡**）」，然後選擇 **Customize**（**自定義**）。

![新的圖像分類用模型](images/new-image-classification-model-attempt2.png?raw=true "新的圖像分類用模型")

在事先訓練好的模型的 prototext 定義中，我們需要將所有參考重命名到最終的**全連結層（_Fully Connected Layer_）**，全連結層負責最終分類。我們這樣做是因為我們希望模型自我們自己的資料集中重新學習新的分類，而不是使用它原本的訓練資料——我們要把它目前的最終層丟掉。我們必須將最終全連結層的名字「fc8」改為別的名字，就改成「fc9」好了。最後，我們需要把類別數量從 `1000` 改為 `2`，也就是將 `num_output` 改為 `2`。

這是我們所需要作出的更動：

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

這裡有一份我實際在使用的修改過後的檔案：[src/alexnet-customized.prototxt](src/alexnet-customized.prototxt)。

這次我們的準確度從 60% 上下然後立刻爬升到 87.5%，接著再到 96% 然後一路上升到 100%，損失也穩定地下降。五分鐘之後我們的結果是 100% 的準確度與 0.0009 的損失。

![模型訓練嘗試 2](images/model-attempt2.png?raw=true "模型訓練嘗試 2")

測試我們前一個網絡判斷錯誤的同一張照片，我們可以看到一個極大的差距：這次的結果是 100% 海馬。

![模型 2 分類 1](images/model-attempt2-classify1.png?raw=true "模型 2 分類 1")

就算是一個小孩畫的海馬都可以：

![模型 2 分類 2](images/model-attempt2-classify2.png?raw=true "模型 2 分類 2")

海豚的結果也一樣：

![模型 2 分類 3](images/model-attempt2-classify3.png?raw=true "模型 2 分類 3")

甚至你覺得可能很難判斷的照片，像是這張照片裡面有很多隻海豚靠在一起，且他們的身體幾乎都在水下，我們的網絡還是能給出正確的答案：

![模型 2 分類 4](images/model-attempt2-classify4.png?raw=true "模型 2 分類 4")


### 訓練：第三次嘗試，微調 GoogLeNet

像是前面被我們拿來微調的的 AlexNet 模型，我們一樣可以用在 GoogLeNet 上。要修改 GoogLeNet 有點棘手，因為你需要重新定義三個全連結層，上次我們只重新定義了一個。

我們要再一次建立一個新的**分類用模型**（_**Classification Model**_）以微調 GoogLeNet 至我們想要的狀態。

![新的分類用模型](images/new-image-classification-model-attempt3.png?raw=true "新的分類用模型")

我們將重新命名所有到這三個全連結辨識層的參考：`loss1/classifier`、`loss2/classifier` 和 `loss3/classifier`。接著我們要重新設定類別的數量（`num_output: 2`）。以下是我們需要更動的地方以修改上述設定：

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

我已經將完整的檔案放在了 [src/googlenet-customized.prototxt](src/googlenet-customized.prototxt)。

> 問：「那對於這些網絡的 prototext 定義修改呢？我們已經修改了全連接層的名字還有類別的數量，還有什麼是我們可以或是應該要修改的東西，且是在什麼情況下？」

很棒的問題，這也是我很想知道的事情。舉例來說，我知道我們可以[「修復」特定的「層」](https://github.com/BVLC/caffe/wiki/Fine-Tuning-or-Training-Certain-Layers-Exclusively)，這樣權重值就不會變動。做別的事情需要理解這些層背後的原理，這已經超出本教學的範圍，也已經超出本教學作者的知識範圍！

就像我們對 AlexNet 所做的微調，我們也降低了 10% 的學習速率（_learning rate_），即從 `0.01` 降低到 `0.001`。

> 問：「在微調時，還有哪些有意義的其他的更動？例如不同的循環週期數（_epochs_）怎麼樣？批尺寸（_batch sizes_）、求解方法（Adam、AdaDelta、AdaGrad 之類的）呢？學習速率（_learning rates_）、策略（Exponential Decay、Inverse Decay 和 Sigmoid Decay 等等）、步長和 gamma 值呢？」

很好的問題，而且也是個我很好奇的問題。我對這些東西也只有很模糊的理解，如果你知道訓練時要如何調整這些數值，我們的設定也應該可以做出一些改進。這東西需要更好的說明文件。

因為 GoogLeNet 的結構比 AlexNet 複雜得多，微調它要花上更多時間。我用了十分鐘用我們的資料集重新在我的筆電上訓練它，達到了 100% 的準確度以及 0.0070 的損失。

![模型 第三次訓練嘗試](images/model-attempt3.png?raw=true "模型 第三次訓練嘗試 3 辨識 3")

跟我們看到微調後 AlexNet 的表現一樣，我們修改過的 GoogLeNet 表現的也十分出色——它是我們目前訓練出最好的模型。

![模型 第三次訓練嘗試 3 辨識 1](images/model-attempt3-classify1.png?raw=true "模型 第三次訓練嘗試 3 辨識 1")

![模型 第三次訓練嘗試 3 辨識 2](images/model-attempt3-classify2.png?raw=true "模型 第三次訓練嘗試 3 辨識 2")

![模型 第三次訓練嘗試 3 辨識 3](images/model-attempt3-classify3.png?raw=true "模型 第三次訓練嘗試 3 辨識 3")

## 使用我們的模型

我們已經訓練並測試好了我們的網絡，是時候下載並實際使用它了。每個我們在 DIGITS 內訓練的模型都有個 **Download Model**（**下載模型**） 的按鈕，也可以用來選擇不同的訓練時快照——例如 `Epoch #30`（`循環週期 #30`）：

![訓練完成的模型](images/trained-models.png?raw=true “訓練完成的模型”)

按下 **Download Model** 將會下載一個 `tar.gz` 壓縮檔，裡面有這些檔案：

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

這裡有個對於如何使用我們剛訓練好的模型的一個[不錯的說明](https://github.com/BVLC/caffe/wiki/Using-a-Trained-Network:-Deploy)，裡面談到了：

> 一個網絡是以其設計（.prototxt）及其權重（.caffemodel）來定義的。
> 當一個網絡在訓練時，該網絡目前的權重狀態存在一個 .caffemodel 檔案中。
> 當有了這兩個檔案，我們就可以從訓練及測試階段進入成品階段（_production phase_）了。
>
> 在它目前的狀態下，這個網絡的設計還沒有為部署準備好。在我們將我們的網絡釋出為產品前，我們常需要用以下的方法來調整它：
>
> 1. 將用來訓練的資料層刪除，因為在分類時我們將不會再為我們的資料提供標籤。
> 2. 刪除任何依賴於資料標籤的層。
> 3. 將網絡設定為可接受資料。
> 4. 確認網絡可以輸出結果。

DIGITS 已經幫我們把這些問題都解決了，也幫我們分離了不同的 `prototxt` 檔案版本。當我們在使用這個網絡的時候我們將會用到以下檔案：

* `deploy.prototxt` —— 網絡的定義檔案，準備好接受影像輸入資料
* `mean.binaryproto` —— 我們的模型會需要我們為每個它要處理的影像減去影像平均值（_image mean_），且這是平均影像資料（_the mean image_）。
* `labels.txt` —— 一個放了我們所有標籤的列表（`dolphin` 與 `seahorse`），如果我們想看到網絡輸出的是這些標籤而不是類別編號時，這個派的上用場。
* `snapshot_iter_90.caffemodel` —— 這是我們網絡訓練好的權重

我們可以用不少方式以這些檔案來分類新的影像。例如，在我們的 `CAFFE_ROOT` 下，我們可以使用 `build/examples/cpp_classification/classification.bin` 來分類一個影像：

```bash
$ cd $CAFFE_ROOT/build/examples/cpp_classification
$ ./classification.bin deploy.prototxt snapshot_iter_90.caffemodel mean.binaryproto labels.txt dolphin1.jpg
```

這會噴出一堆 debug 資訊，接下來是分別對兩個類別的預測結果：

```
0.9997 - “dolphin”
0.0003 - “seahorse”
```

你可以在 [Caffe 範例](https://github.com/BVLC/caffe/tree/master/examples)中閱讀這個東西的[完整 C++ 原始碼](https://github.com/BVLC/caffe/tree/master/examples/cpp_classification)。

對於 Python 應用程式來說，DIGITS 也有提供一個[不錯的範例](https://github.com/NVIDIA/DIGITS/tree/master/examples/classification)。Caffe 範例中也有一個[十分詳細的 Python 版教學](https://github.com/BVLC/caffe/blob/master/examples/00-classification.ipynb)。

### Python 示例

我們來寫一個用圖像分類程式，使用我們微調過的 GoogLeNet 模型來分類我們現有的未經訓練的圖片，它們在 [data/untrained-samples](data/untrained-samples) 裡。我已經把上面的例子都組合了起來，雖然 `caffe` [Python module 的原始碼](https://github.com/BVLC/caffe/tree/master/python) 也已經有了，但是你應該還是會比較喜歡我接下來要講的範例。

接下來我要講的內容都在 [src/classify-samples.py](src/classify-samples.py) 裡，讓我們開始吧！

首先，我們會需要一個叫 [NumPy](http://www.numpy.org/) 的模組（_module_）。我們等下會用它來操作 Caffe 大量使用的 [`ndarray`](https://docs.scipy.org/doc/numpy/reference/generated/numpy.ndarray.html)。如果你還沒使用過他們（我也沒有），你可以先閱讀這篇[快速入門教學](https://docs.scipy.org/doc/numpy-dev/user/quickstart.html)。

接下來，我們會需要從 `CAFFE_ROOT` 載入 `caffe` 模組。如果它還沒有被加入到你的 Python 環境裡，你可以手動加入以強制載入它，我們也會順便載入 Caffe 的 protobuf 模組。

```python
import numpy as np

caffe_root = '/path/to/your/caffe_root'
sys.path.insert(0, os.path.join(caffe_root, 'python'))
import caffe
from caffe.proto import caffe_pb2
```

接下來我們需要告訴 Caffe 要用 [CPU 還是 GPU](https://github.com/BVLC/caffe/blob/61944afd4e948a4e2b4ef553919a886a8a8b8246/python/caffe/_caffe.cpp#L50-L52)。
對於我們的實驗來說，CPU 就夠了：

```python
caffe.set_mode_cpu()
```

現在我們可以使用 `caffe` 來載入我們訓練的網絡，我們會需要一些我們剛從 DIGITS 下載好的檔案：

* `deploy.prototxt` —— 我們的「網絡檔案」，即網絡的描述。
* `snapshot_iter_90.caffemodel` —— 我們訓練好的「權重」資料

我們很顯然地需要提供完整的路徑（_full path_），我假設我的這些檔案放在一個叫做 `model/` 的資料夾：

```python
model_dir = 'model'
deploy_file = os.path.join(model_dir, 'deploy.prototxt')
weights_file = os.path.join(model_dir, 'snapshot_iter_90.caffemodel')
net = caffe.Net(deploy_file, caffe.TEST, weights=weights_file)
```
`caffe.Net()` 的[構造函數（_constructor_）](https://github.com/BVLC/caffe/blob/61944afd4e948a4e2b4ef553919a886a8a8b8246/python/caffe/_caffe.cpp#L91-L117)需要一個網絡檔案、一個階段描述（`caffe.TEST` 或 `caffe.TRAIN`）與一個（可選的）權重檔案名稱。當我們提供一個權重檔案時，`Net` 將會自動幫我們載入它。`Net` 有著不少你可以用的[方法與屬性](https://github.com/BVLC/caffe/blob/master/python/caffe/pycaffe.py)。

**注：** 這個構造函數也有一個[已棄用的版本](https://github.com/BVLC/caffe/blob/61944afd4e948a4e2b4ef553919a886a8a8b8246/python/caffe/_caffe.cpp#L119-L134)，它看起來常常在網絡上的範例程式碼中出現。如果你遇到了它，它看起來會像是這樣：

```python
net = caffe.Net(str(deploy_file), str(model_file), caffe.TEST)
```

我們之後會將各式各樣大小的照片丟到我們的網絡中進行測試。因此，我們將把這些照片**轉換**成一個我們的網絡可以用的形狀（colour、256x256）。Caffe 提供了一個 [`Transformer` class](https://github.com/BVLC/caffe/blob/61944afd4e948a4e2b4ef553919a886a8a8b8246/python/caffe/io.py#L98) 專門用來處理這種情況。我們將會使用它來建立一個適合我們的影像及網絡的轉換器：

```python
transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
# set_transpose: https://github.com/BVLC/caffe/blob/61944afd4e948a4e2b4ef553919a886a8a8b8246/python/caffe/io.py#L187
transformer.set_transpose('data', (2, 0, 1))
# set_raw_scale: https://github.com/BVLC/caffe/blob/61944afd4e948a4e2b4ef553919a886a8a8b8246/python/caffe/io.py#L221
transformer.set_raw_scale('data', 255)
# set_channel_swap: https://github.com/BVLC/caffe/blob/61944afd4e948a4e2b4ef553919a886a8a8b8246/python/caffe/io.py#L203
transformer.set_channel_swap('data', (2, 1, 0))
```

我們也可以使用 DIGITS 給了我們的 `mean.binaryproto` 檔案來設定我們的轉換器：

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

如果我們有很多標籤，我們也可以選擇讀取我們的標籤檔案以用作稍後輸出機率的標籤（如：0=dolphin，1=seahorse）：

```python
labels_file = os.path.join(model_dir, 'labels.txt')
labels = np.loadtxt(labels_file, str, delimiter='\n')
```

現在我們已經準備好來辨識一個影像了。我們要使用 [`caffe.io.load_image()`](https://github.com/BVLC/caffe/blob/61944afd4e948a4e2b4ef553919a886a8a8b8246/python/caffe/io.py#L279) 來讀取我們的影像檔案，然後再使用我們的轉換器來重塑它，最後將它設定為我們網絡的資料層：

```python
# Load the image from disk using caffe's built-in I/O module
image = caffe.io.load_image(fullpath)
# Preprocess the image into the proper format for feeding into the model
net.blobs['data'].data[...] = transformer.preprocess('data', image)
```

> 問：「我要怎麼測試來自相機或視訊流（幀）的影像而不是使用檔案來測試？」

很棒的問題，以下是個供你開始的範例：

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

回到我們的問題，我們接下來需要將我們的影像資料跑一遍我們的網絡然後再讀取我們網絡最終的 `'softmax'` 層返回的機率值，這個機率會依照我們的標籤分類來排序：

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

使用我們微調的 GoogLeNet 以這整個程式（見 [src/classify-samples.py](src/classify-samples.py)）來測試我們的 [data/untrained-samples](data/untrained-samples) 影像，我得到了這些輸出：

```
[...truncated caffe network output...]
dolphin1.jpg is a dolphin dolphin=99.968% seahorse=0.032%
dolphin2.jpg is a dolphin dolphin=99.997% seahorse=0.003%
dolphin3.jpg is a dolphin dolphin=99.943% seahorse=0.057%
seahorse1.jpg is a seahorse dolphin=0.365% seahorse=99.635%
seahorse2.jpg is a seahorse dolphin=0.000% seahorse=100.000%
seahorse3.jpg is a seahorse dolphin=0.014% seahorse=99.986%
```

我還在試著學習所有使用程式碼來處理模型的最佳實踐。我很希望我能告訴你們更多更好的程式碼範例、API 與現有的模組等等。
老實說，我找到的大多數程式碼範例都很簡潔，而且文件都寫的很糟糕——Caffe 的文件有很多問題，而且有著很多的假設。

在我看來，應該有人能夠以 Caffe 介面為基礎來建立更高級別的工具與基本的工作流程，也就是我們以上所做的事情。如果在高級語言中有更多我能夠跟你指出它有「正確地使用我們的模型」的更簡單模組，那一定會很棒；應該有人能夠做到這一點，並且讓*使用* Caffe 模型跟使用 DIGITS 來*訓練*這些模型一樣簡單。舉例來說，我很希望我能夠用 node.js 來操作這些東西。最理想的情況是有一天沒有人會需要知道這麼多有關於模型與 Caffe 的運作模式。[DeepDetect](https://deepdetect.com/) 在這一點看起來十分的有趣，不過我還沒使用過它。而且我認為還有很多我不知道的工具。

## 結果

在最初我們說了我們的目標是寫一個能夠使用一個類神經網絡來正確分類 [data/untrained-samples](data/untrained-samples) 中的所有照片的程式。這些是在上述過程中從來沒用來訓練或是測試過的海豚或海馬的圖片：

### 未訓練的海豚影像

![海豚 1](data/untrained-samples/dolphin1.jpg?raw=true "海豚 1")
![海豚 2](data/untrained-samples/dolphin2.jpg?raw=true "海豚 2")
![海豚 3](data/untrained-samples/dolphin3.jpg?raw=true "海豚 3")

### 未訓練的海馬影像

![海馬 1](data/untrained-samples/seahorse1.jpg?raw=true "海馬 1")
![海馬 2](data/untrained-samples/seahorse2.jpg?raw=true "海馬 2")
![海馬 3](data/untrained-samples/seahorse3.jpg?raw=true "海馬 3")

讓我們看看我們的三個訓練嘗試分別做得怎麼樣：

### 模型第一次訓練嘗試：從頭開始訓練的 AlexNet（第三名）

| 照片 | 海豚 | 海馬 | 結果 |
|-------|---------|----------|--------|
|[dolphin1.jpg](data/untrained-samples/dolphin1.jpg)| 71.11% | 28.89% | :expressionless: |
|[dolphin2.jpg](data/untrained-samples/dolphin2.jpg)| 99.2% | 0.8% | :sunglasses: |
|[dolphin3.jpg](data/untrained-samples/dolphin3.jpg)| 63.3% | 36.7% | :confused: |
|[seahorse1.jpg](data/untrained-samples/seahorse1.jpg)| 95.04% | 4.96% | :disappointed: |
|[seahorse2.jpg](data/untrained-samples/seahorse2.jpg)| 56.64% | 43.36 |  :confused: |
|[seahorse3.jpg](data/untrained-samples/seahorse3.jpg)| 7.06% | 92.94% |  :grin: |

### 模型第二次訓練嘗試: 微調後的 AlexNet（第二名）

| 照片 | 海豚 | 海馬 | 結果 |
|-------|---------|----------|--------|
|[dolphin1.jpg](data/untrained-samples/dolphin1.jpg)| 99.1% | 0.09% |  :sunglasses: |
|[dolphin2.jpg](data/untrained-samples/dolphin2.jpg)| 99.5% | 0.05% |  :sunglasses: |
|[dolphin3.jpg](data/untrained-samples/dolphin3.jpg)| 91.48% | 8.52% |  :grin: |
|[seahorse1.jpg](data/untrained-samples/seahorse1.jpg)| 0% | 100% |  :sunglasses: |
|[seahorse2.jpg](data/untrained-samples/seahorse2.jpg)| 0% | 100% |  :sunglasses: |
|[seahorse3.jpg](data/untrained-samples/seahorse3.jpg)| 0% | 100% |  :sunglasses: |

### 模型第三次訓練嘗試: 微調後的 GoogLeNet（第一名）

| 照片 | 海豚 | 海馬 | 結果 |
|-------|---------|----------|--------|
|[dolphin1.jpg](data/untrained-samples/dolphin1.jpg)| 99.86% | 0.14% |  :sunglasses: |
|[dolphin2.jpg](data/untrained-samples/dolphin2.jpg)| 100% | 0% |  :sunglasses: |
|[dolphin3.jpg](data/untrained-samples/dolphin3.jpg)| 100% | 0% |  :sunglasses: |
|[seahorse1.jpg](data/untrained-samples/seahorse1.jpg)| 0.5% | 99.5% |  :sunglasses: |
|[seahorse2.jpg](data/untrained-samples/seahorse2.jpg)| 0% | 100% |  :sunglasses: |
|[seahorse3.jpg](data/untrained-samples/seahorse3.jpg)| 0.02% | 99.98% |  :sunglasses: |

## 結論

我們的模型跑起來真的十分令人驚訝，微調一個事先訓練好的網絡之後的成效也是。很明顯的，我們使用海豚及海馬作為例子是故意設計好的，且我們的資料集也太有限了——如果我們希望我們的網絡變得很強大，我們真的會需要更多更好的資料。不過既然我們的目的是玩玩看類神經網絡的工具及工作流程，這個結果還是十分理想的，尤其是它不需要昂貴的設備或大量的時間。

我希望以上所有的經驗能讓你拋去所有剛踏進這個領域時產生的壓倒性恐懼。當你看過機器學習及類神經網絡實際運作的小例子之後，你應該能更容易地確定你是否值得在這個領域投入時間來學習它們背後的理論。現在你已經有了一個設定好的環境及一個可行的方法，你可以嘗試做做看其他類型的分類。你也可能會想要看看你還能用 Caffe 和 Digits 做什麼事情，例如在一個影像內尋找物件或是執行分割。

「Have fun with machine learning!」
