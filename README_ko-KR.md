# Have Fun with Machine Learning: 초보자를 위한 가이드

> Author: David Humphrey (original [English version](README.md))   
  

## 머리말

이것은 인공지능에 대한 *배경지식이 없는* 프로그래머들을 위한 머신러닝 **실습 가이드**입니다. 
인공지능 신경망을 사용하는 것은 박사학위를 필요로 하지 않으며, 여러분은 크게 발전할 필요없이
이미 있는 것을 *사용*하면 됩니다. 지금 우리가 가지고 있는 것은 충분히 유용합니다. 저는 이것을 
다른 오픈소스를 대하는 것처럼 더 많은 사람들이 갖고 놀아야 한다고 생각합니다.   

이 가이드에서 우리의 목표는 기계학습을 이용하여 [데이터/훈련되지않은 샘플들](data/untrained-samples)
속 이미지가 **돌고래**인지 **해마**인지 이미지만으로 정확성 있게 예측하는 프로그램을 작성하는 것입니다.
여기 우리가 사용할 두 가지 예제 사진들이 있습니다:

![A dolphin](data/untrained-samples/dolphin1.jpg?raw=true "Dolphin")
![A seahorse](data/untrained-samples/seahorse1.jpg?raw=true "Seahorse")

그러기 위해 우리는 [나선형 신경망(CNN)](https://en.wikipedia.org/wiki/Convolutional_neural_network)
을 훈련시키고 사용할 것입니다. 우리는 이것을 실무자의 관점 또는 첫 번째 원리의 관점에서 접근할 것입니다.
현재 인공지능에 많은 관심이 쏟아지고 있지만, 쓰여진 대부분은 공원의 친구가 아니라 물리학교수가 자전거로
트릭을 가르치는 것처럼 느껴집니다.

저는 이것을 블로그 게시물처럼 깃허브 VS.에 작성하기로 결정했습니다. 제가 밑에 쓴 것들 중 오해를 불러일으키거나
부족하거나 혹은 완전히 잘못된 부분이 있을 수 있습니다. 저는 아직 배워가는 중이고, 견고한 초보자용 문서가 없는 것이
장애물이라고 생각합니다. 실수가 있거나 중요한 세부사항이 누락된 것을 발견하셨다면, Pull request를 보내주십시오.

소개가 끝났으니, 여러분에게 자전거 트릭을 몇 가지 보여드리겠습니다!

## 개요

지금부터 살펴볼 내용은 다음과 같습니다:

* 특히 기존의 오픈 소스 머신러닝 기술을 설정하고 사용합니다.([Caffe](http://caffe.berkeleyvision.org/)와
 [DIGITS](https://developer.nvidia.com/digits))
* 이미지 데이터셋를 만듭니다.
* 신경망을 처음부터 훈련시킵니다.
* 본 적 없는 이미지로 신경망을 테스트합니다.
* 기존 신경망을 하게 튜닝 -*fine Tuning*- 해 신경망의 정확성을 향상시킵니다. (AlexNet 와 GoogLeNet)
* 신경망을 배포하고 사용합니다.

이 가이드는 신경망이 어떻게 설계되는지, 많은 이론을 다루거나, 수학적 표현을 사용하는 법을
가르쳐 주진 않습니다. 여러분에게 보여드릴 내용의 대부분을 이해한다고는 말하지 않겠습니다. 
대신 흥미로운 방식으로 기존의 것들을 사용해 어려운 문제를 해결해 나갈 것입니다.

> Q: "신경망의 이론에 대해서는 이야기하지 않는다고 말씀하셨습니다만, 앞으로 진행하기 전에
>  적어도 목차(overview)가 필요하다고 생각합니다. 어디서부터 시작해야할까요?"

이에 대한 소개는 짧은 게시물부터 온라인 전체 강좌까지 말 그대로 수백가지가 넘습니다. 여러분이
배우고 싶은 방법에 따라 좋은 출밤절을 위한 3가지 선택지가 있습니다.:

* 이 멋진 [블로그 게시물](https://jalammar.github.io/visual-interactive-guide-basics-neural-networks/) 
은 직관적인 예제들을 이용하여 신경망의 개념을 소개합니다.
* 비슷하게, [브랜든 로러](https://www.youtube.com/channel/UCsBKTrp45lTfHa_p49I2AEQ)가 소개하는 
[이 영상](https://www.youtube.com/watch?v=FmpDIaiMIeA) 은 우리가 사용하게 될 나선형 신경망에 대
한 좋은 소개입니다.
* 이론을 좀 더 알고 싶다면,  [마이클 닐슨](http://michaelnielsen.org/) 의 [온라인 책](http://neuralnetworksanddeeplearning.com/chap1.html) 을 추천합니다.

## 설정

사용할 소프트웨어(Caffe와 DIGITS)는 플랫폼 및 운영체제 버전에 따라 설치가 어려울 수 있습니다. 가장 쉬운 방법은 
도커(Docker)를 사용하는 것입니다. 아래에서 도커(Docker)로 하는 방법과 기본으로 설치하는 방법을 살펴봅시다.

### Option 1a: 네이티브하게 Caffe 설치

먼저, 우리는 버클리 비전 및 학습 센터의 [Caffe 딥러닝 프레임워크](http://caffe.berkeleyvision.org/)
를 사용할 것입니다.(BSD licensed)

> Q: “잠깐만요, 왜 Caffe죠? Tensorflow와 같은 것을 사용하는 것은 어떨까요?
> 요즘 모두가 말하는 것이잖아요...”  

좋은 선택지가 많이 있고, 여러분은 모든 선택지를 살펴봐야 합니다. [TensorFlow](https://www.tensorflow.org/)는
훌륭하고 여러분은 TensorFlow를 사용해도 좋습니다. 하지만 전 여러가지 이유로 Caffe를 사용하고 있습니다:

* 컴퓨터 비전 문제에 적격입니다. 
* C++, 파이썬을 지원합니다.([node.js 지원](https://github.com/silklabs/node-caffe) 예정)
* 빠르고 안정적입니다.

하지만 제가 Caffe를 사용하는 **첫번째 이유**는 **어떤 코드도 쓸 필요없기** 때문입니다. 여러분은 선언과 커맨드라인
도구로 모든 것을 할 수 있습니다.(Caffe는 구조화된 텍스트 파일을 사용하여 네트워크 아키텍처를 정의합니다.) 또한, 
여러분은 여러분의 네트워크를 더 쉽게 훈련하고 검증하기 위해 Caffe의 좋은 프론트 엔드들을 사용할 수 있습니다. 
우리는 [nVidia의 DIGITS](https://developer.nvidia.com/digits)도구를 이러한 목적으로 사용할 것입니다.

Caffe는 설치하기에 힘들 수 있습니다. 미리 만들어진 Docker와 AWS 구성을 포함하여 다양한 플랫폼에 대한 [설치 지침](http://caffe.berkeleyvision.org/installation.html)이 있습니다.

**NOTE:** 저는 Github repo에서 출시되지 않은 다음 버전의 Caffe를 사용했습니다:
https://github.com/BVLC/caffe/commit/5a201dd960840c319cefd9fa9e2a40d2c76ddd73

Mac에서는 버전 문제로 인해 빌드 내의 여러 단계에서 진행이 중단되어 작업을 시작하는 것이 어려울 수
있습니다. 이틀동안 시행착오를 겪었습니다. 여러 가이드를 따라해봤지만, 각각은 약간씩 다른 문제들을
가지고 있었습니다. 그 중 [이 가이드](https://gist.github.com/doctorpangloss/f8463bddce2a91b949639522ea1dcbe4)가
가장 가까웠습니다.
또한, [이 게시물](https://eddiesmo.wordpress.com/2016/12/20/how-to-set-up-caffe-environment-and-pycaffe-on-os-x-10-12-sierra/)을 추천합니다. 최근에 제가 봤던 많은 토론들과 연결되어 있습니다.  

Caffe 설치는 저희가 할 것들 중 가장 어려운 일입니다. 꽤 멋진 일이죠. AI쪽은 더 어려울 거라고 생각하셨을테니까요!
몇가지 문제를 겪으시더라도 포기하지마세요. 그것은 그럴 가치가 있습니다. 만약 제가 이 작업을 다시 수행한다면, Mac에서 직접 수행하지 않고 Ubuntu VM을 사용할 것입니다. 도움이 더 필요하시다면, [Caffe 사용자들](https://groups.google.com/forum/#!forum/caffe-users)그룹도 존재합니다.

> Q: “신경망을 훈련시키려면 강력한 장비가 필요할까요? 좋은 GPU에 접근할 수 
> 없다면 어떻게 해야할까요?"

사실 심층 신경망은 훈련시키기 위한 많은 연산능력과 에너지를 필요로 합니다.. 대규모 데이터셋을 이용해 처음부터 훈련시키는 경우라면 말입니다.
우리는 그렇게 하지 않을 거예요. 비결은 다른 사람이 이미 수백 시간에 걸쳐 훈련시켜논 사전 훈련된 신경망을 사용하여, 각자의 데이터셋에 맞게
미세하게 조정하는 것 -*Fine Tuning*-입니다. 아래에서 이 작업을 어떻게 하는 지 알아보겠지만, 제가 여러분에게 보여드릴 것은 최신 GPU가 탑재되지 않은 1년 
된 맥북 프로를 사용하고 있습니다. 

이와는 별도로, 전 통합 인텔 그래픽 카드와 엔비디아 GPU를 가지고 있기 때문에 [OpenCL Caffe branch]
(https://github.com/BVLC/caffe/tree/opencl)를 사용하기로 결정했고, 제 노트북에서 잘 작동했습니다. 

Caffe 설치가 완료되면 다음 작업을 수행하거나 수행해야 합니다:

*  빌드된 caffe가 포함된 디렉토리입니다. 표준으로 이 작업을 수행했다면, caffe, python 바인딩 등을 실행하는
 데 필요한 모든 것이 `build/` 디렉터로에 있을 것입니다. `build/`의 상위 디렉토리는 `CAFFE_ROOT`(나중에 필요)입니다. 
* `make test && make runtest` 는 실행하지 마십시오.
* 모든 python deps를 설치한 후(`python/`에서 `pip install -r requirements.txt` 실행), 
`make pycaffe && make pytest`는 실행하지 마십시오.
* 또한 `distribute/` 안에 있는 모든 필수적인 헤더, 바이너리 등을 포함하는 배포 가능한 버전의 caffe를 생성하려면
 `make distribute`를 실행해야 합니다. 

Caffe가 완전히 빌드된 컴퓨터에서, CARRE_ROOT 디렉토리는 다음과 같은 기본 레이아웃을 따릅니다:

```
caffe/
    build/
        python/
        lib/
        tools/
            caffe ← 메인 바이너리입니다.
    distribute/
        python/
        lib/
        include/
        bin/
        proto/
```

이 시점에서 우리는 신경망으로 훈련, 테스트 및 프로그래밍하는 데 필요한 모든 것을 갖추고 있습니다. 다음 섹션에서는
사용자 친화적인 웹 기반 프론트 엔드를 DIGITS라고 불리는 caffe에 추가하여 신경망을 훨씬 쉽게 교육하고 테스트할
수 있습니다.

### Option 1b: 네이티브하게 DIGITS 설치

nVidia의 [딥러닝 GPU 훈련시스템(DIGITS)](https://github.com/NVIDIA/DIGITS)는 신경망 훈련을 위한
BSD 라이선스의 python 웹 앱입니다. 커맨드 라인이나 코드로 DIGITS가 Caffe에서 하는 모든 작업들을 실행할 수
있지만, DIGITS를 사용하면 훨씬 쉽게 시작할 수 있습니다. 또한 뛰어난 시각화, 실시간 차트 및 기타 그래픽 기능들으로
인해 더 재미있을 것입니다.  배우기 위해선 경험을 쌓고 도전해봐야 하기 때문에 DIGITS로 시작하는 것을 추천합니다. 

https://github.com/NVIDIA/DIGITS/tree/master/docs 에 
[Installation](https://github.com/NVIDIA/DIGITS/blob/master/docs/BuildDigits.md)(설치),
[Configuration](https://github.com/NVIDIA/DIGITS/blob/master/docs/Configuration.md)(구성),
및 [Getting Started](https://github.com/NVIDIA/DIGITS/blob/master/docs/GettingStarted.md)
(시작) 페이지들을 포함하는 좋은 문서들이 꽤 있습니다. 전 DIGITS의 모든 것들을 잘 아는 전문가가 아니기 때문에 계속하기
전에 자세하게 읽어보는 걸 추천합니다. 도움이 더 필요하시다면,  공개 [DIGITS 사용자 그룹](https://groups.google.com/forum/#!forum/digits-users)
도 있습니다.

Docker부터 리눅스에서 패키지들을 pre-baked하거나 소스에서 빌드하기까지, DIGITS를 설치하고 실행하는 데에는 다양한
방법이 있습니다. 저는 Mac을 사용하고 있으므로 소스에서 빌드했습니다.

**NOTE:** 이 가이드에선 Github repo에서 출시되지 않은 DIGITS의 다음 버전을 사용했습니다 : https://github.com/NVIDIA/DIGITS/commit/81be5131821ade454eb47352477015d7c09753d9

python 스크립트 묶음이기 때문에 작업하는 것은 힘들지 않았습니다. 여러분이 해야 할 것은 서버를
시작하기 전에 환경 변수를 설정하여 `CAFFE_ROOT`의 위치를 DIGITS에 알려주는 것입니다:

```bash
export CAFFE_ROOT=/path/to/caffe
./digits-devserver
```

NOTE: Mac에서 파이썬 바이너리가 `pyhon2`라고 가정하고 서버 스크립트에 문제가 있었는데, 여기서 저는
`python2.7`만 가지고 있었습니다. 이것은 `/usr/bin`에서 심볼릭링크로 접근하거나 DIGITS 부팅시
스크립트를 조정하여 해결할 수 있습니다. 서버가 시작되면 http://localhost:5000 에 웹 브라우저를 
통해 밑에서 다룰 모든 작업을 수행할 수 있습니다.

### Option 2: Docker를 사용한 Caffe와 DIGITS 

[Docker](https://www.docker.com/)를 설치하고(설치되어 있지 않은 경우) 전체 Caffe + Digits 
컨테이너를 꺼내기 위해 다음 명령을 실행합니다. 몇 가지의 주의할 사항:
* 포트 8080이 다른 프로그램에 할당되진 않았는지 확인하십시오. 만약 그렇다면, 임의의 다른 포트로
 변경하십시오.
* 이 복제(clone)된 repo의 위치를 `/path/to/this/repostiory`로 옮기십시오. 그러면 컨테이너
 내의 `/data/repo`가 이 디렉토리에 바인딩됩니다. 이것은 아래 설명된 이미지에 접근하는 데
 유용합니다. 

```bash
docker run --name digits -d -p 8080:5000 -v /path/to/this/repository:/data/repo kaixhin/digits
```

이제 컨테이너가 실행 중이므로 우리는 웹 브라우저를 열고 `http://localhost:8080`에 접근할 수 있습니다.
이 레포지토리의 모든 내용은 이제 컨테이너 디렉토리 `/data/repo`에 있습니다. 이제 다 했습니다. 이제 Caffe
와 DIGITS가 실행되고 있습니다. 
셸에 접근이 필요한 경우, 다음 명령을 따라하십시오:

```bash
docker exec -it digits /bin/bash
```

## 신경망 훈련

신경망을 훈련시키는 것은 몇 가지 단계를 수반합니다:

1. 분류된 이미지의 데이터셋를 구성하고 준비하십시오
2. 신경망의 아키텍처를 규정하십시오
3. 준비된 데이터셋를 사용해 이 신경망을 훈련시키고 검증하십시오.

처음부터 시작하는 것과 사전훈련된 신경망을 사용하는 것의 차이를 보여주고 Caffe와 DIGITs에서 흔히  
사용되는 두 가지 인기 있는 사전훈련된 신경망(AlexNet, GoogLeNet)에서 어떻게 실행하는 지 보여주기
위해 우리는 이러한 3단계를 거칠 것입니다. 

우리는 훈련 시도에 돌고래와 해마의 작은 데이터셋를 사용할 것입니다. [data/dolphins-and-seahorses](data/dolphins-and-seahorses)에 제가 사용했던 이미지들을 넣어두었습니다. 2개 이상의 카테고리가 필요하고 여러분은 더
많은 카테고리들을 가질 수도 있습니다(사용할 신경망 중 일부는 1000개 이상의 이미지 카테고리에 대해 
훈련되었습니다). 우리의 목표는 우리의 신경망에 이미지를 주고 그것이 돌고래인지 해마인지 우리에게 
알려주게하는 것입니다.

### 데이터셋 준비

가장 쉬운 방법은 이미지들을 분류된 디렉토리 배치로 나누는 것입니다.:

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

여기 각 디렉토리는 분류할 카테고리이며, 해당 카테고리 디렉토리 내의 각 이미지는 훈련 및 검증에
사용할 예제입니다. 

> Q: “이미지들의 사이즈가 같아야 하나요? 파일명은 어떻게 하죠, 그게 중요한가요?”

둘 다 아닙니다. 이미지 크기는 우리가 신경망에 입력하기 전에 정규화될 것입니다. 우리는 마지막엔
256 x 256 픽셀의 컬러 이미지를 사용하겠지만, DIGITS는 이미지를 자동으로 자르거나 스쿼시할 
것입니다. 파일 이름은 관련이 없습니다--어떤 카테고리에 포함되느냐가 중요할 뿐입니다.  

> Q: “제 카테고리들을 더 복잡하게 세분화해도 되나요?”

네. https://github.com/NVIDIA/DIGITS/blob/digits-4.0/docs/ImageFolderFormat.md 를 참고하세요.

우리는 이 이미지를 디스크에 사용하여 **New Dataset**, 그 중에서도 **Classification Dataset**를
생성하려고 합니다.

![Create New Dataset](images/create-new-dataset.png?raw=true "Create New Dataset")

DIGITS가 제공하는 기본 설정값을 사용하고 [data/dolphins-and-seahorses](data/dolphins-and-seahorses) 
폴더 경로에 **Training Images**를 지정합니다. DIGITS는 카테고리(`돌고래`와 `해마`)를 사용하여 
스쿼시된 256 x 256 Training (75%) 및 Testing (25%) 이미지의 데이터베이스를 만듭니다. 

Dataset에 `dolphins-and-seahorses`라는 이름을 지정하고, **Create**를 클릭합니다.

![New Image Classification Dataset](images/new-image-classification-dataset.png?raw=true "New Image Classification Dataset")

이제 데이터셋이 생성될 것입니다. 제 노트북에선 4초만에 생성되었죠. 마지막으로 2개의 카테고리 속 92개의 
훈련 이미지 -*Training images*- (돌고래 49개, 해마 43개)와 30개의 검증 이미지 -*Validation images*- (돌고래 16개,
해마 14개)가 있습니다. 이것은 매우 작은 데이터셋이지만, 신경망을 훈련하고 검증하는 데 오랜 시간이 
걸리지 않기 때문에 우리의 활동과 학습 목적에 알맞습니다.

이미지가 스쿼시된 후 이미지를 확인하려면 **DB 탐색** -*Explore DB*- 을 하면 됩니다.

![Explore the db](images/explore-dataset.png?raw=true "Explore the db")

### 훈련: 시도 1, 처음부터

DIGITS 홈 화면으로 돌아가서, 우리는 새로운 **분류 모델** -*Classification Model*- 을 생성해야 합니다:

![Create Classification Model](images/create-classification-model.png?raw=true "Create Classification Model")

우리는 우리의 `dolphins-and-seahorses` 데이터셋과 DIGITS가 제공하는 기본 설정값을 
사용하는 모델을 훈련시키는 것부터 시작할 것입니다 첫번째 신경망으로는 표준 신경망 
아키텍처 중 하나인 [AlexNet (pdf)](http://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf)
을 사용할 것입니다. [AlexNet 설계](http://vision.stanford.edu/teaching/cs231b_spring1415/slides/alexnet_tugce_kyunghee.pdf)
는 2012년 ImageNet이라는 큰 컴퓨터 비전 대회에서 우승했습니다. 이 대회는 120만
개의 이미지에 걸쳐 1000개 이상의 이미지 카테고리를 분류해야 했습니다. 
 
![New Classification Model 1](images/new-image-classification-model-attempt1.png?raw=true "Model 1")

Caffe는 구조화된 텍스트 파일을 사용해 신경망 아키텍처를 정의합니다. 이러한 텍스트 파일은
[Google의 프로토콜 버퍼](https://developers.google.com/protocol-buffers/)를 기반으로 합니다.
여러분은 Caffe가 사용하는 [전체 도식](https://github.com/BVLC/caffe/blob/master/src/caffe/proto/caffe.proto)
을 읽어보실 수 있습니다.
대부분의 파트에서 우리는 이것들을 사용하지 않겠지만, 나중에 그것들을 수정해줘야 하기 때문에 
이러한 것들이 있다는 것을 알아두는 것이 좋습니다.
For the most part we’re not going to work with these, but it’s good to be aware of their
existence, since we’ll have to modify them in later steps. AlextNet protxt 파일은 다음과 
같습니다: https://github.com/BVLC/caffe/blob/master/models/bvlc_alexnet/train_val.prototxt. 

**epoch는 30**으로 신경망을 훈련시킵니다. 즉, 신경망이 학습(우리의 training image를 통해)하면 
자체적으로 테스트(validation image을 사용해)하며 결과에 따라 신경망의 가중치를 조정하하는 것을 
30번 반복합니다. 한 사이클이 완료될 때마다 **Accuracy** (0% ~ 100%, 높을수록 좋은 값)와 
**Loss**가 얼마인지(발생한 모든 오류의 합계, 낮을 수록 좋은 값)에 대한 정보를 얻을 수 있습니다.
이상적으로는 우리는 오류(작은 손실 -*loss*-)이 거의 없는, 매우 정확하게 예측할 수 있는 신경망을
원합니다.

**NOTE:** 몇몇 사람들이 이 훈련을 시키면서 [DIGITS에서 hit 오류가 일어났다](https://github.com/humphd/have-fun-with-machine-learning/issues/17)
고 보고했습니다. 대부분의 경우, 이 문제는 가용 메모리와 관련된 것입니다(프로세스가 실행하려면 많은
메모리가 필요합니다). 여러분이 Docker를 사용하고 있다면, DIGITS에서 사용할 수 있는 메모리 양을 
늘릴 수 있습니다. (Docker에서, 환경설정 -*preferences*- -> 고급 -*preferences*- -> 메모리 -*preferences*- )

처음엔, 우리 신경망의 정확도는 50% 미만입니다. 원래 이렇습니다. 처음에는 무작위로 할당된 가중치를 
사용하여 두 카테고리 중 "추측"하는 것이기 때문입니다. 시간이 지남에 따라 0.37의 loss로 87.5%의 
정확도를 달성할 수 있습니다. 전체 30 epoch까지 전 6분도 채 걸리지 않았습니다. 

![Model Attempt 1](images/model-attempt1.png?raw=true "Model Attempt 1")

우리가 업로드한 이미지나 웹 상의 이미지에 URL을 사용하여 우리의 모델을 테스트할 수 있습니다.
훈련/검증 데이터셋에 없는 몇 가지 예제를 통해 테스트해 보겠습니다.

![Model 1 Classify 1](images/model-attempt1-classify1.png?raw=true "Model 1 Classify 1")

![Model 1 Classify 2](images/model-attempt1-classify2.png?raw=true "Model 1 Classify 2")

거의 완벽해 보입니다. 다음 시도를 하기 전까지는 말이죠:

![Model 1 Classify 3](images/model-attempt1-classify3.png?raw=true "Model 1 Classify 3")

여기서 완전히 실패합니다. 해마를 돌고래로 착각하는데, 최악인 것은 높은 자신감으로 해마라고 합니다.  

현실은 우리의 데이터셋이 너무 작아 정말 좋은 신경망을 훈련시키는 데에는 쓸만 하지 않다는 것입니다.
모든 것을 처리하기 위해선 높은 연산능력과 10초에서 100초 정도의 수천 개의 이미지들이 필요합니다.

### 훈련: 시도 2, AlexNet Fine Tuning

#### Fine Tuning 하는 법 

신경망을 처음부터 설계하고, 훈련하기에 충분한 데이터(e.g. 수백만의 이미지)를 수집하고,
훈련을 완료하기 위해 몇 주 동안 GPU에 엑세스하는 것은 우리가 하기엔 벅찹니다. 더 적은 
양의 데이터로도 사용될 수 있도록 우리는 **Transfer Learning** 또는 **Fine Tuning**이라는 
기술을 사용할 것입니다. Fine Tuning은 심층 신경망의 레이아웃을 활용하고 사전훈련된 신경망을
이용해 첫번째 객체 감지 작업을 수행합니다. 

쌍안경으로 멀리 있는 것을 보는 것처럼 신경망을 사용하는 것을 상상해보십시오. 먼저,
쌍안경을 눈에 대보면 모든 게 흐릿해집니다. 초점을 맞추면, 색깔, 선, 모양이 보이기 
시작하고 마지막엔 새의 형태를 인식할 수 있게 됩니다. 그리고 조금 더 조정한다면 새의
종까지 구분해낼 수 있게 됩니다.  

다중 계층 신경망에서, 초기 계층은 특징(e.g. 가장자리)을 추출하고, 다음 계층은 형태
(e.g. 바퀴, 눈)를 알아내기 위해 이러한 특징들을 사용합니다. 즉, 이전 계층들의 누적된 특성을
기반으로 각각의 항목들을 분류하는 최종 분류 계층에 반영됩니다(e.g. 고양이 vs. 개). 신경망은 
픽셀 단위에서 직사각형으로, 다리로, 특정 방향으로 걷는 두 개의 다리까지 인식할 수 
있어야 하며, 마지막엔 이미지가 고양이를 가리킨다는 결론을 내릴 수 있어야 합니다.

우리가 하고자 하는 것은 기존에 훈련되어있던 이미지 클래스 대신 새로운 이미지 클래스 세트로
분류하기 위해 사전훈련된 기존 신경망을 전문적으로 다루는 것입니다. 신경망은 이미 이미지의 
특징을 "인식"하는 법을 알고 있으므로 특정한 이미지 형태로 "인식"하기 위해 우리가 신경망을 
재훈련하고자 합니다. 계층들의 대부분은 처음부터 시작할 필요가 없습니다--이런 계층에서 이미 
수행했던 학습을 새로운 분류 작업으로 이전하고자 합니다. 랜덤한 가중치를 사용했던 이전 
시도와는 달리, 우리는 최종 신경망의 기존 가중치를 훈련시키는 데 사용할 것입니다. 그러나 
우리는 최종 분류 계층을 버리고, *우리의* 이미지 데이터셋을 사용해 신경망을 재교육하여 
이미지 클래스에 맞게 미세 조정 -*fine tuning*- 할 것입니다.

이것이 실행되기 위해서는 학습된 가중치가 쓸만할 만큼 우리의 데이터와 충분히 비슷한 결과가 나오는
사전훈련된 신경망이 필요합니다. 다행히도 우리가 아래에서 사용할 신경망은 [ImageNet](http://image-net.org/)
의 수백만 개의 자연이미지로 훈련되었으며, 광범위한 분류 작업에 뛰어난 성능을 보입니다. 

이 테크닉은 의학이미지에서 눈병을 검사하고, 바다에서 수집한 현미경이미지에서 플랑크톤 종을 
식별하며, Flickr 이미지의 미술 양식을 분류하는 것과 같은 흥미로운 일들을 하는데 사용되어 
왔습니다. 

모든 머신러닝과 마찬가지로 이 작업을 완벽하게 수행하려면 데이터 및 신경망 아키텍처를 이해해야 
합니다--데이터의 과적합에 주의해야 하며 일부 계층을 수정해야 하거나 새 계층을 삽입해야 하는 
경우도 있습니다. 하지만, 제 경험상, 대부분의 경우에 "단지 작동"할 뿐이며 그저 경험을 쌓고
우리의 단순한 접근법을 사용하여 무엇을 달성할 수 있는지 확인하는 것만으로 가치있습니다.

#### 사전훈련된 신경망 업로드

첫 번째 시도에서는 Alexnet의 아키텍처를 사용했지만, 신경망 계층에서 랜덤한 가중치로 시작했습니다.
우리는 대규모 데이터셋에 대해 이미 훈련받은 버전의 AlexNet을 다운로드하고 사용하고자 합니다.

다행히도 우리는 이것을 바로 할 수 있습니다. AlexNet의 스냅샷은 여기서 다운로드할 수 있습니다: https://github.com/BVLC/caffe/tree/master/models/bvlc_alexnet.
우리는 훈련된 가중치를 포함하고 있는 이진 파일 `.caffemodel` 도 필요하고, http://dl.caffe.berkeleyvision.org/bvlc_alexnet.caffemodel 에서 다운로드할 수 있습니다. 
 
사전훈련된 모델을 받는 동안, 하나 더 해봅시다. 
2014년에 Google은 [GoogLeNet](https://research.google.com/pubs/pub43022.html)으로 같은 
ImageNet 대회에서 우승했습니다 (코드명 Inception):
22계층의 신경망, GoogLeNet의 스냅샷도 다운로드할 수 있습니다. https://github.com/BVLC/caffe/tree/master/models/bvlc_googlenet 을 참조하십시오.
다시 말하지만, 우리는  http://dl.caffe.berkeleyvision.org/bvlc_googlenet.caffemodel 에서 다운로드
할 수 있는 모든 사전훈련된 가중치들로 구성된 `.caffemodel` 파일이 필요합니다.

우리는 `.caffemodel` 파일을 가지고 DIGITs에 업로드할 수 있습니다. DIGITS 홈페이지에 
**Pretrained Models** 탭으로 이동하여 **Upload Pretrained Model**을 클릭합니다:

![Load Pretrained Model](images/load-pretrained-model.png?raw=true "Load Pretrained Model")

이러한 사전훈련된 두 모델은 모두 DIGITS가 제공하는 기본설정값을 사용할 수 있습니다(i.e. 
256 x 256의 스쿼시된 컬러 이미지). 
For both of these pretrained models, we can use the defaults DIGITs provides
(i.e., colour, squashed images of 256 x 256). 우리는 `가중치 -Weights- (**.caffemodel)`
및 ` 모델 정의 -Model Definition- (original.prototxt)`만 제공하면 됩니다.
각 버튼을 클릭하여 파일을 선택하십시오.

모델 정의(model definitions)에 대해서는 GoogLeNet의 경우에는 https://github.com/BVLC/caffe/blob/master/models/bvlc_googlenet/train_val.prototxt 을 
참조하고 AlexNet의 경우에는 https://github.com/BVLC/caffe/blob/master/models/bvlc_alexnet/train_val.prototxt
을 참조할 수 있습니다. 우리는 이러한 분류 레이블들을 사용하지 않을 것이므로 `labels.txt` 파일 추가는
생략하겠습니다:

![Upload Pretrained Model](images/upload-pretrained-model.png?raw=true "Upload Pretrained Model")

다음 단계에서 AlexNet과 GoogLeNet을 모두 사용할 것이므로 이 과정을 반복하십시오.

> Q: "미세 조정 -fine tuning- 의 기반으로 적합한 다른 신경망이 있을까요?"

[Caffe Model Zoo](http://caffe.berkeleyvision.org/model_zoo.html) 는 다른 사전훈련된 
신경망들을 꽤 많이 가지고 있습니다. https://github.com/BVLC/caffe/wiki/Model-Zoo 을 
참조하십시오.

#### 돌고래와 해마로 AlexNet을 미세 조정하기 -Fine Tuning-

사전훈련된 Caffe 모델을 사용하여 신경망을 훈련하는 것은 몇 가지 조정을 해야하지만, 처음부터 
시작하는 것과 비슷합니다. 먼저, 이렇게 크게 변화할 필요가 없으므로(즉, *미세*하게 조정 
중입니다.) **기본 학습 속도 -Base Learning Rate-** 를 0.01에서 0.001로 조정합니다. 우리는 
또한 **사전훈련된 신경망 -Pretrained Network-** 을 사용하여 **커스터마이징 -Customize-** 할 
것입니다.

![New Image Classification](images/new-image-classification-model-attempt2.png?raw=true "New Image Classification")

사전훈련된 모델의 정의(i.e. prototext)에서는 모든 참조의 이름을 **완전히 연결된 계층-*Fully Connected Layer*-**(최종 
결과 분류가 이루어지는 곳)로 변경해야 합니다. 모델이 원래의 훈련 데이터와 비교해 새로운 
카테고리를 다시 학습하기를 원하기 때문입니다(즉, 현재의 마지막 계층은 폐기하고자 합니다). 
우리는 최종적으로 완전히 연결된 계층-*fully connected layer*-의 이름을 변경해야만 합니다. 
예를 들면, "fc8"에서 "fc9"로 말입니다. 마지막으로, 우리는 또한 `num_output`을 `2`로 변경하여, 
카테고리 수를 `1000`에서 `2`로 조정해야 합니다. 

여기 우리가 변경해야 할 사항이 있습니다:

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

제가 사용하고 있는 완전히 수정된 파일을 [src/alexnet-customized.prototxt](src/alexnet-customized.prototxt)
에 포함했습니다.

이번에는 정확도가 ~60%에서 시작해 87.5%로 급등하며 96%까지 이윽고 100%까지 상승하며, 
손실 *Loss* 은 꾸준히 감소했습니다. 5분이 지나면 100%의 정확도와 0.0009의 손실이 발생합니다.

![Model Attempt 2](images/model-attempt2.png?raw=true "Model Attempt 2")

이전 신경망이 오류를 일으킨 것과 같은 해마 이미지를 테스트한 결과, 우리는 완전한 반전을 
볼 수 있습니다: 100% 해마!

![Model 2 Classify 1](images/model-attempt2-classify1.png?raw=true "Model 2 Classify 1")

심지어 어린이들이 그린 해마 이미지에도 효과가 있습니다:

![Model 2 Classify 2](images/model-attempt2-classify2.png?raw=true "Model 2 Classify 2")

돌고래도 마찬가지입니다:

![Model 2 Classify 3](images/model-attempt2-classify3.png?raw=true "Model 2 Classify 3")

이처럼 여러 마리의 돌고래들이 서로 가까이 붙어 있고, 그들의 몸 대부분이 물 속에 잠겨 있어 식별하기에
어려워보이는 이미지들임에도 불구하고, 잘 작동됩니다:

![Model 2 Classify 4](images/model-attempt2-classify4.png?raw=true "Model 2 Classify 4")

### 훈련: 시도 3, GoogLeNet 미세 조정-*Fine Tuning*-

우리가 미세 조정-*Fine Tuning*-을 위해 사용했던 이전의 AlexNet 모델과 마찬가지로, GoogLeNet도 
사용할 수 있습니다. 신경망을 수정하는 것은 하나의 계층이 아니라 3개의 완전히 연결된 계층을 
재정의해야 하기 때문에 좀 더 까다롭습니다. 

우리의 유스케이스에 맞게 GoogLeNet을 미세 조정하려면, 우리는 또 다시 새로운 **분류 모델-*Classification Model*-** 
을 만들어야 합니다:

![New Classification Model](images/new-image-classification-model-attempt3.png?raw=true "New Classification Model")

완전히 연결된 세 가지 분류 계층인 `loss1/classifier`, `loss2/classifier`, `loss3/classifier`의 
모든 참조들의 이름을 변경하고 카테고리 수를 재정의합니다(`num_output: 2`). 여기에 3개의 분류 계층의 
이름을 변경하고 카테고리 수를 1000개에서 2개로 변경하기 위해 해야할 사항들이 있습니다:

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

전체파일을 [src/googlenet-customized.prototxt](src/googlenet-customized.prototxt)에 저장했습니다.

> Q: "이러한 신경망의 prototext 정의를 변경하는 건 어떻게 하나요?
> 우리는 완전히 연결된 계층의 이름과 카테고리의 수를 변경해보았습니다.
> 그 밖에 어떤 것이 변경될 수 있으며 어떤 상황에서 변경되어야 하나요?

좋은 질문입니다. 그건 저도 궁금한 것입니다. 예를 들어, 저는 가중치가 변하지 않도록 [특정 계층을 "수정"](https://github.com/BVLC/caffe/wiki/Fine-Tuning-or-Training-Certain-Layers-Exclusively)
할 수 있다는 것을 알고 있습니다. 그 밖에 다른 것들을 하는 것은 계층들이 어떻게 작동하는지 
이해해야 합니다. 이것은 이 안내서를 넘어선 일이고, 지금의 저자도 넘어서는 것입니다!

앞에서 했던 AlexNet 미세 조정과 마찬가지로, 학습률을 `0.01`에서 `0.001`로 10% 낮춥니다.

> Q: "이러한 신경망을 미세 조정할 때 그 외에 어떤 변경이 의미가 있나요?
> 다른 epoch 수, batch size,  솔버 유형 (Adam, AdaDelta, AdaGrad 등), 학습률, 정책
> (Exponential Decay, Inverse Decay, Sigmoid Decay 등), 단계 크기, 감마 값은 어떻나요?"

좋은 질문이고 마찬가지로 저도 궁금한 것들입니다. 저는 이것들에 대해 막연하게 이해하고 있으며, 
훈련시 이러한 값들을 어떤 식으로 변경해야할지 안다면 개선할 수 있을 것입니다. 물론 이보다 
더 좋은 문서를 필요로 할 것입니다.

GoogLeNet은 architecture보다 더 복잡한 아키텍처이므로 미세 조정에 더 많은 시간이 필요합니다.
제 노트북에서는 데이터셋으로 GoogLeNet을 재훈련시키는데 10분이 소요되어 100% 정확도와 0.0070의 
손실을 달성했습니다:

![Model Attempt 3](images/model-attempt3.png?raw=true "Model Attempt 3")

AlexNet의 미세 조정에서 살펴본 것처럼, 수정된 GoogLeNet은 잘 작동합니다--지금까지 중 가장 뛰어난 성능:

![Model Attempt 3 Classify 1](images/model-attempt3-classify1.png?raw=true "Model Attempt 3 Classify 1")

![Model Attempt 3 Classify 2](images/model-attempt3-classify2.png?raw=true "Model Attempt 3 Classify 2")

![Model Attempt 3 Classify 3](images/model-attempt3-classify3.png?raw=true "Model Attempt 3 Classify 3")

## 모델 사용

신경망을 훈련하고 테스트하였으니, 이제 다운받아 사용할 시간입니다. DIGITS로 훈련한 각 모델은 **Download Model** 버튼과 훈련 실행 중 서로 다른 스냅샷을 선택하는 방법이 있습니다(e.g. `Epoch #30`):

![Trained Models](images/trained-models.png?raw=true "Trained Models")

**Download Model** 를 클릭하면 다음 파일들이 압축된 `tar.gz` 파일이 다운로드됩니다:

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

Caffe 문서에 우리가 방금 만든 모델의 사용법에 대한 [멋진 설명](https://github.com/BVLC/caffe/wiki/Using-a-Trained-Network:-Deploy)
이 있습니다. 다음과 같이 쓰여 있습니다:

> 신경망은 구조(.prototxt)와 가중치로(.caffemodel) 정의됩니다. 신경망이 훈련될 때
> 가중치의 현재 상태-*current state*-는 .caffemodel에 저장됩니다. 이 두 가지를 통해 
> 우리는 훈련/테스트 단계에서 생산-*production*- 단계로 이동할 수 있습니다.
> 
> 현재 상태로서는 신경망의 구조는 배포용으로 설계되어 있지 않습니다. 신경망을 제품으로 
> 출시하기 전에 몇 가지 방법으로 신경망을 수정해야합니다:
>
> 1. 분류-*classification*-에 관해서 데이터의 레이블을 더이상 제공하지 않으므로 훈련에 사용된 데이터 계층을 제거하십시오
> 2. 데이터 레이블에 종속된 계층을 제거하십시오.
> 3. 데이터를 수신하도록 신경망을 설정하십시오.
> 4. 신경망이 결과를 출력하게 하십시오.

DIGITS는 `prototxt` 파일의 각각 다른 버전들을 구분하여 이미 할 일을 끝냈습니다.
신경망을 사용할 때 주의해야할 파일:

* `deploy.prototxt` - 이미지 입력 데이터를 받아들일 준비가 된 신경망의 정의
* `mean.binaryproto` - 모델이 처리하는 각각의 이미지에서 빼야할 이미지가 있는데, 그 빼야할 이미지를 말한다.
* `labels.txt` - 출력하고자 하는 레이블 (`dolphin`, `seahorse`)과 카테고리 번호만 출력하는 경우를 위한 목록
* `snapshot_iter_90.caffemodel` - 이것들은 우리 신경망을 위해 훈련된 가중치들이다.

우리는 이 파일들을 새로운 이미지로 분류하기 위해 다양한 방법들을 사용할 수 있습니다. 예를 들어, 
`CAFFE_ROOT`에서는 `build/examples/cpp_classification/classification.bin`을 사용해 하나의 
이미지를 분류할 수 있습니다:

```bash
$ cd $CAFFE_ROOT/build/examples/cpp_classification
$ ./classification.bin deploy.prototxt snapshot_iter_90.caffemodel mean.binaryproto labels.txt dolphin1.jpg
```

이러면 디버그 텍스트 다발들을 뱉어내고, 이어서는 두 카테고리에 대한 예측이 뒤따를 것입니다:

```
0.9997 - “dolphin”
0.0003 - “seahorse”
```

[전체 C++ 소스](https://github.com/BVLC/caffe/tree/master/examples/cpp_classification)는 
[Caffe 예제들](https://github.com/BVLC/caffe/tree/master/examples)에서 확인할 수 있습니다.

Python 인터페이스를 사용하는 분류 버전의 경우, DIGITS에 [좋은 예제](https://github.com/NVIDIA/DIGITS/tree/master/examples/classification)
가 있습니다.또한 Caffe 예제들 안에는 [꽤 잘 문서화된 파이썬 워크스루](https://github.com/BVLC/caffe/blob/master/examples/00-classification.ipynb) 
도 있습니다.

### 파이썬 예제

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
