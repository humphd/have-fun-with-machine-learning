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
* 기존 신경망을 미세하게 튜닝해 신경망의 정확성을 향상시킵니다. (AlexNet 와 GoogLeNet)
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
미세하게 조정하는 것입니다. 아래에서 이 작업을 어떻게 하는 지 알아보겠지만, 제가 여러분에게 보여드릴 것은 최신 GPU가 탑재되지 않은 1년 
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

**NOTE:** 몇몇 사람들이 이 훈련을 시키면서 [DIGITS에서 hit 오류가 보고](https://github.com/humphd/have-fun-with-machine-learning/issues/17)
했습니다. 대부분의 경우, 이 문제는 가용 메모리와 관련된 것입니다(프로세스가 실행하려면 많은
메모리가 필요합니다). 여러분이 Docker를 사용하고 있다면, DIGITS에서 사용할 수 있는 메모리 양을 
늘릴 수 있습니다. (Docker에서, 환경설정 -*preferences*- -> 고급 -*preferences*- -> 메모리 -*preferences*- )

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

![Trained Models](images/trained-models.png?raw=true "Trained Models")

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