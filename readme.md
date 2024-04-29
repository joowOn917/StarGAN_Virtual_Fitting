<img src="https://github.com/joowOn917/StarGAN_Virtual_Fitting/assets/143769249/19426e87-bccf-4c93-b5f8-1aa0f74f3545" >

## 🖥 Overview 
StarGAN 모델을 활용하여 피팅이 불가능한 오프라인 쇼핑몰이나 온라인 쇼핑몰에서 사용할 수 있는 가상 의류 피팅 서비스 개발 및 제안 

사람의 전신 이미지를 받으면 긴팔, 반팔, 후드, 크롭탑, 민소매 5종류의 의상으로 변환하는 기능을 구현하였으며, 오픈소스인 starGAN 코드를 사용할 이미지 데이터에 적합하게 수정하면서 프로젝트를 진행하였습니다. 

## 🛠 Requirement 
- Python 3.7.9
- Pytorch 1.21.1
- nvidia 11.6
- tensorflow
- numpy 
- matplotlib

## ⚙ Project Process

<img src = "https://github.com/joowOn917/StarGAN_Virtual_Fitting/assets/143769249/8e3cdae5-6982-490d-8019-1561380ecac7" height="80%">


### 1️⃣ 참고 논문

- Garment Style Creator: Using StarGAN for Image-to-Image Translation of Multidomain Garments, Chien-Hsing Chou
- StarGAN: Unified Generative Adversarial Networks for Multi-Domain Image-to-Image Translation, Yunjey Choi

### 2️⃣ 데이터 수집 (crwaling.py)

논문에 소개된 celebA 데이터셋이 아닌 새로 이미지를 수집

selenium 라이브러리를 활용하여 네이버와 구글에서 이미지 크롤링 (총 12,327개 이미지 수집)

기존 코드에 맞추기 위해 수집한 이미지들을 'celebA' 상위 폴더에 카테고리별 하위 폴더에 분류하여 저장

<img src="https://github.com/joowOn917/StarGAN_Virtual_Fitting/assets/143769249/97093b37-30d2-4ae3-97e0-bed05eb5228f">

### 3️⃣ 데이터 전처리 (data_loader.py & labeling.py)
    
#### [data_loader.py]

동일 이미지의 위치, 밝기, 색도 등과 같은 요소들을 무작위로 조절하면서 다양한 데이터를 생성하는 작업

주어진 소스 코드 내의 CenterCrop 작업은 사용할 의상 이미지에 적합하지 않아 주석 처리하여 진행하였다. 

    def get_loader(image_dir, attr_path, selected_attrs, crop_size=178, image_size=128, 
               batch_size=16, dataset='CelebA', mode='train', num_workers=0):
    """Build and return a data loader."""
    transform = []
    if mode == 'train':
        transform.append(T.RandomHorizontalFlip())
    # transform.append(T.CenterCrop(crop_size))
    transform.append(T.Resize((128,128)))
    transform.append(T.ToTensor())
    transform.append(T.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)))
    transform = T.Compose(transform)

#### [labeling.py]

이미지에 해당하는 도메인에 적합한 라벨링 진행

도메인에 해당하는 이미지(1)     해당하지 않는 이미지(0)

    for i in range(len(names)):

        check = df['file_name'][i]
        
        if check.startswith('sleeve'):
            sleeveless.append(1)
        else:
            sleeveless.append(-1)
        
        if check.startswith('long'):
            longsleeve.append(1)
            longsleeve.append(-1)

<img src = "https://github.com/joowOn917/StarGAN_Virtual_Fitting/assets/143769249/0eb5c264-2026-421b-a51c-931297064d63" height="80%">


### 4️⃣ 모델 구조 설계

#### [Generator]

<img src="https://github.com/joowOn917/StarGAN_Virtual_Fitting/assets/143769249/b3f499c4-6231-4a5a-bb94-d8acb4d1c8c9" height="80%">



#### [Discriminator]

<img src="https://github.com/joowOn917/StarGAN_Virtual_Fitting/assets/143769249/5de89b1a-6140-40b1-8b70-f91d318a5abd" height="80%">



### 모델 학습 과정 
예시 도메인: 긴팔(longsleeve)
<img src = "https://github.com/joowOn917/StarGAN_Virtual_Fitting/assets/143769249/f74aa55d-a595-4a1b-98d7-fed410ef082e" height="80%">





