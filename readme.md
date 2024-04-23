<img src="https://github.com/joowOn917/StarGAN_Virtual_Fitting/assets/143769249/19426e87-bccf-4c93-b5f8-1aa0f74f3545" >

## 🖥 Overview 
StarGAN 모델을 활용하여 피팅이 불가능한 오프라인 쇼핑몰이나 온라인 쇼핑몰에서 사용할 수 있는 가상 의류 피팅 서비스 개발 및 제안 

## 🛠 Requirement 
- Python 3.7.9
- Pytorch 1.21.1
- nvidia 11.6
- tensorflow
- numpy 
- matplotlib

## ⚙ Project Process
1. 구글과 네이버에서 이미지 크롤링을 진행하여 대량의 의상 이미지 수집 후 분류
2. 이미지들에 대한 도메인(카테고리)별 라벨링
3. 모델에 적합하도록 이미지 사이즈 정규화
4. 생성자와 판별자 모델 설계
5. 12,327개의 데이터와 100,000회의 epoch로 모델 학습 진행
6. tensorboard를 통해 모델 학습 과정 모니터링

### 모델 학습 과정 
예시 도메인: 긴팔(longsleeve)
<img src = "https://github.com/joowOn917/StarGAN_Virtual_Fitting/assets/143769249/f74aa55d-a595-4a1b-98d7-fed410ef082e" height="80%">

<!-- 
### 라벨링 
파일명을 활용하여 도메인별 라벨링 진행

for i in range(len(names)):
    check = df['file_name'][i]
    
    if check.startswith('sleeve'):
        sleeveless.append(1)
    else:
        sleeveless.append(-1)
    
    if check.startswith('long'):
        longsleeve.append(1)
    else:
        longsleeve.append(-1)
 -->
