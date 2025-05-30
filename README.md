﻿# Reliable and Trustworthy AI - Assignment #1

이 프로젝트는 MNIST와 CIFAR-10 데이터셋을 대상으로 adversarial attack을 구현한 것입니다.

## 구현 내용

1. **FGSM (Fast Gradient Sign Method) 공격**
   - 타깃 FGSM: 입력을 특정 목표 클래스로 분류되도록 유도
   - 언타깃 FGSM: 입력이 원래 클래스로 분류되지 않도록 유도

2. **PGD (Projected Gradient Descent) 공격**
   - 타깃 PGD: FGSM을 반복적으로 수행하여 목표 클래스로 분류되도록 유도
   - 언타깃 PGD: FGSM을 반복적으로 수행하여 오분류를 유도

## 실험 결과 및 분석

### 모델 성능
- MNIST: 98.47% 테스트 정확도 (1 epoch)
- CIFAR-10: 73.22% 테스트 정확도 (5 epochs)

### Adversarial Attack 결과

#### MNIST
- FGSM Untargeted: 82.1% 공격 성공률
- FGSM Targeted: 31.0% 공격 성공률
- PGD Untargeted: 100% 공격 성공률
- PGD Targeted: 84.8% 공격 성공률

#### CIFAR-10
- FGSM Untargeted: 86.6% 공격 성공률
- FGSM Targeted: 12.9% 공격 성공률
- PGD Untargeted: 99.8% 공격 성공률
- PGD Targeted: 74.0% 공격 성공률

### 결과 분석

실험 결과, 예상대로 CIFAR-10이 MNIST보다 전반적으로 낮은 공격 성공률을 보였습니다. 특히 주목할 만한 점은:

1. CIFAR-10 모델이 5 epochs 동안 학습되었음에도 불구하고, 1 epoch만 학습한 MNIST 모델보다 adversarial attack에 대한 성공률이 대체로 낮았습니다. 이는 다음과 같은 이유로 설명될 수 있습니다:
   - CIFAR-10은 MNIST보다 훨씬 복잡한 데이터셋입니다 (컬러 이미지, 다양한 객체 형태)
   - MNIST는 단순한 흑백 숫자 이미지로, 픽셀 변화에 더 민감하게 반응합니다
   - CIFAR-10의 이미지는 더 많은 채널(RGB)과 복잡한 특징을 가지고 있어, 단순한 픽셀 조작으로는 모델을 쉽게 속이기 어렵습니다

2. 타깃 공격의 경우 특히 CIFAR-10에서 성공률이 낮았습니다:
   - MNIST: 31.0% (FGSM), 84.8% (PGD)
   - CIFAR-10: 12.9% (FGSM), 74.0% (PGD)
   
   이는 복잡한 데이터셋에서 특정 타깃 클래스로의 공격이 더 어렵다는 것을 보여줍니다.

3. PGD 공격은 두 데이터셋 모두에서 FGSM보다 훨씬 효과적이었습니다:
   - 반복적인 그래디언트 업데이트가 더 강력한 adversarial example을 생성
   - 특히 언타깃 PGD는 거의 100%에 가까운 공격 성공률을 보임

## 프로젝트 구조

```
assignment/
├── attack.py        # FGSM, PGD 공격 함수 구현 (타깃/언타깃)
├── model.py         # MNIST와 CIFAR-10을 위한 CNN 모델 정의
├── test.py         # 모델 학습 및 공격 검증 코드
└── requirements.txt # 필요한 외부 라이브러리 목록
```

## 설치 방법

필요한 패키지 설치:
```bash
pip install -r requirements.txt
```

## 실행 방법

```bash
python test.py
```

이 스크립트는 다음 작업을 수행합니다:
1. MNIST와 CIFAR-10 데이터셋을 다운로드
2. 각 데이터셋에 대한 CNN 모델을 학습
   - MNIST: 1 epoch
   - CIFAR-10: 5 epochs, 데이터 증강 및 정규화 적용
3. 학습된 모델에 대해 FGSM과 PGD 공격을 수행하고 결과를 출력

## 참고 문헌

1. Goodfellow, I. J., Shlens, J., & Szegedy, C. (2014). Explaining and harnessing adversarial examples. arXiv preprint arXiv:1412.6572.
2. Madry, A., Makelov, A., Schmidt, L., Tsipras, D., & Vladu, A. (2017). Towards deep learning models resistant to adversarial attacks. arXiv preprint arXiv:1706.06083.
