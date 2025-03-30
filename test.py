import torch
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from model import MNISTModel, CIFAR10Model
import attack  # attack.py에 정의한 공격 함수 임포트

def train(model, device, train_loader, optimizer, epoch):
    """
    모델 학습 함수: 한 epoch 동안 모델을 학습시킴
    100 배치마다 loss를 출력하여 학습 진행 상황을 보여줌
    """
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 100 == 0:
            print(f'Epoch {epoch} [{batch_idx*len(data)}/{len(train_loader.dataset)}] Loss: {loss.item():.6f}')

def test(model, device, test_loader):
    """
    모델 테스트 함수: 테스트셋에 대한 정확도 계산
    전체 테스트셋에 대한 평균 loss와 정확도를 반환
    """
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.cross_entropy(output, target, reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    test_loss /= len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)
    print(f'\nTest set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} ({accuracy:.2f}%)\n')
    return accuracy

def test_attacks(model, device, test_loader, dataset_name="MNIST"):
    """
    adversarial 공격 테스트 함수: FGSM과 PGD 공격 성공률 측정
    각 공격 방법(FGSM/PGD, 타깃/언타깃)에 대한 성공률을 계산하고 출력
    """
    data, target = next(iter(test_loader))
    data, target = data.to(device), target.to(device)

    print(f"\n=== {dataset_name} Adversarial Attack Results ===")
    
    # 1) 언타깃 FGSM 공격 (eps=0.25)
    data_adv = attack.fgsm_untargeted(model, data, target, eps=0.25)
    output_adv = model(data_adv)
    pred_adv = output_adv.argmax(dim=1, keepdim=True)
    correct_adv = pred_adv.eq(target.view_as(pred_adv)).sum().item()
    print(f"FGSM Untargeted 공격 결과: {correct_adv} / {data.size(0)} 올바르게 분류됨")

    # 2) 타깃 FGSM 공격 (목표 레이블: (target+1)%10, eps=0.25)
    target_adv = (target + 1) % 10
    data_adv_targeted = attack.fgsm_targeted(model, data, target_adv, eps=0.25)
    output_adv_targeted = model(data_adv_targeted)
    pred_adv_targeted = output_adv_targeted.argmax(dim=1, keepdim=True)
    correct_adv_targeted = pred_adv_targeted.eq(target_adv.view_as(pred_adv_targeted)).sum().item()
    print(f"FGSM Targeted 공격 결과: {correct_adv_targeted} / {data.size(0)} 목표 클래스로 분류됨")

    # 3) 언타깃 PGD 공격 (k=10, eps=0.25, eps_step=0.05)
    data_adv_pgd = attack.pgd_untargeted(model, data, target, k=10, eps=0.25, eps_step=0.05)
    output_adv_pgd = model(data_adv_pgd)
    pred_adv_pgd = output_adv_pgd.argmax(dim=1, keepdim=True)
    correct_adv_pgd = pred_adv_pgd.eq(target.view_as(pred_adv_pgd)).sum().item()
    print(f"PGD Untargeted 공격 결과: {correct_adv_pgd} / {data.size(0)} 올바르게 분류됨")

    # 4) 타깃 PGD 공격 (k=10, eps=0.25, eps_step=0.05)
    data_adv_pgd_targeted = attack.pgd_targeted(model, data, target_adv, k=10, eps=0.25, eps_step=0.05)
    output_adv_pgd_targeted = model(data_adv_pgd_targeted)
    pred_adv_pgd_targeted = output_adv_pgd_targeted.argmax(dim=1, keepdim=True)
    correct_adv_pgd_targeted = pred_adv_pgd_targeted.eq(target_adv.view_as(pred_adv_pgd_targeted)).sum().item()
    print(f"PGD Targeted 공격 결과: {correct_adv_pgd_targeted} / {data.size(0)} 목표 클래스로 분류됨")

def main():
    # GPU 사용 가능 여부에 따라 device 설정
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # -----------------------
    # MNIST 데이터셋 준비 및 테스트
    # -----------------------
    transform_mnist = transforms.Compose([transforms.ToTensor()])
    train_dataset_mnist = datasets.MNIST('./data', train=True, download=True, transform=transform_mnist)
    test_dataset_mnist = datasets.MNIST('./data', train=False, transform=transform_mnist)
    train_loader_mnist = DataLoader(train_dataset_mnist, batch_size=64, shuffle=True)
    test_loader_mnist = DataLoader(test_dataset_mnist, batch_size=1000, shuffle=False)

    # MNIST 모델 초기화 및 옵티마이저 설정
    model_mnist = MNISTModel().to(device)
    optimizer_mnist = optim.Adam(model_mnist.parameters(), lr=0.001)

    print("\n=== MNIST Model Training ===")
    epochs = 1
    for epoch in range(1, epochs + 1):
        train(model_mnist, device, train_loader_mnist, optimizer_mnist, epoch)
        test(model_mnist, device, test_loader_mnist)

    # MNIST 모델에 대한 adversarial 공격 테스트
    test_attacks(model_mnist, device, test_loader_mnist, "MNIST")

    # -----------------------
    # CIFAR-10 데이터셋 준비 및 테스트
    # -----------------------
    # 데이터 전처리 개선
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    train_dataset_cifar = datasets.CIFAR10('./data', train=True, download=True, transform=transform_train)
    test_dataset_cifar = datasets.CIFAR10('./data', train=False, transform=transform_test)
    train_loader_cifar = DataLoader(train_dataset_cifar, batch_size=128, shuffle=True)
    test_loader_cifar = DataLoader(test_dataset_cifar, batch_size=1000, shuffle=False)

    # CIFAR-10 모델 초기화 및 옵티마이저 설정
    model_cifar = CIFAR10Model().to(device)
    optimizer_cifar = optim.Adam(model_cifar.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.StepLR(optimizer_cifar, step_size=30, gamma=0.1)

    print("\n=== CIFAR-10 Model Training ===")
    epochs = 5  # 학습 epoch 증가
    best_acc = 0
    for epoch in range(1, epochs + 1):
        print(f"\nEpoch: {epoch}")
        train(model_cifar, device, train_loader_cifar, optimizer_cifar, epoch)
        acc = test(model_cifar, device, test_loader_cifar)
        scheduler.step()
        
        if acc > best_acc:
            best_acc = acc
            print(f"New best accuracy: {best_acc:.2f}%")

    print(f"\nBest accuracy achieved: {best_acc:.2f}%")

    # CIFAR-10 모델에 대한 adversarial 공격 테스트
    test_attacks(model_cifar, device, test_loader_cifar, "CIFAR-10")

if __name__ == '__main__':
    main()
