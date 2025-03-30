import torch
import torch.nn.functional as F

def fgsm_targeted(model, x, target, eps):
    x_adv = x.clone().detach().requires_grad_(True)
    output = model(x_adv)
    loss = F.cross_entropy(output, target)
    model.zero_grad()
    loss.backward()
    grad = x_adv.grad.data
    # 타깃 공격은 gradient 방향의 반대로 이동하여 목표 클래스로 유도
    x_adv = x_adv - eps * grad.sign()
    x_adv = torch.clamp(x_adv, 0, 1)
    return x_adv

def fgsm_untargeted(model, x, label, eps):
    x_adv = x.clone().detach().requires_grad_(True)
    output = model(x_adv)
    loss = F.cross_entropy(output, label)
    model.zero_grad()
    loss.backward()
    grad = x_adv.grad.data
    x_adv = x_adv + eps * grad.sign()
    x_adv = torch.clamp(x_adv, 0, 1)
    return x_adv

def pgd_targeted(model, x, target, k, eps, eps_step):
    x_adv = x.clone().detach()
    # 옵션: 초기값에 작은 random noise 추가
    x_adv = x_adv + torch.zeros_like(x_adv).uniform_(-eps, eps)
    x_adv = torch.clamp(x_adv, 0, 1)
    
    for i in range(k):
        x_adv.requires_grad = True
        output = model(x_adv)
        loss = F.cross_entropy(output, target)
        model.zero_grad()
        loss.backward()
        grad = x_adv.grad.data
        x_adv = x_adv - eps_step * grad.sign()
        # 원본 x로부터의 perturbation이 eps 내에 있도록 제한
        perturbation = torch.clamp(x_adv - x, -eps, eps)
        x_adv = torch.clamp(x + perturbation, 0, 1).detach()
    return x_adv

def pgd_untargeted(model, x, label, k, eps, eps_step):
    x_adv = x.clone().detach()
    # 옵션: 초기값에 작은 random noise 추가
    x_adv = x_adv + torch.zeros_like(x_adv).uniform_(-eps, eps)
    x_adv = torch.clamp(x_adv, 0, 1)
    
    for i in range(k):
        x_adv.requires_grad = True
        output = model(x_adv)
        loss = F.cross_entropy(output, label)
        model.zero_grad()
        loss.backward()
        grad = x_adv.grad.data
        x_adv = x_adv + eps_step * grad.sign()
        perturbation = torch.clamp(x_adv - x, -eps, eps)
        x_adv = torch.clamp(x + perturbation, 0, 1).detach()
    return x_adv
