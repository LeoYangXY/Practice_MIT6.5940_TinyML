"""
=============================================================================
剪枝 + 线性量化 + QAT 完整流程 Demo
=============================================================================
说明：
1. 非结构化权重剪枝（按abs大小置零，不修改shape）
2. 线性量化（INT8对称量化）
3. QAT量化感知训练
4. PTQ训练后量化
=============================================================================
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import models
import copy
from tqdm import tqdm
import numpy as np
from typing import Union, List, Dict, Tuple

# ==========================================
# 配置部分
# ==========================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"当前设备：{DEVICE}")

# 量化配置
QUANT_BITWIDTH = 8
QAT_EPOCHS = 3

# 剪枝配置
PRUNING_AMOUNT = 0.3
FINETUNE_EPOCHS = 3       # 剪枝后恢复精度的 fine-tune 轮数
FINETUNE_LR = 0.001

# 训练配置
BATCH_SIZE = 64
LEARNING_RATE = 0.0005    # QAT 学习率（通常比 fine-tune 小）


# ==========================================
# 1. 线性量化核心模块
# ==========================================

class LinearQuantizer:
    def __init__(self, bitwidth=8, symmetric=True):
        self.bitwidth = bitwidth
        self.symmetric = symmetric
        self.num_levels = 2 ** bitwidth
        
        if symmetric:
            self.qmin = -(2 ** (bitwidth - 1)) + 1
            self.qmax = 2 ** (bitwidth - 1) - 1
        else:
            self.qmin = 0
            self.qmax = 2 ** bitwidth - 1
            
        self.scale = None
        self.zero_point = None
    
    def calculate_qparams(self, x):
        if self.symmetric:
            max_val = torch.abs(x).max()
            max_val = torch.clamp(max_val, min=1e-8)
            self.scale = max_val / self.qmax
            self.zero_point = 0
        else:
            min_val = x.min()
            max_val = x.max()
            max_val = torch.clamp(max_val - min_val, min=1e-8)
            self.scale = max_val / (self.qmax - self.qmin)
            self.zero_point = torch.round(self.qmin - min_val / self.scale)
            self.zero_point = torch.clamp(self.zero_point, self.qmin, self.qmax)
        return self.scale, self.zero_point
    
    def quantize(self, x):
        if self.scale is None:
            self.calculate_qparams(x)
        x_q = torch.round(x / self.scale)
        x_q = torch.clamp(x_q, self.qmin, self.qmax)
        return x_q
    
    def dequantize(self, x_q):
        return x_q * self.scale + self.zero_point
    
    def forward(self, x):
        x_q = self.quantize(x)
        return self.dequantize(x_q)


# ==========================================
# 2. 伪量化模块 (Fake Quantize for QAT)
# ==========================================

class FakeQuantize(nn.Module):
    def __init__(self, bitwidth=8, symmetric=True):
        super().__init__()
        self.quantizer = LinearQuantizer(bitwidth=bitwidth, symmetric=symmetric)
        self.bitwidth = bitwidth
    
    def forward(self, x):
        x_quantized = self.quantizer.forward(x)
        # STE: 前向用量化值，反向梯度直通
        return x_quantized + (x - x.detach())


class QuantizedConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bitwidth=8):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.weight_fake_quant = FakeQuantize(bitwidth=bitwidth, symmetric=True)
        self.act_fake_quant = FakeQuantize(bitwidth=bitwidth, symmetric=True)
    
    def forward(self, x):
        self.conv.weight.data = self.weight_fake_quant(self.conv.weight)
        x = self.conv(x)
        x = self.act_fake_quant(x)
        return x


class QuantizedLinear(nn.Module):
    def __init__(self, in_features, out_features, bitwidth=8):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.weight_fake_quant = FakeQuantize(bitwidth=bitwidth, symmetric=True)
        self.act_fake_quant = FakeQuantize(bitwidth=bitwidth, symmetric=True)
    
    def forward(self, x):
        self.linear.weight.data = self.weight_fake_quant(self.linear.weight)
        x = self.linear(x)
        x = self.act_fake_quant(x)
        return x


# ==========================================
# 3. 模型准备
# ==========================================

def get_base_model(pretrained=True):
    """加载基础 ResNet18 模型（普通 nn.Linear fc，不带伪量化）"""
    print("\n[模型] 加载ResNet18...")
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT if pretrained else None)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 10)  # 普通 Linear 用于剪枝 + fine-tune 阶段
    print(f"[模型] 参数量：{sum(p.numel() for p in model.parameters()):,}")
    return model


def get_model_with_qat(pretrained=True, bitwidth=8):
    print("\n[模型] 加载ResNet18并插入QAT伪量化节点...")
    
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT if pretrained else None)
    
    # 修改最后一层
    num_ftrs = model.fc.in_features
    model.fc = QuantizedLinear(num_ftrs, 10, bitwidth=bitwidth)
    
    qat_layers = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            qat_layers.append(name)
    
    print(f"[模型] 找到 {len(qat_layers)} 个卷积层将应用QAT")
    print(f"[模型] 参数量：{sum(p.numel() for p in model.parameters()):,}")
    
    return model, qat_layers


def apply_qat_to_model(model, qat_layers, bitwidth=8):
    fake_quant_modules = {}
    
    for name, module in model.named_modules():
        if name in qat_layers and isinstance(module, nn.Conv2d):
            fq = FakeQuantize(bitwidth=bitwidth, symmetric=True)
            fake_quant_modules[name] = fq
            
            def make_hook(fq_module):
                def hook(m, input, output):
                    return fq_module(output)
                return hook
            
            module.register_forward_hook(make_hook(fq))
    
    def weight_quant_hook(model, qat_layers, fake_quant_modules):
        for name, module in model.named_modules():
            if name in qat_layers and isinstance(module, nn.Conv2d):
                if name in fake_quant_modules:
                    module.weight.data = fake_quant_modules[name](module.weight)
    
    return model, fake_quant_modules, weight_quant_hook


# ==========================================
# 4. 非结构化权重剪枝 (按abs大小置零，不修改shape)
# ==========================================

@torch.no_grad()
def magnitude_prune(model: nn.Module, prune_ratio: float = 0.3) -> Tuple[nn.Module, Dict, float]:
    """
    【非结构化权重剪枝】按权重绝对值大小将最小的权重置零
    
    优点：不修改模型shape，不需要处理上下游通道匹配、skip connection等
    
    Args:
        model: 原始模型
        prune_ratio: 剪枝比例 (0.3 = 将30%最小权重置零)
    
    Returns:
        pruned_model: 剪枝后的模型
        prune_info: 每层剪枝信息
        sparsity: 实际整体稀疏度
    """
    print(f"\n[权重剪枝] 开始应用 {prune_ratio*100}% 非结构化权重剪枝...")
    
    # 深拷贝防止修改原模型
    model = copy.deepcopy(model)
    
    # 收集所有需要剪枝的权重（Conv2d 和 Linear）
    all_weights = []
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            all_weights.append(module.weight.data.abs().flatten())
    
    # 拼接所有权重的绝对值，计算全局阈值
    all_weights_cat = torch.cat(all_weights)
    total_params = all_weights_cat.numel()
    k = int(total_params * prune_ratio)
    if k == 0:
        print("[权重剪枝] 剪枝比例过小，无需剪枝")
        return model, {}, 0.0
    
    # 全局阈值：第 k 小的绝对值
    threshold = torch.kthvalue(all_weights_cat, k).values
    print(f"[权重剪枝] 全局阈值: {threshold:.6f}")
    
    # 逐层应用剪枝掩码
    prune_info = {}
    total_pruned = 0
    total_weights = 0
    
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            weight = module.weight.data
            mask = (weight.abs() >= threshold).float()
            
            n_total = weight.numel()
            n_pruned = (mask == 0).sum().item()
            layer_sparsity = n_pruned / n_total if n_total > 0 else 0
            
            # 将小权重置零
            module.weight.data *= mask
            
            prune_info[name] = {
                'total_weights': n_total,
                'pruned_weights': int(n_pruned),
                'sparsity': layer_sparsity,
                'shape': list(weight.shape)
            }
            
            total_pruned += n_pruned
            total_weights += n_total
            
            print(f"  {name}: {n_total} 个权重，置零 {int(n_pruned)} 个 "
                  f"(稀疏度 {layer_sparsity:.2%})，shape不变 {list(weight.shape)}")
    
    sparsity = total_pruned / total_weights if total_weights > 0 else 0
    print(f"\n[权重剪枝] 完成。整体稀疏度：{sparsity:.2%}")
    print(f"[权重剪枝] 共 {total_weights} 个权重，置零 {total_pruned} 个")
    
    return model, prune_info, sparsity


# ==========================================
# 5. PTQ (Post-Training Quantization) — 用校准数据确定量化参数
# ==========================================

@torch.no_grad()
def apply_ptq(model, calibration_loader, bitwidth=8, num_batches=20):
    """
    PTQ: 用少量校准数据统计每层权重/激活的 min/max，直接计算量化参数。
    不做任何训练，只做推理收集统计信息后量化权重。
    
    Args:
        model: fine-tune 后的剪枝模型
        calibration_loader: 校准数据（通常用训练集的一个子集）
        bitwidth: 量化位宽
        num_batches: 用多少个 batch 做校准
    
    Returns:
        量化后的模型（深拷贝）
    """
    print(f"\n[PTQ] 应用 {bitwidth}-bit Post-Training Quantization...")
    ptq_model = copy.deepcopy(model)
    ptq_model.eval()
    
    # Step 1: 量化所有权重
    weight_quantizer = LinearQuantizer(bitwidth=bitwidth, symmetric=True)
    for name, module in ptq_model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            w = module.weight.data
            weight_quantizer.scale = None  # 每层重新计算
            module.weight.data = weight_quantizer.forward(w)
    print(f"[PTQ] 所有权重已量化为 {bitwidth}-bit")
    
    # Step 2: 用校准数据收集激活统计量（此处简化为 forward pass）
    print(f"[PTQ] 使用 {num_batches} 个 batch 校准激活...")
    for batch_idx, (inputs, _) in enumerate(calibration_loader):
        if batch_idx >= num_batches:
            break
        inputs = inputs.to(DEVICE)
        ptq_model(inputs)
    
    print("[PTQ] 量化完成")
    return ptq_model


# ==========================================
# 6. QAT训练流程
# ==========================================

class QATTrainer:
    def __init__(self, model, fake_quant_modules, weight_quant_hook, device, lr=0.001):
        self.model = model
        self.fake_quant_modules = fake_quant_modules
        self.weight_quant_hook = weight_quant_hook
        self.device = device
        self.lr = lr
        
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=lr,
            momentum=0.9,
            weight_decay=1e-4
        )
    
    def train_epoch(self, dataloader, epoch):
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(dataloader, desc=f"Epoch [{epoch+1}]", leave=True)
        for batch_idx, (inputs, targets) in enumerate(pbar):
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)
            loss.backward()
            self.optimizer.step()
            
            self.weight_quant_hook(self.model, list(self.fake_quant_modules.keys()), self.fake_quant_modules)
            
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            avg_loss = total_loss / (batch_idx + 1)
            accuracy = 100. * correct / total
            pbar.set_postfix({'Loss': f'{avg_loss:.4f}', 'Acc': f'{accuracy:.2f}%'})
        
        return total_loss / len(dataloader), 100. * correct / total
    
    @staticmethod
    def evaluate(model, dataloader, device=None):
        if device is None:
            device = DEVICE
        model.eval()
        correct = 0
        total = 0
        
        pbar = tqdm(dataloader, desc="Evaluating", leave=True)
        with torch.no_grad():
            for inputs, targets in pbar:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
                accuracy = 100. * correct / total
                pbar.set_postfix({'Acc': f'{accuracy:.2f}%'})
        
        return 100. * correct / total


# ==========================================
# 7. 模型大小与性能评估
# ==========================================

def get_model_size(model):
    total_params = sum(p.numel() for p in model.parameters())
    total_size = sum(p.numel() * p.element_size() for p in model.parameters())
    return total_params, total_size / (1024 * 1024)


def get_sparsity(model):
    """计算模型中零值权重的比例"""
    total = 0
    zeros = 0
    for p in model.parameters():
        total += p.numel()
        zeros += (p.data == 0).sum().item()
    return zeros / total if total > 0 else 0


def compare_all_models(models_dict: Dict[str, nn.Module], test_loader, device=DEVICE):
    """
    对比所有模型的性能。
    
    Args:
        models_dict: {"模型名": model, ...}
        test_loader: 测试数据
    """
    print("\n" + "="*80)
    print("模型性能对比报告")
    print("="*80)
    
    results = []
    first_params = None
    
    for idx, (name, model) in enumerate(models_dict.items(), 1):
        print(f"\n[{idx}] {name}:")
        params, size = get_model_size(model)
        if first_params is None:
            first_params = params
        sparsity = get_sparsity(model)
        acc = QATTrainer.evaluate(model, test_loader, device)  # 确保 evaluate 返回正确的准确率
        print(f"  参数量: {params:,}")
        print(f"  大小: {size:.2f} MB")
        print(f"  稀疏度: {sparsity:.2%}")
        print(f"  准确率: {acc:.2f}%")
        results.append((name, params, size, sparsity, acc))
    
    # 汇总表格
    print("\n" + "="*80)
    print("汇总对比表")
    print("="*80)
    header = f"{'模型':<25} {'参数量':<15} {'大小(MB)':<10} {'稀疏度':<10} {'准确率':<10}"
    print(header)
    print("-"*80)
    for name, params, size, sp, acc in results:
        print(f"{name:<25} {params:<15,} {size:<10.2f} {sp:<10.2%} {acc:<10.2f}%")
    print("="*80)
    
    return results


# ==========================================
# 8. 数据准备
# ==========================================

def get_dataloader(batch_size=32, train=True):
    import torchvision.transforms as transforms
    from torchvision import datasets
    
    print(f"[数据] 加载CIFAR-10数据集 ({'train' if train else 'test'})...")
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    
    dataset = datasets.CIFAR10(
        root='./data',
        train=train,
        download=True,
        transform=transform
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=train,
        num_workers=4,
        drop_last=True,
        pin_memory=True
    )
    
    return dataloader


# ==========================================
# 9. 剪枝后 Fine-tune（纯训练，不带量化）
# ==========================================

def finetune_after_pruning(model, train_loader, test_loader, epochs=3, lr=0.001, device=DEVICE):
    """
    Stage 2: 剪枝后恢复精度的纯 fine-tune（不插入伪量化）
    """
    print(f"\n[Fine-tune] 开始剪枝后恢复训练，共 {epochs} 个 epoch，lr={lr}")
    model.train()
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr, momentum=0.9, weight_decay=1e-4
    )
    
    # fine-tune 前精度
    pre_acc = QATTrainer.evaluate(model, test_loader, device)
    print(f"[Fine-tune] 训练前精度：{pre_acc:.2f}%")
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(train_loader, desc=f"Fine-tune [{epoch+1}/{epochs}]", leave=True)
        for batch_idx, (inputs, targets) in enumerate(pbar):
            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            pbar.set_postfix({'Loss': f'{total_loss/(batch_idx+1):.4f}',
                              'Acc': f'{100.*correct/total:.2f}%'})
        
        epoch_acc = QATTrainer.evaluate(model, test_loader, device)
        print(f"[Fine-tune] Epoch {epoch+1} 测试精度：{epoch_acc:.2f}%")
    
    post_acc = QATTrainer.evaluate(model, test_loader, device)
    print(f"[Fine-tune] 训练后精度：{post_acc:.2f}% (变化: {post_acc - pre_acc:+.2f}%)")
    return model, post_acc


# ==========================================
# 10. 主流程
# ==========================================

def main():
    print("=" * 70)
    print("标准模型压缩流程: 剪枝 → Fine-tune → 量化(PTQ & QAT)")
    print("=" * 70)
    print(f"设备：{DEVICE}")
    print(f"量化位宽：{QUANT_BITWIDTH}-bit (INT8)")
    print(f"剪枝比例：{PRUNING_AMOUNT*100}% (非结构化权重剪枝)")
    print(f"Fine-tune轮数：{FINETUNE_EPOCHS}")
    print(f"QAT训练轮数：{QAT_EPOCHS}")
    print("=" * 70)
    
    # ================================================================
    # 准备数据
    # ================================================================
    print("\n>>> 准备数据")
    train_loader = get_dataloader(batch_size=BATCH_SIZE, train=True)
    test_loader = get_dataloader(batch_size=BATCH_SIZE, train=False)
    
    # ================================================================
    # Stage 0: 加载原始预训练模型（基线）
    # ================================================================
    print("\n" + "="*70)
    print(">>> Stage 0: 加载原始预训练模型")
    print("="*70)
    original_model = get_base_model(pretrained=True).to(DEVICE)
    original_model.eval()
    
    orig_acc = QATTrainer.evaluate(original_model, test_loader, DEVICE)
    print(f"原始模型精度：{orig_acc:.2f}%")
    
    # ================================================================
    # Stage 1: 非结构化权重剪枝（按 abs 大小置零，不改 shape）
    # ================================================================
    print("\n" + "="*70)
    print(">>> Stage 1: 非结构化权重剪枝")
    print("="*70)
    pruned_model, prune_info, sparsity = magnitude_prune(
        original_model, prune_ratio=PRUNING_AMOUNT
    )
    pruned_model = pruned_model.to(DEVICE)
    
    pruned_acc = QATTrainer.evaluate(pruned_model, test_loader, DEVICE)
    print(f"剪枝后精度：{pruned_acc:.2f}% (掉了 {pruned_acc - orig_acc:+.2f}%)")
    
    # ================================================================
    # Stage 2: 剪枝后 Fine-tune（纯训练，不带量化，恢复精度）
    # ================================================================
    print("\n" + "="*70)
    print(">>> Stage 2: 剪枝后 Fine-tune 恢复精度")
    print("="*70)
    finetuned_model, ft_acc = finetune_after_pruning(
        pruned_model, train_loader, test_loader,
        epochs=FINETUNE_EPOCHS, lr=FINETUNE_LR, device=DEVICE
    )
    
    # ================================================================
    # Stage 3A: PTQ — 直接量化（不训练）
    # ================================================================
    print("\n" + "="*70)
    print(">>> Stage 3A: PTQ (Post-Training Quantization)")
    print("="*70)
    ptq_model = apply_ptq(
        finetuned_model, calibration_loader=train_loader,
        bitwidth=QUANT_BITWIDTH, num_batches=20
    )
    ptq_acc = QATTrainer.evaluate(ptq_model, test_loader, DEVICE)
    print(f"PTQ模型精度：{ptq_acc:.2f}%")
    
    # ================================================================
    # Stage 3B: QAT — 量化感知训练
    # ================================================================
    print("\n" + "="*70)
    print(">>> Stage 3B: QAT (Quantization-Aware Training)")
    print("="*70)
    
    # 从 fine-tune 后的模型重新深拷贝一份用于 QAT
    qat_model = copy.deepcopy(finetuned_model)
    
    # 替换 fc 为 QuantizedLinear
    fc_in = qat_model.fc.in_features
    fc_weight = qat_model.fc.weight.data.clone()
    fc_bias = qat_model.fc.bias.data.clone()
    qat_model.fc = QuantizedLinear(fc_in, 10, bitwidth=QUANT_BITWIDTH)
    qat_model.fc.linear.weight.data = fc_weight
    qat_model.fc.linear.bias.data = fc_bias
    qat_model = qat_model.to(DEVICE)
    
    # 收集卷积层并插入伪量化
    qat_layers = [name for name, m in qat_model.named_modules() if isinstance(m, nn.Conv2d)]
    qat_model, fake_quant_modules, weight_quant_hook = apply_qat_to_model(
        qat_model, qat_layers, bitwidth=QUANT_BITWIDTH
    )
    
    pre_qat_acc = QATTrainer.evaluate(qat_model, test_loader, DEVICE)
    print(f"QAT训练前精度：{pre_qat_acc:.2f}%")
    
    trainer = QATTrainer(
        model=qat_model,
        fake_quant_modules=fake_quant_modules,
        weight_quant_hook=weight_quant_hook,
        device=DEVICE,
        lr=LEARNING_RATE
    )
    
    print(f"\n[QAT] 开始训练 {QAT_EPOCHS} 个 epoch:")
    for epoch in range(QAT_EPOCHS):
        trainer.train_epoch(train_loader, epoch)
    
    qat_acc = QATTrainer.evaluate(qat_model, test_loader, DEVICE)
    print(f"QAT训练后精度：{qat_acc:.2f}% (变化: {qat_acc - pre_qat_acc:+.2f}%)")
    
    # ================================================================
    # 最终对比
    # ================================================================
    print("\n" + "="*70)
    print(">>> 最终对比：所有模型")
    print("="*70)
    
    compare_all_models({
        '① 原始模型':          original_model,
        '② 剪枝后(未fine-tune)': magnitude_prune(original_model, PRUNING_AMOUNT)[0].to(DEVICE),
        '③ 剪枝+Fine-tune':    finetuned_model,
        '④ 剪枝+FT+PTQ':       ptq_model,
        '⑤ 剪枝+FT+QAT':       qat_model,
    }, test_loader, DEVICE)
    
    # ==================== 总结 ====================
    print("\n" + "=" * 70)
    print("流程完成总结:")
    print("=" * 70)
    print(f"✓ 基础模型：ResNet18 → CIFAR-10 (10类)")
    print(f"✓ Stage 1 剪枝：{PRUNING_AMOUNT*100}% 非结构化权重剪枝，稀疏度 {sparsity:.2%}")
    print(f"✓ Stage 2 Fine-tune：{FINETUNE_EPOCHS} 轮，精度 {pruned_acc:.2f}% → {ft_acc:.2f}%")
    print(f"✓ Stage 3A PTQ：{QUANT_BITWIDTH}-bit，精度 {ptq_acc:.2f}%")
    print(f"✓ Stage 3B QAT：{QUANT_BITWIDTH}-bit，{QAT_EPOCHS} 轮，精度 {qat_acc:.2f}%")
    print(f"")
    print(f"输出模型 1：ptq_model  (剪枝 + Fine-tune + PTQ)")
    print(f"输出模型 2：qat_model  (剪枝 + Fine-tune + QAT)")
    print("=" * 70)
    
    return ptq_model, qat_model


if __name__ == "__main__":
    ptq_model, qat_model = main()