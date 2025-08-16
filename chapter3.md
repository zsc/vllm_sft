# 第 3 章：SFT 训练策略

监督微调（Supervised Fine-Tuning, SFT）是将预训练的视觉语言模型适配到特定任务的关键步骤。与纯语言模型不同，VLM 的 SFT 需要同时考虑视觉和语言两种模态的对齐，这带来了独特的挑战：如何设计有效的指令格式？如何平衡不同任务的损失？如何在有限的计算资源下高效微调？本章将系统介绍 VLM SFT 的核心技术，从指令设计到训练优化，帮助您掌握将通用 VLM 转化为任务专家的完整流程。

## 3.1 指令微调的设计原则

指令微调的核心在于教会模型理解和遵循人类指令。对于 VLM，这意味着模型不仅要理解文本指令，还要将其与视觉输入关联起来。设计良好的指令格式是成功微调的第一步。

### 3.1.1 指令模板设计

VLM 的指令模板需要明确标识图像位置、用户指令和模型响应的边界。常见的模板格式包括：

**基础单轮对话模板：**
```
<image>
User: {instruction}
Assistant: {response}
```

**带系统提示的模板：**
```
System: {system_prompt}
<image>
User: {instruction}
Assistant: {response}
```

**多图像交织模板：**
```
User: 比较这两张图片 <image1> 和 <image2>，{instruction}
Assistant: {response}
```

关键设计原则：
1. **位置标记明确**：使用特殊 token（如 `<image>`、`<|im_start|>`）标记图像嵌入位置
2. **角色区分清晰**：明确区分 system、user、assistant 角色
3. **边界符号一致**：使用统一的开始/结束标记（如 `<|im_end|>`）

**Token 化示例：**
```
输入文本: "<image>\nUser: 描述这张图片\nAssistant: "
Token IDs: [32000, 13, 2659, 29901, 29871, 31904, 30810, 30775, 30998, 13, 7900, 22137, 29901, 29871]
            ↑图像占位  ↑换行  ↑User:        ↑描述这张图片      ↑换行 ↑Assistant:
```

### 3.1.2 系统提示词的作用

系统提示词（System Prompt）定义模型的角色和行为准则，对 VLM 的表现有显著影响：

**通用视觉助手提示：**
```
你是一个专业的视觉语言助手。请准确描述图像内容，回答用户关于图像的问题。
如果图像中包含文字，请准确识别并转录。避免猜测或编造不存在的内容。
```

**任务特定提示（OCR场景）：**
```
你是一个OCR专家。请：
1. 识别图像中的所有文字
2. 保持原始格式和布局
3. 标注不确定的字符为[?]
4. 忽略装饰性元素，专注文字内容
```

系统提示的优化技巧：
- **长度控制**：过长的系统提示会占用上下文窗口，建议控制在 100-200 token
- **任务聚焦**：针对特定任务定制提示，避免过于宽泛
- **示例引导**：在提示中包含期望输出格式的示例

### 3.1.3 多轮对话的处理

VLM 的多轮对话需要处理历史上下文和新图像输入的关系：

**策略1：图像持久化**
```python
# 第一轮
messages = [
    {"role": "user", "content": "<image> 这是什么动物？"},
    {"role": "assistant", "content": "这是一只橙色的猫。"}
]
# 第二轮（引用同一图像）
messages.append({"role": "user", "content": "它在做什么？"})
# 模型需要记住之前的图像上下文
```

**策略2：显式图像引用**
```python
# 使用图像ID系统
messages = [
    {"role": "user", "content": "<image id='img1'> 描述第一张图"},
    {"role": "assistant", "content": "第一张图显示..."},
    {"role": "user", "content": "<image id='img2'> 比较img1和img2的差异"},
]
```

**上下文窗口管理：**
```
最大上下文 = 4096 tokens
├── 系统提示: ~100 tokens
├── 图像嵌入: 576 tokens × N张图
├── 历史对话: 可变长度
└── 当前回复: 预留 500-1000 tokens
```

### 3.1.4 视觉-语言指令的对齐

确保视觉理解与语言生成的一致性是 VLM SFT 的核心挑战：

**对齐层次：**
1. **对象级对齐**：物体识别与命名一致
2. **属性级对齐**：颜色、大小、纹理描述准确
3. **关系级对齐**：空间关系、动作关系正确
4. **场景级对齐**：整体理解与描述连贯

**对齐技术：**

```
视觉特征对齐矩阵:
       物体  属性  关系  场景
       ________________
视觉  | 1.0  0.8  0.6  0.7 |  <- 视觉编码器输出
语言  | 0.9  0.9  0.7  0.8 |  <- 语言模型理解
       ‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾
对角线值越接近1.0，对齐越好
```

**细粒度对齐示例：**
```python
# Grounding 标注格式
instruction = "找出<click>红色的球</click>在哪里"
response = "红色的球位于<box>[[125, 235, 200, 310]]</box>图像的左下角。"

# Referring 标注格式  
instruction = "描述位于<region>[[x1,y1,x2,y2]]</region>的物体"
response = "这是一个红色的篮球，表面有黑色的线条纹理。"
```

## 3.2 损失函数设计与权重策略

损失函数设计直接影响模型的学习目标和收敛行为。VLM 的 SFT 需要精心设计损失函数来平衡不同类型的预测任务。

### 3.2.1 自回归语言模型损失

VLM 的核心损失是自回归语言建模损失，即预测下一个 token 的交叉熵损失：

$$\mathcal{L}_{LM} = -\sum_{t=1}^{T} \log P(x_t | x_{<t}, I)$$

其中 $x_t$ 是第 $t$ 个 token，$I$ 是输入图像，$x_{<t}$ 是之前的所有 token。

**实现细节：**
```python
def compute_lm_loss(logits, labels, vocab_size=32000):
    """
    logits: [batch_size, seq_len, vocab_size]
    labels: [batch_size, seq_len]
    """
    # Shift：预测位置和标签错位
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    
    # Flatten 便于计算
    shift_logits = shift_logits.view(-1, vocab_size)
    shift_labels = shift_labels.view(-1)
    
    # 交叉熵损失
    loss = F.cross_entropy(
        shift_logits, 
        shift_labels, 
        ignore_index=-100,  # 忽略 padding
        reduction='mean'
    )
    return loss
```

**注意力掩码的影响：**
```
序列: [IMG] User: 描述图片 Assistant: 这是一只猫 [EOS]
掩码:  0    0     0        1          1          1

只在 Assistant 响应部分计算损失
```

### 3.2.2 掩码策略与权重分配

不同部分的 token 对学习的重要性不同，通过掩码和权重调整可以优化训练效果：

**1. 响应掩码（Response Masking）：**
```python
def create_response_mask(input_ids, response_start_token_id):
    """只在模型响应部分计算损失"""
    batch_size, seq_len = input_ids.shape
    mask = torch.zeros_like(input_ids, dtype=torch.bool)
    
    for i in range(batch_size):
        # 找到响应开始位置
        response_start = (input_ids[i] == response_start_token_id).nonzero()
        if len(response_start) > 0:
            start_idx = response_start[0].item()
            mask[i, start_idx:] = True
    
    return mask
```

**2. Token 级别权重：**
```python
# 不同类型 token 的权重
token_weights = {
    "special_tokens": 0.0,    # <image>, <pad> 等
    "instruction": 0.0,        # 用户指令部分
    "response": 1.0,          # 助手响应
    "grounding_box": 2.0,     # 坐标预测
    "key_entities": 1.5       # 关键实体名词
}
```

**3. 动态权重调整：**
```
早期训练（epoch 1-3）：
- 所有 token 权重 = 1.0
- 让模型学习基础的语言模式

中期训练（epoch 4-8）：
- 指令部分权重 = 0.5
- 响应部分权重 = 1.0
- 强化指令遵循能力

后期训练（epoch 9-10）：
- 只计算响应损失
- 精细调整生成质量
```

### 3.2.3 多任务学习的损失平衡

VLM 通常需要同时处理多个任务，如图像描述、VQA、OCR 等。多任务损失平衡是关键：

**损失组合策略：**
$$\mathcal{L}_{total} = \sum_{i=1}^{N} w_i \mathcal{L}_i$$

**自适应权重方法：**

1. **不确定性加权（Uncertainty Weighting）：**
$$\mathcal{L}_{total} = \sum_{i=1}^{N} \frac{1}{2\sigma_i^2} \mathcal{L}_i + \log \sigma_i$$

其中 $\sigma_i$ 是可学习的任务不确定性参数。

2. **梯度归一化（GradNorm）：**
```python
def gradnorm_weights(losses, shared_params, alpha=1.5):
    """根据梯度大小动态调整任务权重"""
    # 计算每个任务的梯度范数
    grad_norms = []
    for loss in losses:
        grads = torch.autograd.grad(loss, shared_params, retain_graph=True)
        grad_norm = torch.norm(torch.cat([g.flatten() for g in grads]))
        grad_norms.append(grad_norm)
    
    # 计算平均梯度范数
    mean_norm = torch.stack(grad_norms).mean()
    
    # 调整权重
    weights = []
    for i, norm in enumerate(grad_norms):
        relative_norm = norm / mean_norm
        weight = relative_norm ** alpha
        weights.append(weight)
    
    return F.softmax(torch.stack(weights), dim=0)
```

**任务采样策略：**
```
批次构建策略:
├── 均匀采样: 每个 batch 包含所有任务
├── 任务分组: 相似任务放在同一 batch
└── 温度采样: P(task_i) ∝ (1/loss_i)^T
    
温度 T 控制采样分布:
- T → 0: 只采样损失最大的任务
- T = 1: 根据损失反比采样  
- T → ∞: 均匀采样所有任务
```

### 3.2.4 视觉 Grounding 损失设计

对于需要定位的任务（如目标检测、referring segmentation），需要专门的损失设计：

**1. 边界框回归损失：**
```python
def box_loss(pred_boxes, gt_boxes):
    """
    pred_boxes: [batch, num_queries, 4]  # (x1, y1, x2, y2) 归一化坐标
    gt_boxes: [batch, num_targets, 4]
    """
    # L1 损失
    l1_loss = F.l1_loss(pred_boxes, gt_boxes)
    
    # GIoU 损失
    giou_loss = 1 - compute_giou(pred_boxes, gt_boxes)
    
    return l1_loss + giou_loss
```

**2. 坐标 Token 化策略：**
```
方法1：连续坐标离散化
[0, 1] → [0, 999] → token_id ∈ [32000, 32999]

方法2：区域编码
图像分成 32×32 网格 → 每个网格一个 token

方法3：特殊数字 token
<x>0.123</x> <y>0.456</y> → 解析时提取
```

**3. Referring 损失设计：**
```python
def referring_loss(pred_mask, gt_mask, pred_box, gt_box):
    """组合分割和检测损失"""
    # 像素级分割损失
    seg_loss = F.binary_cross_entropy_with_logits(pred_mask, gt_mask)
    
    # 边界框损失
    box_loss = compute_box_loss(pred_box, gt_box)
    
    # 一致性损失：确保 mask 和 box 对应
    mask_from_box = box_to_mask(pred_box)
    consistency_loss = F.mse_loss(pred_mask, mask_from_box)
    
    return seg_loss + 0.5 * box_loss + 0.1 * consistency_loss
```

**4. 负样本处理：**
```
Grounding 任务的负样本策略:
├── Hard Negative: 选择最容易混淆的物体
├── Random Negative: 随机选择其他物体
└── Background: 选择背景区域

负样本比例建议:
- 正负比 1:3 for 目标检测
- 正负比 1:1 for referring expression
- 动态调整based on 难度
```

## 3.3 参数高效微调方法（LoRA、QLoRA、Adapter）

参数高效微调（PEFT）方法允许在有限的计算资源下微调大规模 VLM。这些方法通过只更新少量参数来实现与全量微调相近的效果。

### 3.3.1 LoRA 原理与实现

LoRA（Low-Rank Adaptation）通过低秩分解来近似权重更新：

**核心原理：**
$$W' = W + \Delta W = W + BA$$

其中 $B \in \mathbb{R}^{d \times r}$，$A \in \mathbb{R}^{r \times k}$，$r \ll \min(d, k)$。

**VLM 中的 LoRA 配置：**
```python
class LoRAConfig:
    # 语言模型部分
    lm_target_modules = [
        "q_proj", "v_proj",  # 注意力层
        "k_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"  # FFN 层
    ]
    
    # 视觉编码器部分（可选）
    vision_target_modules = [
        "qkv",  # ViT 的 QKV 投影
        "proj", # 输出投影
        "mlp.fc1", "mlp.fc2"  # MLP 层
    ]
    
    # 关键超参数
    r = 16  # rank，常用 8/16/32/64
    alpha = 16  # 缩放因子，通常 = r
    dropout = 0.1
```

**动态 Rank 选择：**
```
不同模块的重要性分析:
模块类型        建议 rank   参数占比
-----------------------------------------
Q, K 投影       8-16       ~15%
V, O 投影       16-32      ~20%  
FFN 上投影      32-64      ~35%
FFN 下投影      16-32      ~25%
Cross-Attn     32-64      ~5% (如果有)
```

**实现细节：**
```python
class LoRALayer(nn.Module):
    def __init__(self, in_features, out_features, rank=16, alpha=16):
        super().__init__()
        self.scaling = alpha / rank
        
        # 低秩矩阵
        self.lora_A = nn.Parameter(torch.randn(rank, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))
        
        # 初始化
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        
    def forward(self, x, base_output):
        # base_output 是原始层的输出
        lora_output = (x @ self.lora_A.T @ self.lora_B.T) * self.scaling
        return base_output + lora_output
```

### 3.3.2 QLoRA 的量化策略

QLoRA 结合 4-bit 量化和 LoRA，大幅降低显存占用：

**量化流程：**
```
原始模型 (16-bit) → NF4 量化 (4-bit) + LoRA 适配器 (16-bit)
显存节省: ~75% (相比全精度)
```

**NF4（NormalFloat4）量化：**
```python
def quantize_nf4(tensor):
    """4-bit NormalFloat 量化"""
    # 1. 归一化到 [-1, 1]
    absmax = tensor.abs().max()
    tensor_normalized = tensor / absmax
    
    # 2. 量化到 16 个级别
    quantization_levels = [
        -1.0, -0.6961, -0.5250, -0.3949, 
        -0.2844, -0.1848, -0.0911, 0.0,
        0.0796, 0.1609, 0.2461, 0.3379,
        0.4407, 0.5626, 0.7230, 1.0
    ]
    
    # 3. 找最近的量化级别
    quantized = quantize_to_nearest(tensor_normalized, quantization_levels)
    
    return quantized, absmax  # 保存 scale 用于反量化
```

**双重量化（Double Quantization）：**
```
第一次量化: 模型权重 → 4-bit
第二次量化: 量化常数 → 8-bit
额外节省: ~0.37 bit/参数
```

**QLoRA 训练配置：**
```python
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

# Paged Optimizer 节省优化器内存
optimizer = PagedAdamW32bit(
    model.parameters(),
    lr=2e-4,
    weight_decay=0.01,
    optim_bits=32  # 优化器状态保持 32-bit
)
```

### 3.3.3 Adapter 层的设计选择

Adapter 通过插入小型网络模块来实现参数高效微调：

**标准 Adapter 架构：**
```
输入 → LayerNorm → Down-projection → 激活 → Up-projection → 残差连接
  ↓                                                               ↑
  └──────────────────────────────────────────────────────────────┘
```

**VLM 中的 Adapter 变体：**

1. **Sequential Adapter：**
```python
class SequentialAdapter(nn.Module):
    def __init__(self, dim, reduction_factor=16):
        super().__init__()
        hidden_dim = dim // reduction_factor
        self.down_proj = nn.Linear(dim, hidden_dim)
        self.activation = nn.GELU()
        self.up_proj = nn.Linear(hidden_dim, dim)
        
    def forward(self, x):
        residual = x
        x = self.down_proj(x)
        x = self.activation(x)
        x = self.up_proj(x)
        return x + residual
```

2. **Parallel Adapter：**
```python
class ParallelAdapter(nn.Module):
    """并行处理，减少延迟"""
    def forward(self, x, original_output):
        adapter_output = self.adapter(x)
        return original_output + self.scale * adapter_output
```

3. **Cross-Modal Adapter：**
```python
class CrossModalAdapter(nn.Module):
    """专门处理视觉-语言交互"""
    def __init__(self, vision_dim, text_dim, hidden_dim):
        super().__init__()
        self.vision_proj = nn.Linear(vision_dim, hidden_dim)
        self.text_proj = nn.Linear(text_dim, hidden_dim)
        self.fusion = nn.MultiheadAttention(hidden_dim, num_heads=8)
        
    def forward(self, vision_features, text_features):
        v = self.vision_proj(vision_features)
        t = self.text_proj(text_features)
        fused, _ = self.fusion(t, v, v)  # text 作 query
        return fused
```

### 3.3.4 PEFT 方法对比与选择

**性能对比表：**
```
方法        参数量   显存占用  训练速度  效果(相对全量)
---------------------------------------------------------
全量微调     100%     100%      1.0x      100%
LoRA        0.1-1%   ~60%      1.5x      95-98%
QLoRA       0.1-1%   ~25%      1.2x      92-96%
Adapter     1-5%     ~70%      1.3x      93-97%
Prefix      <0.1%    ~50%      1.8x      85-92%
IA3         <0.01%   ~55%      1.6x      88-94%
```

**选择决策树：**
```
显存限制严格？
├─ 是 → QLoRA（4-bit量化 + LoRA）
└─ 否 → 需要最佳性能？
        ├─ 是 → 全量微调 or LoRA (r=64)
        └─ 否 → 推理速度优先？
                ├─ 是 → LoRA (可合并权重)
                └─ 否 → Adapter (灵活性高)
```

**组合策略：**
```python
# 混合 PEFT：不同层使用不同方法
config = {
    "vision_encoder": "frozen",  # 冻结
    "projection": "full",        # 全量微调
    "llm_layers_0_16": "lora",  # 底层用 LoRA
    "llm_layers_16_32": "adapter",  # 高层用 Adapter
}
```

**实践建议：**
1. **初始实验**：从 LoRA r=8 开始，逐步增加
2. **视觉编码器**：通常冻结或用很小的 rank（r=4）
3. **投影层**：建议全量微调，参数量小但重要
4. **任务适配**：简单任务用 LoRA，复杂任务考虑 Adapter

## 3.4 训练稳定性与收敛技巧

训练大规模 VLM 时经常遇到不稳定问题：损失突然爆炸、梯度消失、收敛缓慢等。本节介绍实用的稳定性技巧。

### 3.4.1 学习率调度策略

**VLM 常用调度器：**

1. **Cosine with Warmup：**
```python
def cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps):
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            # 线性 warmup
            return float(current_step) / float(max(1, num_warmup_steps))
        # Cosine 衰减
        progress = float(current_step - num_warmup_steps) / \
                  float(max(1, num_training_steps - num_warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))
    
    return LambdaLR(optimizer, lr_lambda)
```

2. **分阶段学习率：**
```
阶段1（预热）: lr = 1e-6 → 2e-4 (线性增长)
阶段2（主训练）: lr = 2e-4 (恒定或缓慢衰减)
阶段3（精调）: lr = 2e-4 → 1e-5 (cosine衰减)
```

**视觉编码器特殊处理：**
```python
# 不同组件不同学习率
param_groups = [
    {"params": vision_encoder.parameters(), "lr": 1e-5},  # 更小
    {"params": projection.parameters(), "lr": 5e-4},      # 更大
    {"params": language_model.parameters(), "lr": 2e-4},  # 标准
]
```

### 3.4.2 梯度裁剪与归一化

**梯度裁剪策略：**
```python
# 1. 全局梯度裁剪（推荐）
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

# 2. 分层梯度裁剪
for name, param in model.named_parameters():
    if "vision" in name:
        torch.nn.utils.clip_grad_norm_([param], max_norm=0.5)
    else:
        torch.nn.utils.clip_grad_norm_([param], max_norm=1.0)
```

**梯度监控：**
```python
def monitor_gradients(model):
    """监控梯度统计信息"""
    grad_stats = {}
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_stats[name] = {
                "mean": param.grad.mean().item(),
                "std": param.grad.std().item(),
                "max": param.grad.abs().max().item(),
            }
    return grad_stats

# 异常检测
if any(stat["max"] > 100 for stat in grad_stats.values()):
    logger.warning("梯度爆炸风险！")
```

### 3.4.3 权重初始化技巧

**关键组件初始化：**
```python
def init_vlm_weights(model):
    # 1. 投影层：Xavier 初始化
    if hasattr(model, 'visual_projection'):
        nn.init.xavier_uniform_(model.visual_projection.weight)
        nn.init.zeros_(model.visual_projection.bias)
    
    # 2. LoRA 层：接近零初始化
    for name, param in model.named_parameters():
        if "lora_B" in name:
            nn.init.zeros_(param)  # B 矩阵初始化为0
        elif "lora_A" in name:
            nn.init.kaiming_uniform_(param, a=math.sqrt(5))
    
    # 3. Layer Scale：小值初始化
    if hasattr(model, 'layer_scale'):
        nn.init.constant_(model.layer_scale, 1e-4)
```

**稳定性技巧：**
```
初始化检查清单：
□ 投影层不能太大（std < 0.02）
□ LoRA B 矩阵初始为 0
□ Layer Norm 权重 = 1, 偏置 = 0
□ 新增 token embedding 用已有 token 平均值
```

### 3.4.4 早停与 Checkpoint 策略

**智能 Checkpoint：**
```python
class SmartCheckpointer:
    def __init__(self, patience=3, delta=0.001):
        self.patience = patience
        self.delta = delta
        self.best_score = None
        self.counter = 0
        
    def should_save(self, val_loss):
        if self.best_score is None:
            self.best_score = val_loss
            return True
        
        if val_loss < self.best_score - self.delta:
            self.best_score = val_loss
            self.counter = 0
            return True
        else:
            self.counter += 1
            return False
    
    def should_stop(self):
        return self.counter >= self.patience
```

**Checkpoint 管理：**
```python
checkpoint = {
    'epoch': epoch,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'scheduler_state_dict': scheduler.state_dict(),
    'best_val_loss': best_val_loss,
    'training_history': {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'learning_rates': lrs,
    }
}

# 保存策略
save_strategies = {
    "best": "checkpoint_best.pt",        # 最佳验证性能
    "latest": "checkpoint_latest.pt",    # 最新状态
    "periodic": f"checkpoint_epoch_{epoch}.pt",  # 定期保存
}
```

## Case Study: Qwen-VL 的三阶段训练策略实战

Qwen-VL 采用渐进式三阶段训练策略，从大规模预训练到精细指令微调，实现了优秀的多模态性能。

### 阶段一：视觉-语言预训练

**目标**：建立基础的视觉-语言对齐能力

**数据配置：**
```
总量：1.4B 图文对
├── LAION-400M: 40%（网络爬取）
├── COYO-700M: 30%（韩语+英语）
├── CC12M: 15%（概念描述）
└── 内部数据: 15%（高质量筛选）
```

**训练配置：**
```python
stage1_config = {
    "vision_encoder": "frozen",  # OpenCLIP ViT-G/14
    "projection": "trainable",   # 新增的 Resampler
    "language_model": "trainable",  # Qwen-7B
    "batch_size": 2048,
    "learning_rate": 1e-4,
    "warmup_steps": 2000,
    "total_steps": 50000,
}
```

### 阶段二：多任务预训练

**目标**：学习多样化的视觉任务能力

**任务分布：**
```
任务类型         数据量    损失权重
--------------------------------------
图像描述         50M      0.3
VQA             30M      0.2
OCR             20M      0.2
Grounding       15M      0.15
Referring       10M      0.15
```

**关键技术：**
```python
# 动态分辨率处理
def dynamic_resolution(image, min_pixels=224*224, max_pixels=1024*1024):
    """保持宽高比的动态分辨率"""
    h, w = image.shape[:2]
    current_pixels = h * w
    
    if current_pixels < min_pixels:
        scale = math.sqrt(min_pixels / current_pixels)
    elif current_pixels > max_pixels:
        scale = math.sqrt(max_pixels / current_pixels)
    else:
        scale = 1.0
    
    new_h, new_w = int(h * scale), int(w * scale)
    # 确保是 14 的倍数（ViT patch size）
    new_h = (new_h // 14) * 14
    new_w = (new_w // 14) * 14
    
    return resize(image, (new_h, new_w))
```

### 阶段三：指令微调

**目标**：优化指令遵循和对话能力

**数据构成：**
```python
sft_data = {
    "high_quality_vqa": 200k,  # 人工标注
    "complex_reasoning": 150k,  # GPT-4V 生成
    "multi_turn_dialog": 100k,  # 多轮对话
    "rejection_sampling": 50k,  # 负样本
}
```

**LoRA 微调配置：**
```python
lora_config = LoRAConfig(
    r=64,  # 较大的 rank
    lora_alpha=16,
    target_modules=[
        "c_attn",  # Qwen 的注意力模块
        "c_proj", 
        "w1", "w2",  # MLP
    ],
    lora_dropout=0.05,
    task_type="CAUSAL_LM",
)

# 只微调语言模型部分
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
total_params = sum(p.numel() for p in model.parameters())
print(f"可训练参数: {trainable_params/1e6:.2f}M ({100*trainable_params/total_params:.2f}%)")
# 输出: 可训练参数: 384.00M (4.92%)
```

**训练曲线监控：**
```
       Loss
   3.5 |
   3.0 |  Stage 1
   2.5 |    ╲___
   2.0 |         ╲__ Stage 2
   1.5 |             ╲____
   1.0 |                  ╲___ Stage 3
   0.5 |                      ╲________
       |__|__|__|__|__|__|__|__|__|__|__
         10k  20k  30k  40k  50k  60k  Steps
```

## 高级话题

### 视觉编码器解冻时机

**解冻策略对比：**
```
策略            优点                缺点              适用场景
-----------------------------------------------------------------
始终冻结        省显存、训练快      可能欠拟合        数据与预训练相似
从头解冻        充分适应新任务      易过拟合、慢      大规模新领域数据
阶段性解冻      平衡性能与效率      需要经验调参      通用场景（推荐）
```

**阶段性解冻实践：**
```python
def staged_unfreeze(model, current_step, total_steps):
    """渐进解冻视觉编码器"""
    progress = current_step / total_steps
    
    if progress < 0.5:
        # 前 50%: 全部冻结
        freeze_vision_encoder(model)
    elif progress < 0.8:
        # 50-80%: 解冻最后 4 层
        for i, layer in enumerate(model.vision_encoder.layers):
            if i < len(model.vision_encoder.layers) - 4:
                freeze_layer(layer)
            else:
                unfreeze_layer(layer)
    else:
        # 最后 20%: 全部解冻，但用更小学习率
        unfreeze_vision_encoder(model)
        # 视觉编码器学习率 = 0.1 * 基础学习率
```

### LoRA Rank 自适应选择

**基于重要性的 Rank 分配：**
```python
def compute_layer_importance(model, dataloader, num_samples=100):
    """计算各层的 Fisher 信息矩阵迹"""
    importance_scores = {}
    
    for batch in dataloader[:num_samples]:
        outputs = model(batch)
        loss = compute_loss(outputs, batch['labels'])
        
        for name, param in model.named_parameters():
            if param.requires_grad:
                grad = torch.autograd.grad(loss, param, retain_graph=True)[0]
                if name not in importance_scores:
                    importance_scores[name] = 0
                importance_scores[name] += (grad ** 2).sum().item()
    
    # 归一化
    total = sum(importance_scores.values())
    for name in importance_scores:
        importance_scores[name] /= total
    
    return importance_scores

# 根据重要性分配 rank
def adaptive_rank_allocation(importance_scores, total_rank_budget=512):
    rank_allocation = {}
    for name, score in importance_scores.items():
        # rank ∈ [4, 64]
        rank = min(64, max(4, int(score * total_rank_budget)))
        # 确保是 4 的倍数（硬件友好）
        rank = (rank // 4) * 4
        rank_allocation[name] = rank
    return rank_allocation
```

### 混合精度训练的稳定性

**BF16 vs FP16 选择：**
```
特性          FP16            BF16
-----------------------------------------
动态范围      ±65504          ±3.4e38
精度          高              中
硬件支持      广泛            A100+
溢出风险      高              极低
推荐场景      推理为主        训练为主
```

**混合精度最佳实践：**
```python
# 自动混合精度配置
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler(
    init_scale=2.**16,  # 初始缩放因子
    growth_factor=2.0,   # 增长因子
    backoff_factor=0.5,  # 回退因子
    growth_interval=2000,  # 增长间隔
)

# 训练循环
for batch in dataloader:
    optimizer.zero_grad()
    
    with autocast(dtype=torch.bfloat16):  # 或 torch.float16
        outputs = model(batch)
        loss = compute_loss(outputs, batch['labels'])
    
    # 梯度缩放
    scaler.scale(loss).backward()
    
    # 梯度裁剪（在缩放空间）
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    
    # 优化器步骤
    scaler.step(optimizer)
    scaler.update()
    
    # 监控溢出
    if scaler.get_scale() < 1.0:
        logger.warning(f"梯度溢出，当前 scale: {scaler.get_scale()}")
```

## 本章小结

本章系统介绍了 VLM 的监督微调策略，涵盖了从指令设计到训练优化的完整流程：

**核心要点回顾：**
1. **指令设计**：清晰的模板、合理的系统提示、多轮对话处理、视觉-语言对齐
2. **损失函数**：自回归损失、掩码策略、多任务平衡、grounding 损失
3. **PEFT 方法**：LoRA、QLoRA、Adapter 的原理与选择
4. **训练稳定性**：学习率调度、梯度裁剪、权重初始化、checkpoint 策略

**关键公式汇总：**
- 语言模型损失：$\mathcal{L}_{LM} = -\sum_{t=1}^{T} \log P(x_t | x_{<t}, I)$
- LoRA 分解：$W' = W + BA$，其中 $r \ll \min(d, k)$
- 多任务损失：$\mathcal{L}_{total} = \sum_{i=1}^{N} w_i \mathcal{L}_i$
- 不确定性加权：$\mathcal{L}_{total} = \sum_{i=1}^{N} \frac{1}{2\sigma_i^2} \mathcal{L}_i + \log \sigma_i$

## 练习题

### 基础题（理解概念）

**题 1：指令模板设计**
设计一个支持多图像输入和 CoT（Chain of Thought）推理的指令模板。要求能够处理图像间的比较任务。

<details>
<summary>💡 提示</summary>
考虑：1) 如何标记不同图像 2) CoT 的步骤分隔 3) 图像引用方式
</details>

<details>
<summary>📝 参考答案</summary>

```
System: 你是一个视觉推理助手，请一步步分析问题。

User: 比较 <image_1> 和 <image_2>，找出主要差异。Assistant: 让我逐步分析：
步骤1：观察图像1的主要元素...
步骤2：观察图像2的主要元素...
步骤3：对比差异...
结论：主要差异包括...
```
</details>

**题 2：LoRA Rank 选择**
给定一个 7B 参数的 VLM，显存限制为 24GB，如何选择合适的 LoRA rank？考虑训练效率和模型性能的权衡。

<details>
<summary>💡 提示</summary>
计算不同 rank 下的参数量和显存占用，考虑梯度和优化器状态
</details>

<details>
<summary>📝 参考答案</summary>

对于 7B 模型，24GB 显存下的 rank 选择：
- QLoRA 4-bit: r=64 可行（~20GB）
- LoRA 16-bit: r=16-32 合适（~18-22GB）
- 建议从 r=16 开始，监控验证集性能，逐步增加到 r=32
</details>

**题 3：多任务损失平衡**
三个任务的初始损失分别为：图像描述 2.5、VQA 3.2、OCR 1.8。如何设置初始权重？

<details>
<summary>💡 提示</summary>
考虑损失量级差异和任务重要性
</details>

<details>
<summary>📝 参考答案</summary>

初始权重设置：
- 图像描述: 1.0 / 2.5 = 0.4
- VQA: 1.0 / 3.2 = 0.31
- OCR: 1.0 / 1.8 = 0.56
归一化后：[0.32, 0.25, 0.43]
</details>

### 挑战题（深入思考）

**题 4：梯度累积策略**
显存只够 batch_size=2，但最优 batch_size=32。设计一个考虑 VLM 特性的梯度累积方案。

<details>
<summary>💡 提示</summary>
考虑：1) 累积步数 2) 学习率缩放 3) 梯度裁剪时机
</details>

<details>
<summary>📝 参考答案</summary>

```python
accumulation_steps = 16  # 2 * 16 = 32
effective_batch_size = 32

for step, batch in enumerate(dataloader):
    loss = compute_loss(batch) / accumulation_steps
    loss.backward()
    
    if (step + 1) % accumulation_steps == 0:
        # 在累积完成后裁剪
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad()
        
# 学习率线性缩放
lr = base_lr * sqrt(effective_batch_size / base_batch_size)
```
</details>

**题 5：视觉编码器微调决策**
新任务是医学图像分析，与预训练数据（自然图像）差异很大。设计一个渐进式解冻方案。

<details>
<summary>💡 提示</summary>
医学图像的低层特征（边缘、纹理）可能相似，但高层语义差异大
</details>

<details>
<summary>📝 参考答案</summary>

三阶段解冻方案：
1. 阶段1（0-30%）：冻结所有层，只训练投影层
2. 阶段2（30-70%）：解冻后 50% 层，学习率 0.1x
3. 阶段3（70-100%）：全部解冻，前 50% 层用 0.01x 学习率，后 50% 用 0.1x
理由：保留低层通用特征，重点调整高层语义理解
</details>

**题 6：训练崩溃诊断**
训练到 40% 时损失突然变成 NaN。给出系统的排查流程和可能原因。

<details>
<summary>💡 提示</summary>
从数据、模型、优化器三个角度排查
</details>

<details>
<summary>📝 参考答案</summary>

排查流程：
1. **数据检查**：
   - 是否有损坏图像（全黑、全白）
   - 标签是否有异常值
   - Token ID 是否超出词表范围

2. **梯度监控**：
   - 检查梯度范数历史
   - 定位第一个 NaN 出现的层
   - 查看该 batch 的具体数据

3. **可能原因及解决**：
   - 学习率过大 → 降低学习率
   - 除零错误 → 添加 epsilon
   - FP16 溢出 → 切换到 BF16 或增大 loss scale
   - 某层未初始化 → 检查新增模块
</details>

**题 7：PEFT 组合优化**
设计一个针对 VLM 不同组件的混合 PEFT 策略，目标是在 16GB 显存限制下最大化性能。

<details>
<summary>💡 提示</summary>
不同组件的重要性和参数量不同
</details>

<details>
<summary>📝 参考答案</summary>

混合策略：
```python
config = {
    "vision_encoder": "frozen",         # 省显存
    "vision_projection": "full",        # 关键组件，参数少
    "llm_embed": "frozen",              # 词嵌入不动
    "llm_layers[0:8]": "lora_r8",      # 底层小 rank
    "llm_layers[8:24]": "lora_r16",    # 中层中 rank
    "llm_layers[24:32]": "lora_r32",   # 高层大 rank
    "llm_head": "lora_r8",             # 输出头小 rank
}
```
预计显存：~14GB，可训练参数：~200M
</details>

**题 8：开放性思考**
如果要设计下一代 VLM 的 SFT 策略，你认为最需要改进的三个方向是什么？

<details>
<summary>💡 提示</summary>
思考当前方法的局限性和实际应用需求
</details>

<details>
<summary>📝 参考答案</summary>

三个改进方向：
1. **动态计算分配**：根据图像复杂度动态调整计算资源，简单图像用少量 token，复杂图像用更多
2. **主动学习**：训练过程中自动识别模型薄弱环节，动态调整数据采样策略
3. **跨模态一致性**：设计更好的对齐机制，确保视觉理解和语言生成的一致性，减少幻觉

理由：当前 SFT 策略较为静态，没有充分利用模型的自适应能力
</details>

## 常见陷阱与错误 (Gotchas)

### 数据相关陷阱

**1. 图像 Token 计算错误**
```python
# 错误：忘记图像 token 占用
max_length = 2048  # 以为有 2048 个文本 token

# 正确：扣除图像占用
image_tokens = 576  # ViT-L/14 
text_budget = 2048 - image_tokens  # 实际只有 1472
```

**2. 响应截断问题**
```python
# 陷阱：响应被截断但仍计算损失
if len(tokens) > max_length:
    tokens = tokens[:max_length]  # 可能截断到响应中间
    
# 解决：确保完整响应或不计算损失
if len(tokens) > max_length:
    # 找到最后一个完整句子
    last_period = tokens[:max_length].rfind(period_token_id)
    tokens = tokens[:last_period+1]
```

### 训练相关陷阱

**3. LoRA 与正则化冲突**
```python
# 陷阱：对 LoRA 参数使用 weight decay
optimizer = AdamW(model.parameters(), weight_decay=0.01)

# 正确：LoRA 参数不用 weight decay
lora_params = [p for n, p in model.named_parameters() if 'lora' in n]
other_params = [p for n, p in model.named_parameters() if 'lora' not in n]
optimizer = AdamW([
    {'params': lora_params, 'weight_decay': 0.0},
    {'params': other_params, 'weight_decay': 0.01}
])
```

**4. 混合精度的 NaN 陷阱**
```python
# 陷阱：某些操作在 FP16 下不稳定
attention_scores = Q @ K.T / sqrt(d_k)  # 可能溢出

# 解决：关键操作用 FP32
with autocast(enabled=False):
    attention_scores = Q.float() @ K.float().T / sqrt(d_k)
```

**5. 梯度累积与 Batch Norm**
```python
# 陷阱：梯度累积时 BN 统计不准
# BN 只看当前 micro-batch，不是完整 batch

# 解决：使用 Layer Norm 或 RMSNorm
# 或者同步 BN（但会增加通信开销）
```

### 评估相关陷阱

**6. 生成长度偏差**
```python
# 陷阱：不同长度的生成影响评估
# 短回答可能 perplexity 更低但信息不足

# 解决：控制生成长度或使用长度归一化
score = log_prob / (length ** alpha)  # alpha ~ 0.6-0.8
```

**7. Teacher Forcing 与推理不一致**
```python
# 训练时：每步都用真实标签
# 推理时：用自己的预测，误差累积

# 缓解：Scheduled Sampling
if random.random() < teacher_forcing_ratio:
    input_token = ground_truth[t]
else:
    input_token = predicted[t-1]
```

### 调试技巧

**快速诊断检查点：**
```bash
# 1. 检查梯度
python -c "import torch; ckpt=torch.load('model.pt'); print([(k,v.abs().max().item()) for k,v in ckpt['grad_dict'].items() if v.abs().max() > 100])"

# 2. 检查权重分布
python -c "import torch; ckpt=torch.load('model.pt'); print([(k, v.std().item()) for k,v in ckpt['model_state_dict'].items() if 'weight' in k])"

# 3. 检查损失历史
python -c "import torch; import matplotlib.pyplot as plt; ckpt=torch.load('model.pt'); plt.plot(ckpt['loss_history']); plt.show()"
```

## 最佳实践检查清单

### 训练前准备

- [ ] **数据验证**
  - [ ] 所有图像可正常加载
  - [ ] 图像尺寸分布合理（没有极端大/小）
  - [ ] 文本长度分布检查
  - [ ] 特殊字符正确转义
  
- [ ] **模型配置**
  - [ ] 图像 token 数计算正确
  - [ ] 上下文长度设置合理
  - [ ] LoRA rank 根据显存选择
  - [ ] 检查点保存路径可写

- [ ] **训练配置**
  - [ ] 学习率设置（通常 1e-4 到 5e-4）
  - [ ] Warmup 步数（建议 3-10% 总步数）
  - [ ] 梯度裁剪阈值（通常 1.0）
  - [ ] 混合精度设置（BF16 优于 FP16）

### 训练中监控

- [ ] **性能指标**
  - [ ] GPU 利用率 > 90%
  - [ ] 显存使用稳定（无泄漏）
  - [ ] 训练速度（samples/sec）稳定
  - [ ] 数据加载不是瓶颈

- [ ] **模型指标**
  - [ ] 损失平稳下降
  - [ ] 梯度范数稳定
  - [ ] 学习率按计划衰减
  - [ ] 验证集指标提升

- [ ] **异常检测**
  - [ ] 无 NaN/Inf 出现
  - [ ] 无梯度爆炸/消失
  - [ ] 权重更新幅度合理
  - [ ] 生成样本质量检查

### 训练后验证

- [ ] **模型质量**
  - [ ] 基础能力保持（没有灾难性遗忘）
  - [ ] 新任务性能达标
  - [ ] 生成多样性适中
  - [ ] 无明显偏见或有害输出

- [ ] **部署准备**
  - [ ] 模型可正确加载
  - [ ] 推理速度满足要求
  - [ ] 量化后精度损失可接受
  - [ ] 边界case测试通过

- [ ] **文档完善**
  - [ ] 训练配置记录
  - [ ] 数据集版本记录
  - [ ] 性能基准记录
  - [ ] 已知问题记录

### 问题排查顺序

遇到问题时，按以下顺序排查：

1. **数据问题**（50% 的问题来源）
   - 检查当前 batch 的数据
   - 验证数据预处理流程
   
2. **配置问题**（30% 的问题来源）
   - 学习率是否过大
   - Batch size 是否合适
   
3. **代码问题**（20% 的问题来源）
   - 是否有维度不匹配
   - 是否有未初始化的参数

---

*通过本章的学习，您应该已经掌握了 VLM 监督微调的核心技术。下一章我们将探讨分布式训练与优化，进一步提升训练效率。*

[← 返回目录](index.md) | [下一章：分布式训练与优化 →](chapter4.md)