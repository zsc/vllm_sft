# 第 5 章：RLHF 基础与实现

经过监督微调（SFT）后的视觉语言模型虽然能够理解和响应多模态指令，但其输出往往无法完全符合人类的偏好和价值观。模型可能会产生事实性错误、生成有害内容，或在描述图像时出现幻觉。基于人类反馈的强化学习（RLHF）提供了一种直接优化人类偏好的训练范式，使模型的输出更加准确、有用且安全。本章将深入探讨如何在 VLM 中实施 RLHF，包括奖励模型训练、PPO 算法应用以及训练稳定性保障等关键技术。

## 学习目标

完成本章学习后，您将能够：

- **理解 RLHF 的核心原理**：掌握从偏好数据到策略优化的完整流程
- **构建多模态奖励模型**：设计适合 VLM 特点的奖励函数和模型架构
- **实施 PPO 训练**：正确配置和调试 PPO 算法的关键超参数
- **处理 VLM 特有挑战**：解决视觉幻觉、多模态对齐等独特问题
- **保证训练稳定性**：诊断和修复奖励崩溃、KL 散度爆炸等常见问题
- **优化训练效率**：通过合理的架构和算法选择提升 3-5 倍训练速度

## 5.1 RLHF 概述与动机

### 为什么 VLM 需要 RLHF

监督微调虽然让模型学会了遵循指令的基本能力，但存在几个根本性限制：

1. **幻觉问题严重**：VLM 经常描述图像中不存在的物体或细节，这种视觉幻觉比纯文本 LLM 的事实性错误更难通过 SFT 解决。

2. **偏好对齐困难**：人类对图像描述的偏好是主观且上下文相关的。同一张图片，用户可能期望简洁概括或详细分析，SFT 数据难以覆盖所有偏好模式。

3. **安全性挑战**：VLM 需要同时处理视觉和文本两个模态的安全问题，包括识别并拒绝处理有害图像内容。

4. **评估指标失配**：传统的 BLEU、CIDEr 等指标与人类判断相关性较低，直接优化这些指标反而可能降低实际体验。

RLHF 通过引入人类反馈作为优化信号，直接对齐模型输出与人类偏好，从根本上解决这些问题。

### RLHF 的核心流程

VLM 的 RLHF 训练通常包含三个阶段：

```
阶段 1: 监督微调（SFT）
├── 输入: (图像, 指令) 对
├── 输出: 初始策略模型 π_SFT
└── 目标: 基础指令遵循能力

阶段 2: 奖励模型训练（RM）
├── 输入: 人类偏好数据 {(x, y_win, y_lose)}
├── 输出: 奖励模型 R(x, y)
└── 目标: 学习人类偏好函数

阶段 3: 强化学习优化（PPO）
├── 输入: 策略模型 π + 奖励模型 R
├── 输出: 优化后的策略 π_RLHF
└── 目标: 最大化期望奖励同时控制 KL 散度
```

### VLM 中 RLHF 的独特挑战

相比纯文本 LLM，VLM 的 RLHF 面临额外的技术挑战：

**1. 多模态奖励建模复杂性**

奖励模型需要同时理解视觉和语言的对齐关系。一个高质量的回答不仅要语言流畅，更要准确反映图像内容。这要求奖励模型具备：

- 细粒度的视觉理解能力（物体、属性、关系）
- 跨模态一致性判断（文本是否准确描述图像）
- 上下文相关的质量评估（不同任务的评判标准不同）

**2. 计算和内存开销激增**

```
内存占用对比（以 13B 模型为例）：
纯文本 LLM RLHF:
- Actor 模型: 26GB (bf16)
- Critic 模型: 26GB
- Reference 模型: 26GB (冻结)
- 总计: ~78GB

VLM RLHF:
- Actor 模型: 26GB + Vision Encoder 4GB = 30GB
- Critic 模型: 30GB
- Reference 模型: 30GB (冻结)
- 图像缓存: 10-20GB (取决于批次大小)
- 总计: ~100GB+
```

**3. 训练不稳定性加剧**

视觉特征的高维度和多样性使得奖励信号的方差更大，容易导致：

- 奖励崩溃：模型找到视觉特征的捷径，产生高奖励但无意义的输出
- KL 散度爆炸：视觉条件下的策略分布更容易偏离参考分布
- 梯度爆炸：多模态交互层的梯度不稳定

### 与纯文本 RLHF 的关键差异

| 维度 | 纯文本 LLM | VLM | 影响 |
|------|------------|-----|------|
| **输入复杂度** | 文本序列 | 文本+图像 | 需要更大批次缓存 |
| **奖励信号** | 基于文本质量 | 需考虑跨模态对齐 | 奖励模型设计更复杂 |
| **幻觉类型** | 事实性错误 | 视觉幻觉+事实错误 | 需要专门的幻觉惩罚 |
| **计算开销** | 基准 | 1.5-2倍 | 训练成本显著增加 |
| **数据标注** | 文本偏好 | 需要理解图像内容 | 标注成本和难度更高 |

### 技术路线选择

实践中有三种主要的 RLHF 实现路线：

**1. 全模型 RLHF**
- 优点：效果最好，可以充分优化多模态交互
- 缺点：计算开销巨大，需要 8×A100 以上资源
- 适用：资源充足的研究团队

**2. LoRA-based RLHF**
- 优点：显存需求降低 50%，训练稳定
- 缺点：效果略有下降（5-10%）
- 适用：大多数实践场景

**3. 仅语言模型 RLHF**
- 优点：复用文本 RLHF 基础设施，实现简单
- 缺点：无法优化视觉编码器，改进有限
- 适用：快速原型验证

## 5.2 奖励模型的构建与训练

奖励模型是 RLHF 的核心组件，它将人类偏好转化为可优化的数值信号。在 VLM 场景下，奖励模型需要准确评估多模态输出的质量，这比纯文本场景复杂得多。

### 偏好数据收集策略

高质量的偏好数据是训练优秀奖励模型的基础。VLM 的偏好数据收集需要特别注意以下几点：

**1. 数据收集流程设计**

```
标准收集流程：
1. 采样阶段
   ├── 输入: (图像, 指令) 对
   ├── 生成: 使用不同温度/策略生成 K 个回答（K=4-7）
   └── 去重: 移除相似度 > 0.9 的回答

2. 标注阶段
   ├── 展示: 向标注者展示图像、指令和候选回答
   ├── 排序: 标注者对回答进行全排序或成对比较
   └── 质检: 计算标注者间一致性（Kappa > 0.6）

3. 数据构造
   ├── 成对比较: 从排序中提取 (chosen, rejected) 对
   ├── 权重分配: 根据排名差距设置样本权重
   └── 平衡处理: 确保正负样本比例合理（1:1 到 1:3）
```

**2. 多样性保证策略**

偏好数据需要覆盖多种失败模式，避免奖励模型过拟合：

- **任务多样性**：包含图像描述、视觉问答、推理等多种任务
- **错误类型覆盖**：
  - 幻觉错误（描述不存在的物体）
  - 属性错误（颜色、数量、位置错误）
  - 关系错误（物体间关系描述错误）
  - 逻辑错误（推理过程有误）
- **难度梯度**：从明显错误到细微差异的样本都要包含

**3. 标注质量控制**

VLM 偏好标注的挑战在于标注者需要同时理解图像和文本：

```python
# 标注指南示例
标注原则优先级：
1. 事实准确性 (40%)：描述是否与图像内容一致
2. 完整性 (25%)：是否回答了用户的问题
3. 相关性 (20%)：是否聚焦于问题相关内容
4. 流畅性 (15%)：语言表达是否自然

特殊情况处理：
- 当两个回答都包含错误时，选择错误较少的
- 当事实都正确时，优先选择信息量大的
- 对于主观问题，考虑回答的合理性和论证质量
```

### 多模态奖励模型架构

奖励模型的架构设计直接影响其判别能力和训练效率：

**1. 基础架构选择**

```
方案 A: 独立奖励头
┌─────────┐     ┌─────────┐
│  Vision │────▶│         │
│ Encoder │     │  Fusion │────▶ [Language Model] ────▶ [Reward Head]
└─────────┘     │  Layer  │                               (单个标量)
                └─────────┘
优点：参数效率高，可复用 SFT 模型
缺点：表达能力受限

方案 B: 序列级奖励建模
┌─────────┐     ┌─────────┐
│  Vision │────▶│         │
│ Encoder │     │  Fusion │────▶ [Language Model] ────▶ [Token Rewards]
└─────────┘     │  Layer  │                               (序列长度)
                └─────────┘                                    ↓
                                                          [Aggregation]
优点：可以细粒度建模，识别具体错误位置
缺点：训练复杂，需要更多标注
```

**2. 关键设计决策**

- **参数共享策略**：奖励模型通常从 SFT 模型初始化，可以选择：
  - 全参数微调：效果最好但开销大
  - 冻结视觉编码器：平衡效果和效率
  - LoRA 微调：显存友好，适合资源受限场景

- **池化策略**：如何从序列表示得到标量奖励
  - 最后一个 token：简单但可能丢失信息
  - 加权平均：考虑所有 token 但需要设计权重
  - 注意力池化：学习权重，更灵活

- **归一化方案**：保证奖励值的稳定性
  - Batch 归一化：训练时稳定但推理时需要统计量
  - Layer 归一化：更稳定，推荐使用
  - 标准化到 [-1, 1]：便于后续 PPO 训练

### 训练技巧与避坑指南

**1. 损失函数设计**

标准的 Bradley-Terry 模型：

$$\mathcal{L}_{BT} = -\mathbb{E}_{(x,y_w,y_l)} \left[ \log \sigma(r(x,y_w) - r(x,y_l)) \right]$$

实践改进：

$$\mathcal{L}_{total} = \mathcal{L}_{BT} + \lambda_1 \mathcal{L}_{margin} + \lambda_2 \mathcal{L}_{reg}$$

其中：
- $\mathcal{L}_{margin}$：确保好坏回答的奖励差距足够大
- $\mathcal{L}_{reg}$：防止奖励值过大，通常使用 L2 正则

```python
# 边际损失实现
margin_loss = F.relu(margin - (r_chosen - r_rejected))
# margin 通常设为 0.5-1.0
```

**2. 训练稳定性技巧**

- **梯度裁剪**：必须使用，clip_norm 设为 1.0
- **学习率预热**：前 10% 步数线性预热
- **早停策略**：验证集准确率不再提升时停止
- **数据增强**：对同一图像使用不同 crop/augmentation

**3. 常见问题诊断**

| 问题 | 现象 | 解决方案 |
|------|------|----------|
| 奖励崩溃 | 所有输出奖励趋于相同 | 增加 margin loss，检查数据质量 |
| 过拟合 | 训练集准确率 > 90%，验证集 < 70% | 增加 dropout，减小学习率，数据增强 |
| 奖励分布偏斜 | 奖励值集中在极端值 | 调整归一化策略，使用 tanh 激活 |
| 视觉偏见 | 只看图像忽略文本 | 增加纯文本负样本，平衡模态权重 |

**4. 幻觉惩罚机制**

专门针对视觉幻觉设计的奖励调整：

```python
# 幻觉检测与惩罚
def hallucination_penalty(image_features, text_output):
    # 1. 提取文本中提到的物体
    mentioned_objects = extract_objects(text_output)
    
    # 2. 使用目标检测模型验证
    detected_objects = object_detector(image_features)
    
    # 3. 计算惩罚
    false_mentions = mentioned_objects - detected_objects
    penalty = -alpha * len(false_mentions)
    
    return penalty

# 集成到总奖励中
final_reward = base_reward + hallucination_penalty
```

## 5.3 PPO 算法详解

Proximal Policy Optimization (PPO) 是 RLHF 中最常用的强化学习算法。它通过限制策略更新幅度来保证训练稳定性，特别适合大规模语言模型的优化。

### PPO 核心原理回顾

PPO 的核心思想是在最大化期望奖励的同时，限制新策略与旧策略的差异：

**目标函数**：

$$\mathcal{L}^{PPO}(\theta) = \mathbb{E}_t \left[ \min \left( r_t(\theta) \hat{A}_t, \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon) \hat{A}_t \right) \right]$$

其中：
- $r_t(\theta) = \frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{old}}(a_t|s_t)}$ 是重要性采样比率
- $\hat{A}_t$ 是优势函数估计
- $\epsilon$ 是裁剪参数（通常 0.1-0.2）

**VLM 场景下的状态-动作定义**：

```
状态 s_t = (图像 I, 指令 q, 已生成文本 y_{<t})
动作 a_t = 下一个 token y_t
奖励 r_t = {
    0,                  if t < T (中间步骤)
    R(I, q, y_{1:T}),  if t = T (序列结束)
}
```

### VLM 特定的 PPO 改进

标准 PPO 在 VLM 上直接应用会遇到特殊挑战，需要针对性改进：

**1. 多模态价值函数设计**

价值函数需要准确估计多模态状态的未来回报：

```python
class MultiModalValueHead(nn.Module):
    def __init__(self, hidden_dim, vision_dim):
        super().__init__()
        # 视觉特征投影
        self.vision_proj = nn.Linear(vision_dim, hidden_dim)
        # 跨模态注意力
        self.cross_attention = CrossAttention(hidden_dim)
        # 价值预测头
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1)
        )
    
    def forward(self, text_hidden, vision_features):
        # 融合视觉信息
        vision_proj = self.vision_proj(vision_features)
        fused = self.cross_attention(text_hidden, vision_proj)
        # 预测价值
        value = self.value_head(fused.mean(dim=1))
        return value
```

**2. 优势函数估计改进**

GAE (Generalized Advantage Estimation) 在 VLM 中需要特别处理：

$$\hat{A}_t = \sum_{l=0}^{\infty} (\gamma \lambda)^l \delta_{t+l}$$

其中 $\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)$

VLM 特殊处理：
- **稀疏奖励问题**：只在序列末尾有奖励，中间步骤使用 shaping reward
- **视觉条件折扣**：根据图像复杂度动态调整 $\gamma$

```python
def compute_advantages_vlm(rewards, values, vision_complexity):
    # 动态折扣因子
    gamma = 0.99 - 0.05 * vision_complexity  # 复杂图像使用更小折扣
    
    advantages = []
    gae = 0
    for t in reversed(range(len(rewards))):
        if t == len(rewards) - 1:
            next_value = 0
        else:
            next_value = values[t + 1]
        
        delta = rewards[t] + gamma * next_value - values[t]
        gae = delta + gamma * lambda_ * gae
        advantages.insert(0, gae)
    
    return advantages
```

**3. 批次构造策略**

VLM 的批次构造需要平衡视觉多样性和计算效率：

```
批次组织原则：
1. 图像去重：同一批次避免重复图像（节省视觉编码）
2. 长度排序：相似长度的序列放在一起（减少 padding）
3. 难度混合：简单和困难样本混合（稳定训练）
4. 模态平衡：确保纯文本和多模态样本都有
```

### KL 散度约束的重要性

KL 散度约束是防止策略崩溃的关键机制，在 VLM 中尤其重要：

**1. KL 惩罚项设计**

$$\mathcal{L}_{total} = \mathcal{L}^{PPO} - \beta \cdot \text{KL}[\pi_\theta || \pi_{ref}]$$

其中 $\beta$ 需要自适应调整：

```python
class AdaptiveKLController:
    def __init__(self, init_kl_coef=0.1, target_kl=6.0):
        self.kl_coef = init_kl_coef
        self.target_kl = target_kl
        
    def update(self, current_kl):
        if current_kl > 1.5 * self.target_kl:
            self.kl_coef *= 1.5  # 增大惩罚
        elif current_kl < 0.5 * self.target_kl:
            self.kl_coef *= 0.75  # 减小惩罚
        
        # 裁剪范围
        self.kl_coef = np.clip(self.kl_coef, 0.001, 10.0)
        return self.kl_coef
```

**2. 视觉条件下的 KL 计算**

VLM 的 KL 散度需要考虑视觉条件：

$$\text{KL}_{vlm} = \mathbb{E}_{(I,q)} \left[ \text{KL}[\pi_\theta(\cdot|I,q) || \pi_{ref}(\cdot|I,q)] \right]$$

实践技巧：
- 对视觉 token 和文本 token 使用不同的 KL 权重
- 图像复杂度高时允许更大的 KL 散度
- 监控每个模态的 KL 贡献，防止单一模态主导

**3. KL 爆炸的预防与处理**

```python
def compute_kl_with_clipping(logits_new, logits_ref, attention_mask):
    # 计算 log 概率
    log_probs_new = F.log_softmax(logits_new, dim=-1)
    log_probs_ref = F.log_softmax(logits_ref, dim=-1)
    
    # KL 散度
    kl = (log_probs_ref.exp() * (log_probs_ref - log_probs_new)).sum(-1)
    
    # 裁剪异常值
    kl = torch.clamp(kl, min=0, max=100)
    
    # 应用 mask 并平均
    kl = (kl * attention_mask).sum() / attention_mask.sum()
    
    return kl
```

监控指标：
- KL 散度均值和方差
- 最大单步 KL
- 视觉/文本 KL 比率

当 KL > 10 时的紧急处理：
1. 立即降低学习率至 1/10
2. 增大 KL 惩罚系数 $\beta$
3. 回滚到上一个检查点
4. 检查是否有数据分布偏移

## 5.4 VLM 中的 RLHF 实践

将 RLHF 理论应用到实际 VLM 训练中需要精心设计流程和参数。本节提供经过验证的实践方案。

### 训练流程设计

**完整训练 Pipeline**：

```
Phase 1: 准备阶段（1-2 天）
├── SFT 模型验证
│   ├── 确保 SFT 模型收敛
│   ├── 验证生成质量基线
│   └── 冻结 SFT 权重作为参考模型
├── 奖励模型训练
│   ├── 收集偏好数据（10k-50k 对）
│   ├── 训练奖励模型（准确率 > 65%）
│   └── 验证奖励分布合理性
└── 环境配置
    ├── 设置多 GPU 并行
    ├── 配置梯度累积
    └── 准备监控工具

Phase 2: PPO 训练（3-5 天）
├── 预热阶段（10% steps）
│   ├── 小学习率（1e-7）
│   ├── 大 KL 系数（0.5）
│   └── 监控稳定性
├── 主训练阶段（80% steps）
│   ├── 正常学习率（1e-6）
│   ├── 自适应 KL 系数
│   └── 定期评估
└── 收尾阶段（10% steps）
    ├── 学习率衰减
    ├── 增大 KL 约束
    └── 选择最佳检查点

Phase 3: 后处理（1 天）
├── 模型评估
├── 消融实验
└── 部署准备
```

**数据流设计**：

```python
class RLHFDataPipeline:
    def __init__(self, batch_size=32, buffer_size=1000):
        self.batch_size = batch_size
        self.buffer = []
        self.buffer_size = buffer_size
        
    def generate_batch(self, model, prompts):
        """生成阶段：收集模型输出"""
        with torch.no_grad():
            outputs = []
            for prompt in prompts:
                # 多样性采样
                for temp in [0.7, 0.9, 1.1]:
                    output = model.generate(
                        prompt, 
                        temperature=temp,
                        do_sample=True,
                        max_length=512
                    )
                    outputs.append({
                        'prompt': prompt,
                        'response': output,
                        'temperature': temp
                    })
        return outputs
    
    def score_batch(self, reward_model, experiences):
        """评分阶段：计算奖励"""
        rewards = []
        for exp in experiences:
            reward = reward_model(exp['prompt'], exp['response'])
            # 添加长度惩罚
            length_penalty = -0.01 * len(exp['response'])
            exp['reward'] = reward + length_penalty
            rewards.append(exp)
        return rewards
    
    def update_model(self, model, experiences):
        """更新阶段：PPO 优化"""
        # 计算优势
        advantages = self.compute_advantages(experiences)
        
        # 多轮更新
        for _ in range(4):  # PPO epochs
            for batch in self.get_batches(experiences):
                loss = self.ppo_loss(model, batch, advantages)
                loss.backward()
                # 梯度裁剪
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
```

### 超参数选择策略

VLM RLHF 的超参数选择需要考虑模型规模、数据特点和计算资源：

**核心超参数推荐值**：

| 参数 | 7B 模型 | 13B 模型 | 34B+ 模型 | 说明 |
|------|---------|----------|-----------|------|
| **学习率** | 5e-7 | 1e-6 | 2e-6 | Actor 模型学习率 |
| **Critic 学习率** | 1e-6 | 2e-6 | 5e-6 | 通常是 Actor 的 2-5 倍 |
| **批次大小** | 32 | 64 | 128 | 每个 GPU 的批次 |
| **PPO epochs** | 4 | 4 | 2 | 每批数据的更新轮数 |
| **裁剪参数 ε** | 0.2 | 0.2 | 0.1 | 大模型用更小值 |
| **GAE λ** | 0.95 | 0.95 | 0.97 | 优势估计平滑度 |
| **折扣因子 γ** | 0.99 | 0.99 | 0.995 | 未来奖励折扣 |
| **KL 目标** | 6.0 | 6.0 | 3.0 | 目标 KL 散度 |
| **初始 KL 系数** | 0.1 | 0.2 | 0.5 | KL 惩罚初始值 |

**学习率调度策略**：

```python
def get_lr_scheduler(optimizer, num_training_steps):
    # 余弦退火 + 预热
    def lr_lambda(step):
        # 10% 预热
        warmup_steps = int(0.1 * num_training_steps)
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        
        # 余弦衰减
        progress = float(step - warmup_steps) / float(max(1, num_training_steps - warmup_steps))
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    
    return LambdaLR(optimizer, lr_lambda)
```

### 多模态特有问题处理

**1. 视觉编码器的处理策略**

```python
class VisionEncoderStrategy:
    """视觉编码器在 RLHF 中的处理策略"""
    
    @staticmethod
    def frozen_strategy():
        """策略 1：完全冻结"""
        # 优点：稳定、省显存
        # 缺点：无法优化视觉表示
        return {'vision_encoder': False, 'projection': True}
    
    @staticmethod
    def staged_unfreezing():
        """策略 2：阶段性解冻"""
        stages = [
            (0, 0.3, {'vision_encoder': False, 'projection': True}),    # 前 30%：冻结
            (0.3, 0.7, {'vision_encoder': 'last_layer', 'projection': True}),  # 中 40%：解冻最后层
            (0.7, 1.0, {'vision_encoder': True, 'projection': True})     # 后 30%：全部解冻
        ]
        return stages
    
    @staticmethod
    def lora_strategy():
        """策略 3：LoRA 微调"""
        # 在视觉编码器中插入 LoRA
        return {
            'vision_lora_rank': 16,
            'vision_lora_alpha': 32,
            'vision_lora_dropout': 0.1
        }
```

**2. 多模态奖励对齐**

确保文本和视觉贡献平衡的奖励：

```python
def compute_multimodal_reward(text_reward, vision_reward, alignment_score):
    """
    组合多个奖励信号
    text_reward: 文本质量评分
    vision_reward: 视觉相关性评分
    alignment_score: 跨模态对齐评分
    """
    # 自适应权重
    if alignment_score < 0.5:
        # 对齐差时，更重视对齐
        weights = {'text': 0.2, 'vision': 0.2, 'alignment': 0.6}
    else:
        # 对齐好时，平衡各项
        weights = {'text': 0.4, 'vision': 0.3, 'alignment': 0.3}
    
    final_reward = (
        weights['text'] * text_reward +
        weights['vision'] * vision_reward +
        weights['alignment'] * alignment_score
    )
    
    return final_reward
```

**3. 幻觉抑制机制**

```python
class HallucinationSupressor:
    def __init__(self, detection_model, penalty_weight=0.5):
        self.detector = detection_model
        self.penalty_weight = penalty_weight
        
    def compute_penalty(self, image, generated_text):
        # 检测幻觉
        hallucinations = self.detector.detect(image, generated_text)
        
        # 分级惩罚
        penalties = {
            'object_hallucination': -2.0,  # 物体幻觉最严重
            'attribute_error': -1.0,        # 属性错误次之
            'relation_error': -0.5          # 关系错误较轻
        }
        
        total_penalty = 0
        for h_type, h_count in hallucinations.items():
            total_penalty += penalties.get(h_type, 0) * h_count
            
        return self.penalty_weight * total_penalty
```

## 5.5 训练稳定性与调试

RLHF 训练的不稳定性是实践中的主要挑战。本节提供系统的诊断和解决方案。

### 常见不稳定现象诊断

**1. 奖励崩溃模式识别**

```
症状观察表：
┌─────────────────┬──────────────────┬─────────────────┐
│ 现象            │ 可能原因          │ 诊断方法         │
├─────────────────┼──────────────────┼─────────────────┤
│ 奖励持续上升    │ 奖励黑客          │ 检查生成文本质量 │
│ 奖励突然下降    │ 策略崩溃          │ 查看 KL 散度     │
│ 奖励震荡        │ 学习率过大        │ 梯度范数监控     │
│ 奖励停滞        │ 局部最优/过拟合   │ 验证集表现       │
└─────────────────┴──────────────────┴─────────────────┘
```

**2. 快速诊断脚本**

```python
class RLHFDiagnostics:
    def __init__(self, threshold_config):
        self.thresholds = threshold_config
        self.history = defaultdict(list)
        
    def diagnose(self, metrics):
        """实时诊断训练状态"""
        issues = []
        
        # 检查奖励异常
        if metrics['reward'] > self.thresholds['max_reward']:
            issues.append(('CRITICAL', 'Reward hacking detected'))
        
        # 检查 KL 散度
        if metrics['kl_div'] > self.thresholds['max_kl']:
            issues.append(('WARNING', f"KL divergence too high: {metrics['kl_div']:.2f}"))
        
        # 检查梯度
        if metrics['grad_norm'] > self.thresholds['max_grad']:
            issues.append(('WARNING', 'Gradient explosion risk'))
        
        # 检查生成长度
        avg_length = np.mean(metrics['response_lengths'])
        if avg_length < 20 or avg_length > 500:
            issues.append(('INFO', f'Abnormal response length: {avg_length:.0f}'))
        
        # 趋势分析
        self.history['reward'].append(metrics['reward'])
        if len(self.history['reward']) > 100:
            recent_std = np.std(self.history['reward'][-100:])
            if recent_std > self.thresholds['reward_std']:
                issues.append(('WARNING', 'Reward instability detected'))
        
        return issues
```

### 奖励黑客（Reward Hacking）防范

奖励黑客是模型找到欺骗奖励模型的捷径，产生高奖励但无意义的输出。

**1. 典型奖励黑客模式**

```
VLM 常见奖励黑客行为：
1. 重复描述：不断重复图像中的显著物体
2. 模板化回答：使用固定句式获得稳定奖励
3. 过度详细：生成冗长但信息量低的描述
4. 关键词堆砌：堆积奖励模型偏好的词汇
5. 忽略指令：只描述图像，不回答问题
```

**2. 防范机制实现**

```python
class RewardHackingDefense:
    def __init__(self):
        self.detectors = {
            'repetition': self.detect_repetition,
            'template': self.detect_template,
            'keyword_stuffing': self.detect_keyword_stuffing,
            'length_gaming': self.detect_length_gaming
        }
        
    def detect_repetition(self, text):
        """检测重复模式"""
        sentences = text.split('.')
        if len(sentences) < 3:
            return 0
        
        # 计算句子相似度
        similarities = []
        for i in range(len(sentences)-1):
            sim = self.sentence_similarity(sentences[i], sentences[i+1])
            similarities.append(sim)
        
        # 高相似度表示重复
        return max(similarities) if similarities else 0
    
    def detect_template(self, responses):
        """检测模板化回答"""
        # 提取结构特征
        structures = [self.extract_structure(r) for r in responses]
        
        # 计算结构多样性
        unique_structures = len(set(structures))
        diversity_score = unique_structures / len(structures)
        
        return 1 - diversity_score  # 低多样性 = 高模板化
    
    def apply_defense(self, reward, text, detection_scores):
        """应用防御机制调整奖励"""
        penalty = 0
        
        for detector_name, score in detection_scores.items():
            if score > 0.7:  # 高置信度检测
                penalty += self.penalty_weights[detector_name] * score
        
        # 应用惩罚
        adjusted_reward = reward - penalty
        
        # 确保不会过度惩罚
        return max(adjusted_reward, reward * 0.3)
```

**3. 多样性奖励机制**

```python
def diversity_bonus(current_response, previous_responses, alpha=0.1):
    """
    鼓励多样性的奖励调整
    """
    if not previous_responses:
        return 0
    
    # 计算与历史回答的最小距离
    min_distance = float('inf')
    for prev in previous_responses[-10:]:  # 只看最近10个
        distance = edit_distance(current_response, prev) / max(len(current_response), len(prev))
        min_distance = min(min_distance, distance)
    
    # 距离越大，奖励越高
    bonus = alpha * min_distance
    return bonus
```

### 监控指标设计

全面的监控是保证训练稳定的关键：

**1. 核心监控指标**

```python
class RLHFMonitor:
    def __init__(self):
        self.metrics = {
            # 奖励相关
            'reward_mean': [],
            'reward_std': [],
            'reward_max': [],
            'reward_min': [],
            
            # KL 散度
            'kl_div_mean': [],
            'kl_div_max': [],
            'kl_per_token': [],
            
            # 生成质量
            'response_length': [],
            'unique_tokens_ratio': [],
            'perplexity': [],
            
            # 训练稳定性
            'grad_norm': [],
            'value_loss': [],
            'policy_loss': [],
            'entropy': [],
            
            # 视觉特定
            'vision_attention_entropy': [],
            'cross_modal_alignment': []
        }
    
    def log_step(self, batch_metrics):
        """记录每步指标"""
        for key, value in batch_metrics.items():
            if key in self.metrics:
                self.metrics[key].append(value)
    
    def get_dashboard_data(self):
        """生成监控面板数据"""
        dashboard = {}
        
        # 计算移动平均
        for key, values in self.metrics.items():
            if len(values) > 0:
                dashboard[f'{key}_ma10'] = np.mean(values[-10:])
                dashboard[f'{key}_ma100'] = np.mean(values[-100:])
        
        # 计算趋势
        if len(self.metrics['reward_mean']) > 100:
            recent = np.mean(self.metrics['reward_mean'][-50:])
            past = np.mean(self.metrics['reward_mean'][-100:-50])
            dashboard['reward_trend'] = (recent - past) / abs(past)
        
        return dashboard
```

**2. 实时告警系统**

```python
class AlertSystem:
    def __init__(self):
        self.alert_rules = [
            ('kl_div_mean > 10', 'CRITICAL', 'KL divergence explosion'),
            ('reward_std > 2', 'WARNING', 'Reward instability'),
            ('grad_norm > 100', 'CRITICAL', 'Gradient explosion'),
            ('response_length < 10', 'WARNING', 'Degenerate responses'),
            ('entropy < 0.1', 'WARNING', 'Low generation diversity')
        ]
    
    def check_alerts(self, metrics):
        alerts = []
        for rule, level, message in self.alert_rules:
            if self.evaluate_rule(rule, metrics):
                alerts.append({
                    'level': level,
                    'message': message,
                    'metrics': metrics,
                    'timestamp': time.time()
                })
        return alerts
```

## Case Study: LLaVA-RLHF 的人类偏好对齐实践

LLaVA-RLHF 是首个成功将 RLHF 应用于开源 VLM 的工作，其方法论值得深入分析。

### 数据收集与标注流程

LLaVA-RLHF 构建了包含 10k 比较对的高质量偏好数据集：

**1. 数据源选择**
- COCO 数据集：日常场景图像
- Visual Genome：复杂场景和关系
- A-OKVQA：需要推理的视觉问答

**2. 回答生成策略**
```
对每个 (图像, 问题) 对：
1. 使用 4 个不同模型生成回答：
   - LLaVA-13B (base)
   - LLaVA-13B-v1.5
   - GPT-4V (作为高质量参考)
   - 人工编写 (ground truth)

2. 温度采样增加多样性：
   - T=0.7, 0.9, 1.1 各生成一次
   - 总计 12 个候选回答

3. 去重和过滤：
   - 移除完全相同的回答
   - 过滤长度 < 10 或 > 500 的回答
   - 保留 4-6 个最多样的回答
```

**3. 标注协议**
```
标注者指南：
优先级 1: 事实准确性（是否正确描述图像）
优先级 2: 相关性（是否回答了问题）
优先级 3: 有用性（信息量和洞察深度）
优先级 4: 表达质量（清晰度和连贯性）

标注接口：
- 并排显示图像和问题
- 随机顺序展示候选回答
- 支持拖拽排序或成对比较
- 要求标注者解释排序理由
```

### 三阶段训练策略

**阶段 1：视觉指令微调（Visual Instruction Tuning）**
```
目标：建立基础的多模态理解能力
数据：595K 指令跟随样本
配置：
- 学习率: 2e-5 (第一轮), 2e-6 (第二轮)
- 批次大小: 128
- 训练轮数: 1 epoch
- 视觉编码器: 冻结 CLIP ViT-L/14

关键技巧：
- 两阶段训练：先训练投影层，再微调 LLM
- 数据配比：80% 多模态，20% 纯文本（保持语言能力）
```

**阶段 2：奖励模型训练**
```
架构：基于 LLaVA-13B + 线性奖励头
数据：10K 人类偏好对
配置：
- 学习率: 1e-6
- 批次大小: 64
- 训练步数: 3 epochs
- 损失函数: Bradley-Terry + Margin Loss

性能指标：
- 成对准确率: 67.3%
- 与人类一致性: Kappa = 0.62
- 验证集泛化: 65.1%
```

**阶段 3：PPO 强化学习**
```
配置：
- Actor 学习率: 5e-7
- Critic 学习率: 1e-6
- KL 系数: 初始 0.1，自适应调整
- 批次大小: 32
- PPO epochs: 4
- 训练步数: 50K

训练技巧：
1. 预热阶段（前 10%）：
   - 小学习率防止崩溃
   - 大 KL 惩罚保持稳定

2. 主训练阶段：
   - 每 1000 步评估验证集
   - 动态调整 KL 系数
   - 监控奖励黑客

3. 收尾阶段（后 10%）：
   - 线性衰减学习率
   - 选择最佳检查点（非最后一个）
```

### 效果评估与分析

**1. 定量评估结果**

| 指标 | LLaVA-13B | LLaVA-RLHF | 提升 |
|------|-----------|------------|------|
| **MMBench** | 67.7 | 71.3 | +3.6 |
| **幻觉率** | 31.2% | 18.7% | -12.5% |
| **人类偏好胜率** | - | 62.3% | - |
| **平均回答长度** | 89 tokens | 126 tokens | +41% |

**2. 定性改进分析**

```
改进 1：幻觉显著减少
Before: "图中有一只猫在桌子上，旁边有一个红色的球。"
       （实际图中无球）
After:  "图中有一只灰色的猫躺在木桌上。"

改进 2：细节描述更准确
Before: "这是一个房间。"
After:  "这是一个现代风格的客厅，有米色沙发、玻璃茶几和大窗户。"

改进 3：推理能力增强
Question: "为什么这个人戴着安全帽？"
Before: "因为他在工作。"
After:  "这个人戴着安全帽是因为他在建筑工地工作，这是安全规定要求的防护装备。"
```

**3. 失败模式分析**

尽管取得改进，仍存在一些问题：
- 过度保守：有时拒绝回答实际可以回答的问题
- 长度偏见：倾向于生成更长的回答，即使简短回答更合适
- 模态不平衡：某些情况下过度依赖语言模型先验，忽视视觉信息

## 高级话题

### 多模态奖励建模的挑战

多模态奖励建模面临独特的技术挑战，需要创新的解决方案：

**1. 跨模态一致性建模**

传统的奖励模型主要关注文本质量，但 VLM 需要同时评估跨模态一致性：

```python
class CrossModalConsistencyReward:
    """评估文本与图像的一致性"""
    
    def compute_consistency(self, image_features, text_embeddings):
        # 方法 1：对比学习相似度
        similarity = F.cosine_similarity(
            self.image_proj(image_features),
            self.text_proj(text_embeddings)
        )
        
        # 方法 2：细粒度对齐
        # 检测图像中的物体
        detected_objects = self.object_detector(image_features)
        # 提取文本中提到的实体
        mentioned_entities = self.entity_extractor(text_embeddings)
        # 计算重叠度
        overlap = len(detected_objects & mentioned_entities) / len(mentioned_entities)
        
        # 组合两种信号
        consistency_score = 0.6 * similarity + 0.4 * overlap
        return consistency_score
```

**2. 多粒度奖励设计**

不同任务需要不同粒度的奖励信号：

- **Token 级奖励**：识别具体的幻觉位置
- **句子级奖励**：评估逻辑连贯性
- **段落级奖励**：整体质量评估

**3. 组合奖励函数的优化**

$$R_{total} = \alpha R_{accuracy} + \beta R_{relevance} + \gamma R_{safety} + \delta R_{diversity}$$

挑战在于如何自动学习权重 $\alpha, \beta, \gamma, \delta$：

```python
class AdaptiveRewardWeighting:
    def __init__(self):
        self.weights = nn.Parameter(torch.ones(4) / 4)
        
    def forward(self, rewards_dict):
        # 归一化权重
        normalized_weights = F.softmax(self.weights, dim=0)
        
        # 计算加权奖励
        total_reward = sum(
            w * r for w, r in zip(normalized_weights, rewards_dict.values())
        )
        
        # 添加熵正则化，防止权重退化
        entropy = -(normalized_weights * normalized_weights.log()).sum()
        total_reward += 0.01 * entropy
        
        return total_reward
```

### 幻觉惩罚机制设计

视觉幻觉是 VLM 的主要问题，需要专门的检测和惩罚机制：

**1. 幻觉类型分类**

```
视觉幻觉分类体系：
├── 对象幻觉（Object Hallucination）
│   ├── 存在性错误：描述不存在的物体
│   └── 数量错误：物体数量描述错误
├── 属性幻觉（Attribute Hallucination）
│   ├── 颜色错误
│   ├── 大小错误
│   └── 材质错误
├── 关系幻觉（Relation Hallucination）
│   ├── 空间关系错误
│   └── 动作关系错误
└── 知识幻觉（Knowledge Hallucination）
    └── 错误的背景知识推断
```

**2. 分级惩罚策略**

```python
class HallucinationPenaltySchedule:
    def __init__(self):
        # 不同类型幻觉的基础惩罚
        self.base_penalties = {
            'object_existence': -2.0,    # 最严重
            'object_count': -1.5,
            'attribute': -1.0,
            'relation': -0.8,
            'knowledge': -0.5            # 相对较轻
        }
        
    def compute_penalty(self, hallucination_report, training_step):
        # 随训练进程增强惩罚
        severity_multiplier = min(2.0, 1.0 + training_step / 10000)
        
        total_penalty = 0
        for h_type, count in hallucination_report.items():
            base = self.base_penalties.get(h_type, -0.5)
            # 多个幻觉的超线性惩罚
            penalty = base * (count ** 1.2) * severity_multiplier
            total_penalty += penalty
            
        return total_penalty
```

**3. 主动幻觉预防**

```python
class HallucinationPrevention:
    def __init__(self, vision_grounder):
        self.grounder = vision_grounder
        
    def guided_generation(self, model, image, partial_text):
        """引导生成以减少幻觉"""
        # 1. 提取已生成文本中的实体
        entities = self.extract_entities(partial_text)
        
        # 2. 视觉接地验证
        grounded = self.grounder.verify(image, entities)
        
        # 3. 调整生成概率
        if not grounded:
            # 降低继续描述该实体的概率
            mask = self.create_entity_mask(entities[-1])
            # 应用到 logits
            logits = model.get_logits()
            logits[mask] -= 5.0  # 强惩罚
            
        return logits
```

### Constitutional AI 在 VLM 中的应用

Constitutional AI (CAI) 通过自我批评和修正来提升模型安全性和有用性：

**1. VLM 的 Constitutional 原则**

```python
VLM_CONSTITUTION = [
    # 准确性原则
    "只描述图像中实际可见的内容",
    "不对图像内容进行未经证实的推测",
    "承认视觉信息的局限性",
    
    # 安全性原则
    "不生成可能造成伤害的内容",
    "尊重图像中人物的隐私",
    "避免强化偏见和刻板印象",
    
    # 有用性原则
    "提供信息丰富且相关的回答",
    "根据用户需求调整详细程度",
    "承认不确定性而非猜测"
]
```

**2. 自我批评与修正流程**

```python
class ConstitutionalVLM:
    def __init__(self, base_model, constitution):
        self.model = base_model
        self.constitution = constitution
        
    def generate_with_critique(self, image, prompt):
        # 步骤 1：初始生成
        initial_response = self.model.generate(image, prompt)
        
        # 步骤 2：自我批评
        critique_prompt = f"""
        请评估以下回答是否违反了这些原则：
        {self.constitution}
        
        回答：{initial_response}
        """
        critique = self.model.generate(image, critique_prompt)
        
        # 步骤 3：修正
        if "违反" in critique:
            revision_prompt = f"""
            基于以下批评，修正回答：
            批评：{critique}
            原回答：{initial_response}
            """
            revised_response = self.model.generate(image, revision_prompt)
            return revised_response
        
        return initial_response
```

**3. Constitutional RLHF**

将 CAI 原则集成到 RLHF 训练中：

```python
def constitutional_reward(response, image, constitution):
    """基于 constitution 的奖励函数"""
    rewards = []
    
    for principle in constitution:
        # 评估是否遵守原则
        adherence = evaluate_adherence(response, image, principle)
        rewards.append(adherence)
    
    # 加权平均，关键原则权重更高
    weights = [2.0 if "安全" in p else 1.0 for p in constitution]
    final_reward = np.average(rewards, weights=weights)
    
    return final_reward
```

## 本章小结

本章深入探讨了 VLM 的 RLHF 训练，从理论基础到实践细节。关键要点包括：

**核心概念**：
- RLHF 通过人类反馈直接优化模型输出，解决 SFT 无法处理的偏好对齐问题
- VLM 的 RLHF 比纯文本更复杂，需要处理跨模态对齐和视觉幻觉
- PPO 算法需要针对 VLM 特点进行改进，特别是价值函数和 KL 约束设计

**关键技术**：
- 奖励模型需要同时评估文本质量和视觉一致性
- 多阶段训练策略：SFT → 奖励模型 → PPO 优化
- 稳定性保障机制：KL 散度控制、奖励黑客防范、实时监控

**实践经验**：
- 计算资源需求比纯文本 RLHF 高 1.5-2 倍
- 幻觉率可降低 40-60%，但需要专门的检测和惩罚机制
- Constitutional AI 可以进一步提升安全性和有用性

## 练习题

### 基础题（理解概念）

**练习 5.1**：解释为什么 VLM 的 RLHF 比纯文本 LLM 更具挑战性？列举至少三个独特挑战。

💡 **提示**：考虑输入模态、奖励建模、计算资源等方面。

<details>
<summary>参考答案</summary>

VLM RLHF 的独特挑战包括：
1. **多模态奖励建模**：需要同时评估文本质量和视觉一致性，奖励函数设计更复杂
2. **计算开销激增**：视觉编码器增加显存占用，图像缓存需要额外 10-20GB
3. **视觉幻觉问题**：比纯文本的事实性错误更难检测和纠正
4. **训练不稳定性**：视觉特征的高维度导致奖励信号方差更大
5. **数据标注成本**：标注者需要同时理解图像和文本，要求更高

</details>

**练习 5.2**：PPO 算法中的裁剪参数 $\epsilon$ 起什么作用？在 VLM 场景下应该如何设置？

💡 **提示**：考虑策略更新的稳定性和模型规模的关系。

<details>
<summary>参考答案</summary>

裁剪参数 $\epsilon$ 限制重要性采样比率 $r_t(\theta)$ 的范围为 $[1-\epsilon, 1+\epsilon]$，防止策略更新过大导致训练崩溃。

在 VLM 场景下的设置原则：
- 小模型（7B）：$\epsilon = 0.2$，允许较大更新
- 中等模型（13B）：$\epsilon = 0.2$，标准设置
- 大模型（34B+）：$\epsilon = 0.1$，更保守的更新

VLM 由于多模态输入的复杂性，建议使用比纯文本略小的 $\epsilon$ 值以保证稳定性。

</details>

**练习 5.3**：描述 Bradley-Terry 模型在奖励模型训练中的作用，并写出损失函数。

💡 **提示**：这是一个经典的成对比较模型。

<details>
<summary>参考答案</summary>

Bradley-Terry 模型将人类偏好建模为成对比较的概率：

$$P(y_w \succ y_l | x) = \frac{\exp(r(x, y_w))}{\exp(r(x, y_w)) + \exp(r(x, y_l))} = \sigma(r(x, y_w) - r(x, y_l))$$

损失函数：
$$\mathcal{L}_{BT} = -\mathbb{E}_{(x,y_w,y_l)} [\log \sigma(r(x, y_w) - r(x, y_l))]$$

其中 $y_w$ 是偏好的回答，$y_l$ 是不偏好的回答，$\sigma$ 是 sigmoid 函数。

该模型假设偏好概率与奖励差值的 sigmoid 成正比，自然地将偏好学习转化为二分类问题。

</details>

### 挑战题（深入思考）

**练习 5.4**：设计一个检测和量化视觉幻觉的评估框架，包括指标定义和计算方法。

💡 **提示**：考虑不同类型的幻觉（物体、属性、关系）和自动化评估的可行性。

<details>
<summary>参考答案</summary>

视觉幻觉评估框架：

1. **幻觉率指标**：
   - 物体幻觉率 = 错误提及的物体数 / 总提及物体数
   - 属性错误率 = 错误属性描述数 / 总属性描述数
   - 关系错误率 = 错误关系描述数 / 总关系描述数

2. **自动检测流程**：
```python
def evaluate_hallucination(image, generated_text):
    # 步骤1：使用目标检测模型获取 ground truth
    gt_objects = detect_objects(image)
    gt_attributes = extract_attributes(image)
    
    # 步骤2：从生成文本提取声明
    claimed_objects = extract_entities(generated_text)
    claimed_attributes = extract_attributes_from_text(generated_text)
    
    # 步骤3：计算各类幻觉
    object_hallucination = len(claimed_objects - gt_objects) / len(claimed_objects)
    attribute_errors = compute_attribute_mismatch(claimed_attributes, gt_attributes)
    
    # 步骤4：加权综合得分
    hallucination_score = 0.5 * object_hallucination + 0.3 * attribute_errors + 0.2 * relation_errors
    
    return {
        'overall_score': hallucination_score,
        'object_rate': object_hallucination,
        'attribute_rate': attribute_errors,
        'details': detailed_report
    }
```

3. **人工验证采样**：
   - 随机抽取 5% 样本人工验证
   - 计算自动评估与人工评估的一致性
   - 迭代改进检测模型

</details>

**练习 5.5**：如果在 PPO 训练过程中发现 KL 散度持续增大超过目标值 10 倍，应该如何诊断和解决？

💡 **提示**：这通常意味着策略偏离参考模型太远，需要多方面调整。

<details>
<summary>参考答案</summary>

诊断和解决步骤：

1. **立即应急措施**：
   - 停止训练，防止策略完全崩溃
   - 降低学习率至当前的 1/10
   - 增大 KL 惩罚系数 $\beta$ 至 2-5 倍

2. **根因分析**：
```python
# 检查各个组件的 KL 贡献
def diagnose_kl_explosion():
    # 分析文本和视觉部分的 KL
    text_kl = compute_kl(text_logits_new, text_logits_ref)
    vision_kl = compute_kl(vision_features_new, vision_features_ref)
    
    # 检查是否有特定 token 导致 KL 爆炸
    per_token_kl = compute_per_token_kl()
    problematic_tokens = tokens[per_token_kl > 50]
    
    # 检查奖励分布
    reward_stats = analyze_reward_distribution()
    if reward_stats['std'] > 5:
        print("奖励方差过大导致策略不稳定")
    
    return diagnostic_report
```

3. **修复策略**：
   - 回滚到最近的稳定检查点
   - 使用更保守的 PPO 裁剪参数（减小 $\epsilon$）
   - 增加参考模型的权重（混合当前策略和参考策略）
   - 检查是否有数据分布偏移

4. **预防措施**：
   - 设置 KL 散度的硬上限，超过时自动停止
   - 使用自适应 KL 控制器动态调整 $\beta$
   - 增加监控频率，及早发现异常

</details>

**练习 5.6**：设计一个多目标 RLHF 系统，同时优化准确性、安全性和多样性，如何处理目标间的冲突？

💡 **提示**：考虑 Pareto 最优和动态权重调整。

<details>
<summary>参考答案</summary>

多目标 RLHF 系统设计：

1. **目标定义与度量**：
   - 准确性：视觉一致性得分 + 事实正确性
   - 安全性：有害内容检测得分的负值
   - 多样性：生成内容的熵 + 词汇丰富度

2. **冲突处理机制**：
```python
class MultiObjectiveRLHF:
    def __init__(self):
        self.objectives = ['accuracy', 'safety', 'diversity']
        self.weights = nn.Parameter(torch.ones(3) / 3)
        
    def compute_pareto_reward(self, rewards_dict):
        # 检测 Pareto 支配关系
        is_pareto_optimal = self.check_pareto_dominance(rewards_dict)
        
        if is_pareto_optimal:
            # Pareto 最优解，使用当前权重
            return self.weighted_sum(rewards_dict, self.weights)
        else:
            # 非 Pareto 最优，惩罚主导的目标
            dominated_objective = self.find_dominated_objective(rewards_dict)
            penalty_weights = self.weights.clone()
            penalty_weights[dominated_objective] *= 2.0
            return self.weighted_sum(rewards_dict, penalty_weights)
    
    def adaptive_weight_update(self, performance_history):
        # 基于历史性能动态调整权重
        for obj in self.objectives:
            if performance_history[obj][-10:].mean() < threshold[obj]:
                # 该目标表现不佳，增加权重
                self.weights[obj] *= 1.1
        
        # 重新归一化
        self.weights = F.softmax(self.weights, dim=0)
```

3. **约束优化方法**：
   - 将安全性作为硬约束：$R_{total} = R_{accuracy} + R_{diversity}$, s.t. $R_{safety} > \tau$
   - 使用拉格朗日乘子法处理约束

4. **实践建议**：
   - 初期侧重安全性（权重 0.5），确保基础安全
   - 中期平衡三个目标（各 0.33）
   - 后期根据应用场景微调权重

</details>

**练习 5.7**：比较 RLHF 和 DPO（Direct Preference Optimization）在 VLM 上的优缺点，什么情况下选择哪种方法？

💡 **提示**：DPO 直接优化偏好，无需训练奖励模型和 PPO。

<details>
<summary>参考答案</summary>

RLHF vs DPO 比较：

**RLHF 优势**：
- 灵活性高：可以组合多个奖励信号
- 在线学习：可以持续从新反馈中学习
- 探索能力：PPO 的随机策略有助于探索

**RLHF 劣势**：
- 训练复杂：需要奖励模型 + PPO 两阶段
- 资源消耗大：需要 4 个模型同时在显存中
- 不稳定：容易出现奖励黑客、KL 爆炸

**DPO 优势**：
- 简单直接：一步优化，无需奖励模型
- 稳定性好：不存在奖励黑客问题
- 资源友好：只需要 2 个模型（策略+参考）

**DPO 劣势**：
- 离线学习：依赖预先收集的偏好数据
- 表达能力受限：难以组合复杂的奖励信号
- 探索不足：倾向于保守策略

**选择建议**：

选择 RLHF 当：
- 有充足的计算资源（8×A100 以上）
- 需要在线学习和持续改进
- 奖励函数复杂，需要组合多个信号
- 追求最佳效果

选择 DPO 当：
- 资源受限（4×A100 以下）
- 有高质量的离线偏好数据
- 追求训练稳定性和可重复性
- 快速原型和迭代

混合方案：
- 先用 DPO 快速获得基线模型
- 再用 RLHF 精细调优关键指标

</details>

## 常见陷阱与错误

### 1. 奖励模型过拟合
**症状**：训练集准确率 > 90%，验证集 < 60%

**原因**：偏好数据量太少或多样性不足

**解决方案**：
- 增加数据增强（图像变换、文本改写）
- 使用 dropout（0.1-0.2）和 L2 正则化
- 早停策略，不要训练太多 epoch

### 2. KL 散度爆炸
**症状**：KL > 50，生成文本质量急剧下降

**原因**：学习率太大或奖励信号不稳定

**解决方案**：
- 立即降低学习率
- 增大 KL 惩罚系数
- 检查奖励模型是否正常

### 3. 奖励黑客
**症状**：奖励持续上升但生成质量下降

**原因**：模型找到欺骗奖励模型的捷径

**解决方案**：
- 添加多样性奖励
- 定期更新奖励模型
- 人工审查高奖励样本

### 4. 视觉-文本不平衡
**症状**：模型忽视图像或过度依赖图像

**原因**：模态权重设置不当

**解决方案**：
- 分别监控各模态的梯度范数
- 使用梯度裁剪平衡更新
- 调整损失函数中的模态权重

### 5. 训练效率低下
**症状**：GPU 利用率 < 70%，训练速度慢

**原因**：数据加载瓶颈或批次组织不当

**解决方案**：
- 预先缓存视觉特征
- 优化批次组织（相似长度分组）
- 使用梯度累积减少通信开销

## 最佳实践检查清单

### 训练前准备
- [ ] SFT 模型已充分收敛（验证集损失稳定）
- [ ] 偏好数据质量检查（标注一致性 > 0.6）
- [ ] 奖励模型验证集准确率 > 65%
- [ ] 计算资源充足（至少 4×A100 或等效）
- [ ] 设置完整的监控指标体系
- [ ] 准备回滚机制和检查点策略

### 训练中监控
- [ ] KL 散度保持在目标范围（3-10）
- [ ] 奖励分布正常（无异常峰值）
- [ ] 生成长度合理（20-200 tokens）
- [ ] 梯度范数稳定（< 10）
- [ ] GPU 利用率 > 80%
- [ ] 定期人工评估生成质量

### 超参数配置
- [ ] 学习率：Actor < Critic（通常 1:2 到 1:5）
- [ ] PPO epochs：4（小模型）或 2（大模型）
- [ ] 裁剪参数：0.1-0.2
- [ ] KL 系数：初始 0.1-0.2，自适应调整
- [ ] 批次大小：尽可能大（受显存限制）
- [ ] 梯度累积：4-8 步（平衡效率和稳定性）

### 质量保证
- [ ] 实施奖励黑客检测机制
- [ ] 添加幻觉惩罚
- [ ] 保持训练数据多样性
- [ ] 定期更新验证集
- [ ] 对比多个检查点选择最佳
- [ ] 进行消融实验验证各组件贡献

### 部署准备
- [ ] 模型量化测试（精度损失 < 2%）
- [ ] 推理速度优化（批处理、缓存）
- [ ] 安全过滤器集成
- [ ] A/B 测试框架准备
- [ ] 回滚方案制定
- [ ] 监控和告警系统配置