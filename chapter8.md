# 第 8 章：模型部署与服务化

将训练好的 VLM 模型高效部署到生产环境是整个项目落地的关键环节。本章将系统介绍从模型优化到服务化部署的完整流程，重点关注如何在保证推理精度的前提下，最大程度提升推理速度和降低资源消耗。我们将深入探讨量化技术、推理优化、服务架构设计以及生产环境的监控与迭代策略。

## 8.1 模型量化与压缩

### 8.1.1 量化基础理论

模型量化通过降低权重和激活值的数值精度来减少模型大小和计算开销。对于 VLM 模型，量化策略需要同时考虑视觉编码器和语言模型两部分的特性。

**量化的数学表示**：

对于权重 $W \in \mathbb{R}^{m \times n}$，量化过程可表示为：

$$W_q = \text{round}\left(\frac{W - Z}{S}\right)$$

其中 $S$ 是缩放因子（scale），$Z$ 是零点（zero point），$W_q$ 是量化后的整数权重。

反量化过程：
$$W_{dq} = S \cdot W_q + Z$$

### 8.1.2 INT8 量化实践

INT8 量化是最常用的量化方案，可以将模型大小减少 75%，推理速度提升 2-4 倍。

**对称量化 vs 非对称量化**：

```
对称量化（Symmetric）:
    范围: [-127, 127]
    零点 Z = 0
    适用: 权重量化
    
非对称量化（Asymmetric）:
    范围: [0, 255]  
    零点 Z ≠ 0
    适用: 激活值量化
```

**VLM 特有的量化挑战**：

1. **视觉编码器的量化敏感性**：
   - ViT 的自注意力层对量化更敏感
   - Patch embedding 层通常保持 FP16
   - 建议: 视觉编码器使用 INT8 动态量化

2. **跨模态投影层的处理**：
   - MLP projector 是精度瓶颈
   - 建议保持 FP16 或使用更高比特量化

3. **混合精度策略**：
   ```
   模型组件量化配置:
   ├── 视觉编码器: INT8 动态量化
   ├── 投影层: FP16 保持
   ├── 语言模型
   │   ├── Embedding: INT8
   │   ├── Attention: INT8 + FP16 (QK计算)
   │   └── FFN: INT8
   └── LM Head: FP16 (关键层保护)
   ```

### 8.1.3 GPTQ 量化技术

GPTQ（Gradient-based Post-training Quantization）通过优化重构误差实现高质量的 4-bit 量化。

**GPTQ 核心算法**：

优化目标：
$$\min_{W_q} ||WX - W_qX||_2^2$$

其中 $X$ 是校准数据，通过逐层优化最小化重构误差。

**实施步骤**：

1. **准备校准数据集**（关键）：
   - 使用 100-200 个代表性样本
   - 必须包含图像-文本对
   - 覆盖不同任务类型

2. **逐层量化流程**：
   ```
   for layer in model.layers:
       # 收集该层输入激活值
       X = collect_activations(layer, calibration_data)
       
       # 计算 Hessian 矩阵
       H = 2 * X @ X.T
       
       # 逐列量化权重
       for col in range(W.shape[1]):
           w_q = quantize_column(W[:, col], H)
           # 更新剩余列以补偿量化误差
           update_remaining_columns(W, w_q, col)
   ```

3. **Group-wise 量化**：
   - 将权重分组（通常 128 个权重一组）
   - 每组独立计算 scale 和 zero point
   - 平衡压缩率和精度

### 8.1.4 AWQ 量化技术

AWQ（Activation-aware Weight Quantization）通过激活值感知的权重缩放提升量化质量。

**AWQ 核心创新**：

基于观察：权重的重要性与对应激活值的大小相关。

缩放策略：
$$W_{scaled} = W \cdot \text{diag}(s)$$
$$X_{scaled} = X \cdot \text{diag}(s^{-1})$$

其中 $s$ 是根据激活值统计计算的缩放因子。

**AWQ vs GPTQ 对比**：

| 特性 | AWQ | GPTQ |
|------|-----|------|
| 量化速度 | 快（10-20分钟） | 慢（1-2小时） |
| 推理速度 | 更快（硬件友好） | 较快 |
| 精度保持 | 优秀（4-bit） | 优秀（4-bit） |
| 显存占用 | 更低 | 较低 |
| 实现复杂度 | 中等 | 较高 |

### 8.1.5 量化方案选择指南

```
决策树：
显存充足？
├── 是 → FP16/BF16 推理
└── 否 → 需要量化
    ├── 延迟敏感？
    │   ├── 是 → INT8 量化（最快）
    │   └── 否 → 继续评估
    └── 精度要求？
        ├── 高 → GPTQ 4-bit
        └── 中 → AWQ 4-bit（推荐）
```

## 8.2 推理优化技术

### 8.2.1 KV Cache 优化

KV Cache 是 Transformer 推理的核心优化，对 VLM 尤其重要。

**内存占用计算**：

$$M_{kv} = 2 \times L \times H \times D \times (N_{text} + N_{image}) \times B \times P$$

其中：
- $L$: 层数
- $H$: 注意力头数  
- $D$: 每个头的维度
- $N_{text}$, $N_{image}$: 文本和图像 token 数
- $B$: batch size
- $P$: 精度字节数

**优化策略**：

1. **PagedAttention**（vLLM 核心）：
   ```
   传统 KV Cache:
   [连续内存块] → 浪费严重
   
   PagedAttention:
   [页表管理] → [按需分配] → [内存共享]
   优势: 减少 50-80% 内存浪费
   ```

2. **Multi-Query Attention (MQA)**：
   - 所有查询头共享一组 KV
   - 内存减少 $H$ 倍
   - 速度提升 30-50%

3. **Grouped-Query Attention (GQA)**：
   - 折中方案：$G$ 组共享 KV
   - 平衡速度和质量

### 8.2.2 Flash Attention 集成

Flash Attention 通过 IO 优化大幅提升注意力计算效率。

**核心优化**：

1. **分块计算**：
   ```python
   # 伪代码展示原理
   def flash_attention(Q, K, V, block_size=64):
       # 分块遍历，减少 HBM 访问
       for q_block in split(Q, block_size):
           for kv_block in split(K, V, block_size):
               # 在 SRAM 中计算
               attn_block = softmax(q_block @ kv_block.T)
               out_block = attn_block @ v_block
               # 增量更新结果
               update_output(out_block)
   ```

2. **VLM 特殊考虑**：
   - 图像 token 通常连续且数量固定
   - 可以预计算图像部分的注意力
   - 文本生成时只更新文本部分

**性能提升**：
- 速度: 2-4× 提升
- 显存: 线性而非二次增长
- 长序列: 支持 32K+ token

### 8.2.3 动态 Batching 优化

动态 batching 是提高吞吐量的关键技术。

**实现策略**：

1. **Continuous Batching**：
   ```
   传统 Static Batching:
   [等待所有请求完成] → GPU 利用率低
   
   Continuous Batching:
   [持续加入新请求] → [动态调度] → GPU 利用率高
   ```

2. **VLM 特有挑战**：
   - 图像预处理时间不一致
   - 图像 token 数量可变（动态分辨率）
   - 需要平衡视觉编码和文本生成

3. **优化方案**：
   ```python
   class VLMBatchScheduler:
       def schedule(self, requests):
           # 按图像大小分组
           groups = group_by_image_size(requests)
           
           # 视觉编码批处理
           for group in groups:
               vision_features = batch_encode_images(group)
               cache_features(vision_features)
           
           # 文本生成动态batching
           while active_requests:
               batch = select_compatible_requests()
               tokens = generate_batch(batch)
               update_requests(batch, tokens)
   ```

### 8.2.4 投机解码（Speculative Decoding）

使用小模型加速大模型推理。

**原理**：
1. 小模型快速生成候选 token
2. 大模型并行验证
3. 接受/拒绝候选结果

**VLM 适配**：
- 视觉编码器可以共享
- 仅语言模型部分使用投机解码
- 典型加速: 2-3×

## 8.3 服务化架构设计

### 8.3.1 整体架构

```
┌─────────────────────────────────────┐
│         Load Balancer               │
└────────────┬────────────────────────┘
             │
    ┌────────┴────────┐
    │                 │
┌───▼───┐       ┌────▼────┐
│ API   │       │  API    │
│Server │       │ Server  │
└───┬───┘       └────┬────┘
    │                │
    └────────┬───────┘
             │
    ┌────────▼────────┐
    │  Request Queue  │
    └────────┬────────┘
             │
    ┌────────▼────────────┐
    │  Inference Engine   │
    │  ┌──────────────┐  │
    │  │ Vision       │  │
    │  │ Encoder Pool │  │
    │  └──────┬───────┘  │
    │         │          │
    │  ┌──────▼───────┐  │
    │  │   Language   │  │
    │  │  Model Pool  │  │
    │  └──────────────┘  │
    └─────────────────────┘
```

### 8.3.2 关键组件设计

**1. 请求路由层**：
```python
class RequestRouter:
    def route(self, request):
        # 根据模型版本路由
        if request.model_version:
            return self.version_pools[request.model_version]
        
        # 根据负载均衡
        return self.select_least_loaded()
        
    def health_check(self):
        # 定期检查后端健康状态
        for backend in self.backends:
            if not backend.is_healthy():
                self.remove_backend(backend)
```

**2. 缓存策略**：
```python
class VLMCache:
    def __init__(self):
        # 图像特征缓存
        self.vision_cache = LRUCache(size=10000)
        # Prompt 缓存
        self.prompt_cache = LRUCache(size=5000)
        
    def get_vision_features(self, image_hash):
        if image_hash in self.vision_cache:
            return self.vision_cache[image_hash]
        return None
        
    def cache_vision_features(self, image_hash, features):
        self.vision_cache[image_hash] = features
```

**3. 资源管理**：
```python
class ResourceManager:
    def allocate_request(self, request):
        required_memory = self.estimate_memory(request)
        
        # 等待资源可用
        while not self.has_available_memory(required_memory):
            time.sleep(0.1)
            
        # 分配资源
        self.current_memory += required_memory
        return self.process_request(request)
```

### 8.3.3 高可用设计

**1. 模型热更新**：
```python
class ModelManager:
    def update_model(self, new_model_path):
        # 加载新模型
        new_model = load_model(new_model_path)
        
        # 逐步切换流量
        for ratio in [0.1, 0.3, 0.5, 0.7, 1.0]:
            self.traffic_ratio = ratio
            time.sleep(60)  # 观察指标
            
            if self.has_errors():
                self.rollback()
                break
```

**2. 故障恢复**：
- 请求重试机制
- 降级策略（使用更小模型）
- 熔断保护

### 8.3.4 API 设计

**RESTful API 示例**：
```python
@app.post("/v1/chat/completions")
async def chat_completion(request: ChatRequest):
    # 请求验证
    validate_request(request)
    
    # 图像预处理
    if request.images:
        vision_features = await encode_images(request.images)
    
    # 生成响应
    response = await generate_response(
        prompt=request.messages,
        vision_features=vision_features,
        **request.parameters
    )
    
    return response
```

**流式响应**：
```python
@app.post("/v1/chat/completions/stream")
async def stream_chat_completion(request: ChatRequest):
    async def generate():
        async for token in generate_tokens(request):
            yield f"data: {json.dumps({'token': token})}\n\n"
    
    return StreamingResponse(generate(), media_type="text/event-stream")
```

## 8.4 监控与迭代优化

### 8.4.1 关键指标监控

**性能指标**：

1. **延迟指标**：
   - TTFT (Time To First Token): 首个 token 延迟
   - TPS (Tokens Per Second): 生成速度
   - E2E Latency: 端到端延迟

2. **吞吐量指标**：
   - QPS (Queries Per Second)
   - GPU 利用率
   - 内存使用率

3. **质量指标**：
   - 生成质量评分
   - 错误率
   - 用户满意度

**监控实现**：
```python
class MetricsCollector:
    def __init__(self):
        self.metrics = {
            'ttft': [],
            'tps': [],
            'gpu_util': [],
            'memory_usage': []
        }
    
    def record_inference(self, request_id, start_time, tokens):
        ttft = time.time() - start_time
        tps = len(tokens) / (time.time() - start_time)
        
        self.metrics['ttft'].append(ttft)
        self.metrics['tps'].append(tps)
        
        # 记录到 Prometheus
        TTFT_HISTOGRAM.observe(ttft)
        TPS_GAUGE.set(tps)
```

### 8.4.2 性能分析工具

**1. GPU 性能分析**：
```bash
# 使用 nsys 进行性能分析
nsys profile -o model_profile python inference_server.py

# 使用 nvprof 分析 kernel 执行
nvprof --print-gpu-trace python benchmark.py
```

**2. 内存分析**：
```python
def analyze_memory():
    # 显存快照
    snapshot = torch.cuda.memory_snapshot()
    
    # 分析内存分配
    for block in snapshot:
        if block['allocated']:
            print(f"Size: {block['size']}, Stream: {block['stream']}")
    
    # 内存统计
    print(f"Allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
    print(f"Reserved: {torch.cuda.memory_reserved() / 1e9:.2f} GB")
```

### 8.4.3 A/B 测试框架

```python
class ABTestManager:
    def __init__(self):
        self.experiments = {}
        
    def create_experiment(self, name, variants):
        self.experiments[name] = {
            'variants': variants,
            'metrics': defaultdict(list)
        }
    
    def route_request(self, request, experiment_name):
        # 基于用户 ID 的一致性哈希
        user_hash = hash(request.user_id)
        variant_index = user_hash % len(self.experiments[experiment_name]['variants'])
        return self.experiments[experiment_name]['variants'][variant_index]
    
    def record_metric(self, experiment_name, variant, metric_name, value):
        self.experiments[experiment_name]['metrics'][f"{variant}_{metric_name}"].append(value)
```

### 8.4.4 自动优化策略

**1. 动态批大小调整**：
```python
class DynamicBatchSizer:
    def __init__(self):
        self.current_batch_size = 1
        self.latency_history = []
        
    def adjust_batch_size(self):
        avg_latency = np.mean(self.latency_history[-100:])
        
        if avg_latency < TARGET_LATENCY * 0.8:
            # 延迟充裕，增加批大小
            self.current_batch_size = min(self.current_batch_size + 1, MAX_BATCH)
        elif avg_latency > TARGET_LATENCY:
            # 延迟超标，减小批大小
            self.current_batch_size = max(self.current_batch_size - 1, 1)
```

**2. 模型副本自动扩缩容**：
```python
class AutoScaler:
    def scale_decision(self, metrics):
        # 基于队列长度和延迟决策
        if metrics['queue_length'] > QUEUE_THRESHOLD:
            return 'scale_up'
        elif metrics['avg_gpu_util'] < 0.3:
            return 'scale_down'
        return 'maintain'
```

## Case Study: vLLM 部署 VLM 的最佳实践

### 背景介绍

vLLM 是目前最流行的 LLM 推理框架之一，通过 PagedAttention 等创新显著提升了推理效率。本案例将详细介绍如何使用 vLLM 部署 LLaVA-NeXT 模型。

### 环境准备

```bash
# 安装 vLLM (支持 VLM)
pip install vllm>=0.3.0

# 验证 GPU 支持
python -c "import torch; print(torch.cuda.get_device_capability())"
# 需要 compute capability >= 7.0
```

### 模型部署配置

```python
from vllm import LLM, SamplingParams
from vllm.multimodal import MultiModalData

class VLMDeployment:
    def __init__(self, model_path):
        self.llm = LLM(
            model=model_path,
            # 关键参数配置
            tensor_parallel_size=2,  # TP 并行度
            max_model_len=4096,      # 最大序列长度
            gpu_memory_utilization=0.9,  # GPU 内存利用率
            
            # VLM 特定配置
            image_input_type="pixel_values",
            image_token_id=32000,
            image_input_shape=(3, 336, 336),
            image_feature_size=576,  # 24*24 patches
            
            # 优化参数
            enable_prefix_caching=True,  # 启用前缀缓存
            enable_chunked_prefill=True,  # 分块预填充
            max_num_batched_tokens=8192,
            max_num_seqs=256,
            
            # 量化配置（可选）
            quantization="awq",  # 使用 AWQ 4-bit 量化
        )
        
        self.sampling_params = SamplingParams(
            temperature=0.7,
            top_p=0.9,
            max_tokens=512,
        )
```

### 推理优化配置

```python
# 1. 启用 Flash Attention
os.environ["VLLM_USE_FLASH_ATTN"] = "1"

# 2. 配置 CUDA Graph
os.environ["VLLM_USE_CUDA_GRAPH"] = "1"
os.environ["VLLM_CUDA_GRAPH_MAX_SEQS"] = "32"

# 3. 调整调度策略
engine_args = {
    "scheduler_config": {
        "max_num_batched_tokens": 8192,
        "max_num_seqs": 256,
        "max_paddings": 512,
        "delay_factor": 0.1,  # 控制批处理等待时间
    }
}
```

### 性能调优实战

**1. 批处理优化**：
```python
def optimized_batch_inference(requests):
    # 按图像大小分组
    grouped = defaultdict(list)
    for req in requests:
        img_size = req.image.shape
        grouped[img_size].append(req)
    
    results = []
    for size, batch in grouped.items():
        # 同尺寸图像批处理
        outputs = llm.generate(
            prompts=[r.prompt for r in batch],
            multi_modal_data=[r.image for r in batch],
            sampling_params=sampling_params
        )
        results.extend(outputs)
    
    return results
```

**2. 内存优化**：
```python
# 监控内存使用
def monitor_memory():
    stats = llm.get_model_memory_usage()
    print(f"KV Cache: {stats['kv_cache_usage'] / 1e9:.2f} GB")
    print(f"Model Weights: {stats['model_weights'] / 1e9:.2f} GB")
    
    # 动态调整 KV cache 大小
    if stats['kv_cache_usage'] > MEMORY_THRESHOLD:
        llm.reduce_max_num_seqs(factor=0.8)
```

### 生产部署检查清单

- [x] 配置健康检查端点
- [x] 实现优雅关闭机制
- [x] 设置请求超时
- [x] 配置日志和监控
- [x] 实现降级策略
- [x] 准备回滚方案

### 性能基准测试结果

| 配置 | TTFT (ms) | TPS | QPS | GPU 利用率 |
|-----|-----------|-----|-----|-----------|
| 基础配置 | 450 | 42 | 8 | 65% |
| + PagedAttention | 380 | 48 | 12 | 75% |
| + Flash Attention | 320 | 56 | 15 | 82% |
| + AWQ 量化 | 280 | 68 | 20 | 88% |
| + Dynamic Batching | 250 | 72 | 28 | 92% |

## 高级话题

### AWQ vs GPTQ 深度对比

**量化精度对比实验**：

测试模型：LLaVA-NeXT-13B
测试数据集：COCO Captions Validation

| 量化方法 | Perplexity | BLEU-4 | 推理速度 | 显存占用 |
|---------|------------|--------|---------|---------|
| FP16 (基准) | 8.32 | 35.2 | 1.0x | 26GB |
| INT8 | 8.45 | 34.8 | 2.1x | 13GB |
| GPTQ 4-bit | 8.68 | 34.1 | 3.2x | 8.5GB |
| AWQ 4-bit | 8.59 | 34.4 | 3.8x | 8.2GB |

**关键发现**：

1. **AWQ 在推理速度上优势明显**：
   - 原因：权重布局更适合硬件加速
   - kernel 实现更高效

2. **GPTQ 在某些任务上精度略高**：
   - 特别是需要精确数值计算的任务
   - 但差异通常 < 1%

3. **混合策略**：
   ```python
   # 对不同层使用不同量化
   quantization_config = {
       "vision_encoder": "int8",      # 视觉编码器用 INT8
       "projection": None,             # 投影层不量化
       "llm_layers_0_15": "awq_4bit", # 前半部分用 AWQ
       "llm_layers_16_31": "gptq_4bit", # 后半部分用 GPTQ
       "lm_head": None                # 输出层不量化
   }
   ```

### 动态 Batching 高级优化

**1. 请求优先级调度**：
```python
class PriorityBatchScheduler:
    def __init__(self):
        self.queues = {
            'high': PriorityQueue(),
            'normal': Queue(),
            'low': Queue()
        }
    
    def schedule_next_batch(self, max_batch_size):
        batch = []
        
        # 优先处理高优先级请求
        for priority in ['high', 'normal', 'low']:
            while len(batch) < max_batch_size and not self.queues[priority].empty():
                batch.append(self.queues[priority].get())
        
        return batch
```

**2. 自适应 Padding 策略**：
```python
def adaptive_padding(sequences):
    lengths = [len(seq) for seq in sequences]
    
    # 计算最优 padding 长度
    # 考虑硬件特性（如 tensor core 需要 8 的倍数）
    max_len = max(lengths)
    optimal_len = ((max_len + 7) // 8) * 8
    
    # 如果浪费超过阈值，考虑分批
    waste_ratio = (optimal_len * len(sequences) - sum(lengths)) / (optimal_len * len(sequences))
    
    if waste_ratio > 0.3:  # 30% 浪费阈值
        # 分成两批处理
        return split_by_length(sequences)
    
    return pad_sequences(sequences, optimal_len)
```

**3. 预测性批处理**：
```python
class PredictiveBatcher:
    def __init__(self):
        self.arrival_predictor = ArrivalRatePredictor()
        
    def should_wait_for_batch(self, current_batch_size):
        # 预测未来请求到达
        expected_arrivals = self.arrival_predictor.predict(window=100)  # 100ms
        
        # 计算等待收益
        current_efficiency = batch_efficiency(current_batch_size)
        future_efficiency = batch_efficiency(current_batch_size + expected_arrivals)
        
        # 决策：等待 vs 立即处理
        if future_efficiency / current_efficiency > 1.2:  # 20% 提升阈值
            return True, 100  # 等待 100ms
        return False, 0
```

## 本章小结

本章系统介绍了 VLM 模型从优化到部署的完整流程。我们深入探讨了以下关键技术：

### 核心要点回顾

1. **模型量化技术**：
   - INT8 量化可实现 2-4× 加速，适合延迟敏感场景
   - GPTQ 和 AWQ 4-bit 量化可减少 75% 显存，精度损失 < 2%
   - 混合精度策略：视觉编码器 INT8，投影层 FP16，语言模型 4-bit

2. **推理优化**：
   - PagedAttention 减少 50-80% KV cache 浪费
   - Flash Attention 实现 2-4× 速度提升
   - 动态 batching 提升 GPU 利用率至 90%+

3. **服务化架构**：
   - 分离视觉编码和文本生成，独立扩展
   - 实施多级缓存策略（图像特征、prompt）
   - 支持流式响应和批处理 API

4. **监控与优化**：
   - 关注 TTFT、TPS、QPS 三大核心指标
   - 实施 A/B 测试验证优化效果
   - 自动调整批大小和模型副本数

### 关键公式汇总

**量化误差**：
$$\epsilon = ||W - W_q||_F \approx \frac{\sigma_W \cdot n}{\sqrt{12} \cdot 2^b}$$

**KV Cache 内存**：
$$M_{kv} = 2LHD(N_{text} + N_{image})BP$$

**批处理效率**：
$$\eta = \frac{\sum_{i=1}^B l_i}{B \cdot \max(l_i)}$$

**推理延迟模型**：
$$T_{total} = T_{encode} + N_{tokens} \cdot T_{decode} + T_{overhead}$$

## 练习题

### 基础题

**练习 8.1**: 计算 KV Cache 内存需求

一个 13B 参数的 VLM 模型，40 层，40 个注意力头，每头维度 128，处理批大小为 8，每个样本包含 576 个图像 token 和平均 512 个文本 token。使用 FP16 精度，计算 KV cache 的内存需求。

<details>
<summary>💡 提示</summary>

使用 KV cache 内存公式，注意单位转换（GB）。

</details>

<details>
<summary>📝 参考答案</summary>

$$M_{kv} = 2 \times 40 \times 40 \times 128 \times (512 + 576) \times 8 \times 2$$
$$= 2 \times 40 \times 40 \times 128 \times 1088 \times 8 \times 2$$
$$= 3,565,158,400 \text{ bytes} \approx 3.32 \text{ GB}$$

这解释了为什么 KV cache 优化如此重要。

</details>

**练习 8.2**: AWQ 量化压缩率计算

将一个 FP16 的 7B 模型量化为 AWQ 4-bit，假设模型权重占 14GB，计算：
1. 量化后的模型大小
2. 理论压缩率
3. 考虑额外的 scale/zero point 开销（group size = 128），实际模型大小

<details>
<summary>💡 提示</summary>

4-bit 量化理论上压缩 4 倍，但需要存储额外的量化参数。

</details>

<details>
<summary>📝 参考答案</summary>

1. 理论量化后大小：14GB ÷ 4 = 3.5GB

2. 理论压缩率：16 bits / 4 bits = 4×

3. 实际大小计算：
   - 每 128 个权重需要额外 32 bits (FP16 scale + zero)
   - 开销率：32 / (128 × 4) = 6.25%
   - 实际大小：3.5GB × 1.0625 ≈ 3.72GB
   - 实际压缩率：14GB / 3.72GB ≈ 3.76×

</details>

**练习 8.3**: Flash Attention 内存节省

传统注意力计算需要存储 N×N 的注意力矩阵，Flash Attention 通过分块计算避免这一开销。对于序列长度 4096，批大小 8，注意力头数 32，计算两种方法的峰值内存差异。

<details>
<summary>💡 提示</summary>

传统方法需要存储完整注意力矩阵，Flash Attention 只需存储块大小的矩阵。

</details>

<details>
<summary>📝 参考答案</summary>

传统注意力：
- 注意力矩阵：8 × 32 × 4096 × 4096 × 2 bytes (FP16)
- = 8,589,934,592 bytes ≈ 8 GB

Flash Attention（块大小 64）：
- 块矩阵：8 × 32 × 64 × 64 × 2 bytes
- = 2,097,152 bytes ≈ 2 MB

内存节省：8 GB → 2 MB，减少 4000 倍！

</details>

### 挑战题

**练习 8.4**: 动态 Batching 调度算法设计

设计一个动态 batching 调度器，需要考虑：
- 不同请求的优先级（P0/P1/P2）
- 图像大小差异（224×224, 336×336, 448×448）
- 最大批大小限制（32）
- 延迟 SLA 要求（P0 < 100ms, P1 < 500ms, P2 < 2000ms）

请给出调度策略的伪代码。

<details>
<summary>💡 提示</summary>

考虑多队列设计，按优先级和图像大小分组，实施抢占机制。

</details>

<details>
<summary>📝 参考答案</summary>

```python
class AdaptiveBatchScheduler:
    def __init__(self):
        # 多维度队列
        self.queues = {
            (priority, img_size): Queue()
            for priority in ['P0', 'P1', 'P2']
            for img_size in [224, 336, 448]
        }
        self.sla_timers = {}
        
    def schedule_next_batch(self):
        batch = []
        selected_size = None
        
        # 步骤1：检查 P0 紧急请求
        for size in [224, 336, 448]:
            queue = self.queues[('P0', size)]
            urgent = self.check_sla_violation(queue, 80)  # 80ms 警戒线
            if urgent:
                return self.create_batch(urgent, size)
        
        # 步骤2：贪心选择最优批次
        best_score = -1
        best_config = None
        
        for (priority, size), queue in self.queues.items():
            if queue.empty():
                continue
                
            # 计算得分：队列长度 × 优先级权重 / 等待时间
            score = len(queue) * self.priority_weight[priority]
            score /= (1 + self.avg_wait_time(queue))
            
            if score > best_score:
                best_score = score
                best_config = (priority, size)
                selected_size = size
        
        # 步骤3：构建批次
        if best_config:
            # 同尺寸图像打包
            primary_queue = self.queues[best_config]
            while len(batch) < 32 and not primary_queue.empty():
                batch.append(primary_queue.get())
            
            # 填充相同尺寸的低优先级请求
            for priority in ['P0', 'P1', 'P2']:
                if (priority, selected_size) != best_config:
                    queue = self.queues[(priority, selected_size)]
                    while len(batch) < 32 and not queue.empty():
                        batch.append(queue.get())
        
        return batch
        
    def check_sla_violation(self, queue, threshold_ms):
        """检查是否有接近 SLA 违约的请求"""
        urgent = []
        for req in queue:
            if time.time() - req.arrival_time > threshold_ms / 1000:
                urgent.append(req)
        return urgent
```

关键设计点：
1. 多维度队列避免头部阻塞
2. SLA 感知的抢占调度
3. 同尺寸图像批处理提升效率
4. 动态权重平衡吞吐量和延迟

</details>

**练习 8.5**: 量化策略选择

你需要部署一个 34B 参数的 VLM 模型到配备 2×A100 (40GB) 的服务器。模型 FP16 权重占 68GB，预期 QPS 为 50，平均序列长度 2048。请设计完整的量化和优化方案。

<details>
<summary>💡 提示</summary>

需要综合考虑显存限制、推理速度要求和精度保持。

</details>

<details>
<summary>📝 参考答案</summary>

**分析**：
- 总显存：80GB
- 模型权重：68GB (FP16)
- KV Cache：约 8-10GB (批大小 16)
- 激活值：约 4-6GB

**方案设计**：

1. **混合量化策略**：
```python
config = {
    # 关键层保持高精度
    "vision_encoder": "int8",        # 14GB → 7GB
    "projection_layer": "fp16",      # 0.5GB (不变)
    "llm.layers[0:8]": "fp16",      # 13.5GB (不变)
    "llm.layers[8:32]": "awq_4bit", # 40.5GB → 10GB  
    "lm_head": "fp16"               # 0.5GB (不变)
}
# 总计：7 + 0.5 + 13.5 + 10 + 0.5 = 31.5GB
```

2. **推理优化**：
- 启用 PagedAttention：KV cache 10GB → 6GB
- 使用 Flash Attention 2
- Continuous batching，维持批大小 12-20

3. **部署配置**：
```python
deployment = {
    "tensor_parallel": 2,
    "max_batch_size": 20,
    "max_seq_length": 2048,
    "gpu_memory_fraction": 0.95,
    "enable_cuda_graph": True
}
```

4. **预期性能**：
- 显存使用：31.5GB (模型) + 6GB (KV) + 4GB (激活) = 41.5GB / 80GB
- TTFT：< 200ms
- TPS：60-80 tokens/s
- 支持 QPS：50-60

5. **降级方案**：
- 高负载时：批大小降至 8，全模型 4-bit
- 紧急情况：切换至 13B 备用模型

</details>

**练习 8.6**: 推理服务故障诊断

你的 VLM 推理服务出现以下症状：
- GPU 利用率只有 40%
- P99 延迟是 P50 的 10 倍
- 每小时有 2-3 次 OOM 错误
- 用户报告偶尔生成内容不完整

请分析可能的原因并给出解决方案。

<details>
<summary>💡 提示</summary>

从资源利用、调度策略、内存管理等多个角度分析。

</details>

<details>
<summary>📝 参考答案</summary>

**问题分析**：

1. **GPU 利用率低 (40%)**：
   - 原因：IO 瓶颈或批处理不足
   - 诊断：检查数据加载时间、批大小分布

2. **P99 延迟异常**：
   - 原因：长尾请求或资源竞争
   - 诊断：分析请求长度分布、检查是否有巨型请求

3. **间歇性 OOM**：
   - 原因：内存泄漏或突发大请求
   - 诊断：监控内存增长曲线、检查特定输入模式

4. **生成不完整**：
   - 原因：超时截断或 OOM 静默失败
   - 诊断：检查超时配置、错误处理逻辑

**解决方案**：

```python
# 1. 优化批处理策略
class ImprovedScheduler:
    def __init__(self):
        self.max_tokens_per_batch = 8192  # 总 token 限制
        self.max_seq_length = 2048        # 单请求限制
        
    def create_batch(self, requests):
        # 按长度排序，避免 padding 浪费
        requests.sort(key=lambda x: len(x.tokens))
        
        batch = []
        total_tokens = 0
        
        for req in requests:
            if len(req.tokens) > self.max_seq_length:
                # 拒绝超长请求
                req.reject("Sequence too long")
                continue
                
            req_tokens = len(req.tokens) * len(batch + [req])
            if total_tokens + req_tokens <= self.max_tokens_per_batch:
                batch.append(req)
                total_tokens += req_tokens
            else:
                break
                
        return batch

# 2. 内存保护机制
class MemoryGuard:
    def __init__(self):
        self.memory_threshold = 0.85
        
    def check_memory(self):
        usage = torch.cuda.memory_reserved() / torch.cuda.max_memory_allocated()
        if usage > self.memory_threshold:
            # 触发内存清理
            torch.cuda.empty_cache()
            # 降级策略
            self.reduce_batch_size()
            
    def estimate_request_memory(self, request):
        # 预估内存需求
        kv_cache = 2 * layers * heads * dim * len(request.tokens)
        activation = len(request.tokens) * hidden_size * 4
        return kv_cache + activation

# 3. 请求预处理和验证
def validate_request(request):
    # 检查图像大小
    if request.image.size > MAX_IMAGE_SIZE:
        return resize_image(request.image)
    
    # 检查 token 长度
    if len(request.tokens) > MAX_TOKENS:
        request.tokens = request.tokens[:MAX_TOKENS]
        request.add_warning("Truncated to max length")
    
    return request

# 4. 监控和告警
@app.middleware("http")
async def monitor_middleware(request, call_next):
    start_time = time.time()
    
    # 记录请求前状态
    gpu_util_before = get_gpu_utilization()
    memory_before = torch.cuda.memory_allocated()
    
    response = await call_next(request)
    
    # 计算指标
    latency = time.time() - start_time
    memory_delta = torch.cuda.memory_allocated() - memory_before
    
    # 异常检测
    if latency > LATENCY_THRESHOLD:
        logger.warning(f"High latency: {latency}s")
        
    if memory_delta > MEMORY_SPIKE_THRESHOLD:
        logger.warning(f"Memory spike: {memory_delta / 1e9}GB")
    
    return response
```

**具体措施**：
1. 实施请求大小限制和预验证
2. 动态调整批大小基于内存使用
3. 分离长短请求到不同处理队列  
4. 添加详细监控和自动降级机制
5. 实施优雅的错误处理和重试

</details>

## 常见陷阱与错误 (Gotchas)

### 1. 量化相关陷阱

**陷阱：盲目追求低比特量化**
```python
# ❌ 错误：所有层都用 2-bit
model = quantize_model(model, bits=2)  # 精度严重下降

# ✅ 正确：混合精度策略
critical_layers = identify_sensitive_layers(model)
for name, layer in model.named_modules():
    if name in critical_layers:
        quantize_layer(layer, bits=8)  # 关键层保持高精度
    else:
        quantize_layer(layer, bits=4)
```

**陷阱：忽视校准数据质量**
```python
# ❌ 错误：使用随机数据校准
calibration_data = torch.randn(100, 3, 224, 224)

# ✅ 正确：使用真实分布的数据
calibration_data = load_representative_samples(
    dataset, 
    n_samples=200,
    stratified=True  # 确保覆盖各种情况
)
```

### 2. 推理优化陷阱

**陷阱：过度优化单一指标**
```python
# ❌ 错误：只优化吞吐量，忽视延迟
config = {"max_batch_size": 128}  # P99 延迟爆炸

# ✅ 正确：平衡多个指标
config = {
    "max_batch_size": 32,
    "max_wait_time": 50,  # ms
    "target_latency": 200  # ms
}
```

**陷阱：KV Cache 内存泄漏**
```python
# ❌ 错误：不清理已完成请求的 cache
kv_cache[request_id] = compute_kv(request)
# 请求完成后未删除...

# ✅ 正确：及时清理
try:
    kv_cache[request_id] = compute_kv(request)
    result = generate(kv_cache[request_id])
finally:
    del kv_cache[request_id]  # 确保清理
```

### 3. 服务化陷阱

**陷阱：忽视冷启动问题**
```python
# ❌ 错误：直接处理第一个请求
@app.on_event("startup")
async def startup():
    global model
    model = load_model()  # 加载完就结束

# ✅ 正确：预热模型
@app.on_event("startup") 
async def startup():
    global model
    model = load_model()
    # 预热：运行几个推理避免首次调用慢
    warmup_inputs = create_dummy_inputs()
    for _ in range(3):
        model.generate(warmup_inputs)
```

**陷阱：同步阻塞操作**
```python
# ❌ 错误：同步图像处理阻塞事件循环
def process_request(image, text):
    processed_image = cv2.resize(image, (336, 336))  # 阻塞
    return model.generate(processed_image, text)

# ✅ 正确：异步处理
async def process_request(image, text):
    processed_image = await asyncio.to_thread(
        cv2.resize, image, (336, 336)
    )
    return await model.generate_async(processed_image, text)
```

### 4. 监控盲区

**陷阱：只监控平均值**
```python
# ❌ 错误：平均延迟看起来很好
print(f"Avg latency: {np.mean(latencies)}ms")  # 200ms

# ✅ 正确：关注分位数
print(f"P50: {np.percentile(latencies, 50)}ms")  # 150ms
print(f"P95: {np.percentile(latencies, 95)}ms")  # 800ms！
print(f"P99: {np.percentile(latencies, 99)}ms")  # 2000ms！！
```

## 最佳实践检查清单

### 部署前检查

**模型优化**
- [ ] 选择合适的量化方案（INT8/4-bit）
- [ ] 验证量化后精度损失 < 阈值
- [ ] 关键层保持高精度
- [ ] 使用代表性数据校准

**推理配置**
- [ ] 启用 Flash Attention
- [ ] 配置 PagedAttention
- [ ] 设置合理的批大小上限
- [ ] 实施动态 batching

**服务架构**
- [ ] 实现健康检查接口
- [ ] 配置负载均衡
- [ ] 设置请求超时
- [ ] 实现优雅关闭

### 部署中监控

**性能指标**
- [ ] TTFT < 目标值
- [ ] TPS 满足需求
- [ ] GPU 利用率 > 80%
- [ ] 内存使用稳定

**质量指标**
- [ ] 错误率 < 0.1%
- [ ] 生成质量评分达标
- [ ] 无内容截断问题

**稳定性**
- [ ] 无内存泄漏
- [ ] P99 延迟稳定
- [ ] 自动故障恢复工作

### 持续优化

**A/B 测试**
- [ ] 新优化先小流量测试
- [ ] 收集足够样本量
- [ ] 多维度指标评估
- [ ] 有回滚预案

**迭代改进**
- [ ] 定期 review 慢查询
- [ ] 分析错误日志模式  
- [ ] 收集用户反馈
- [ ] 跟踪新技术进展

**容量规划**
- [ ] 预测流量增长
- [ ] 制定扩容计划
- [ ] 优化资源利用率
- [ ] 成本效益分析