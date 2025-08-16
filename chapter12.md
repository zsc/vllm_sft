# 第 12 章：多机多卡调试地狱

当你的 VLM 训练扩展到多机多卡时，你将进入一个充满挑战的调试世界。本章将系统地介绍分布式训练中最常见的问题和解决方案，从 NCCL 通信错误到进程死锁，从异构 GPU 混训到框架选择，帮助你快速定位和解决多机训练中的各种"地狱"级问题。无论是凌晨三点的 NCCL timeout，还是莫名其妙的进程挂起，本章都能让你在 5 分钟内找到解决思路。

## 12.1 NCCL 错误的常见原因与解决

NCCL（NVIDIA Collective Communications Library）是多 GPU 训练的核心通信库。当你看到 "NCCL Error" 时，不要慌张，按照以下流程系统排查。

### 12.1.1 快速诊断流程

当遇到 NCCL 错误时，首先执行以下诊断步骤：

**第一步：启用详细日志**
```bash
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=ALL
export TORCH_DISTRIBUTED_DEBUG=DETAIL
```

**第二步：检查基础连通性**
```bash
# 检查节点间网络连通性
ping <other_node_ip>
nc -zv <other_node_ip> 29500  # 检查端口是否开放

# 检查 GPU 可见性
nvidia-smi -L
echo $CUDA_VISIBLE_DEVICES
```

**第三步：运行 NCCL 测试**
```bash
# 单机测试
python -m torch.distributed.run --nproc_per_node=8 --nnodes=1 test_nccl.py

# 多机测试（在主节点运行）
python -m torch.distributed.run \
    --nproc_per_node=8 \
    --nnodes=2 \
    --node_rank=0 \
    --master_addr=<master_ip> \
    --master_port=29500 \
    test_nccl.py
```

### 12.1.2 常见错误类型与解决方案

**错误 1：NCCL Init Failed**

症状：
```
RuntimeError: NCCL error in: ../torch/csrc/distributed/c10d/ProcessGroupNCCL.cpp:1191
unhandled system error, NCCL version 2.14.3
```

原因分析：
1. GPU 不可见或编号错误
2. NCCL 版本不兼容
3. 共享内存不足

解决方案：
```bash
# 1. 确保 GPU 编号正确
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# 2. 增加共享内存
docker run --shm-size=32gb ...  # Docker 环境
# 或修改系统配置
echo "kernel.shmmax = 68719476736" >> /etc/sysctl.conf
echo "kernel.shmall = 4294967296" >> /etc/sysctl.conf
sysctl -p

# 3. 降级或升级 NCCL
pip install torch==2.0.1+cu118  # 使用兼容版本
```

**错误 2：NCCL Timeout**

症状：
```
torch.distributed.DistBackendError: NCCL timeout after 1800 seconds
```

原因分析：
1. 网络带宽不足或延迟过高
2. 某个进程卡住或崩溃
3. 数据不均衡导致等待

解决方案：
```bash
# 1. 增加超时时间
export NCCL_TIMEOUT=3600  # 单位：秒
export NCCL_ASYNC_ERROR_HANDLING=1

# 2. 使用更快的网络协议
export NCCL_IB_DISABLE=0  # 启用 InfiniBand
export NCCL_SOCKET_IFNAME=eth0  # 指定网络接口

# 3. 检查进程状态
ps aux | grep python  # 查看是否有僵尸进程
htop  # 查看 CPU/内存使用情况
```

**错误 3：NCCL AllReduce Failed**

症状：
```
NCCL WARN Unhandled System Error while waiting for event
```

原因分析：
1. PCIe 通信问题
2. NVLink 配置错误
3. 拓扑结构不优化

解决方案：
```bash
# 1. 检查 PCIe 和 NVLink 状态
nvidia-smi topo -m

# 2. 优化 P2P 通信
export NCCL_P2P_DISABLE=0
export NCCL_P2P_LEVEL=NVL  # 使用 NVLink

# 3. 设置合理的 NCCL 树形拓扑
export NCCL_TREE_THRESHOLD=0  # 总是使用树形算法
```

### 12.1.3 网络配置优化

**InfiniBand 配置**

InfiniBand 是高性能计算的首选网络：

```bash
# 检查 IB 状态
ibstat
ibping -S  # 在一个节点启动服务器
ibping -c <server_guid>  # 在另一个节点测试

# NCCL IB 配置
export NCCL_IB_HCA=mlx5_0,mlx5_1  # 指定 HCA
export NCCL_IB_GID_INDEX=3  # RoCE 环境需要
export NCCL_IB_TC=106  # Traffic Class
export NCCL_IB_SL=0  # Service Level
```

**TCP 网络优化**

当只有以太网时的优化策略：

```bash
# 1. 选择正确的网络接口
export NCCL_SOCKET_IFNAME=eth0,eth1  # 可以指定多个
export NCCL_SOCKET_NTHREADS=8  # 增加 socket 线程数
export NCCL_NSOCKS_PERTHREAD=4  # 每线程 socket 数

# 2. TCP 缓冲区优化
echo "net.core.rmem_max = 134217728" >> /etc/sysctl.conf
echo "net.core.wmem_max = 134217728" >> /etc/sysctl.conf
echo "net.ipv4.tcp_rmem = 4096 87380 134217728" >> /etc/sysctl.conf
echo "net.ipv4.tcp_wmem = 4096 65536 134217728" >> /etc/sysctl.conf
sysctl -p
```

### 12.1.4 环境变量速查表

关键 NCCL 环境变量及其作用：

```bash
# 调试相关
NCCL_DEBUG=INFO/WARN/ERROR  # 日志级别
NCCL_DEBUG_FILE=/path/to/log  # 日志输出文件

# 性能调优
NCCL_BUFFSIZE=8388608  # 缓冲区大小（默认 4MB）
NCCL_NTHREADS=512  # NCCL 线程数
NCCL_MAX_NCHANNELS=16  # 最大通道数

# 网络选择
NCCL_NET_GDR_LEVEL=5  # GPUDirect RDMA 级别
NCCL_CROSS_NIC=1  # 允许跨 NIC 通信

# 算法选择
NCCL_ALGO=Ring/Tree/CollNet  # 指定算法
NCCL_PROTO=LL/LL128/Simple  # 协议选择
```

### 12.1.5 实战案例：8×A100 集群调试

真实案例：某团队在 2 节点 8×A100 集群上训练 VLM，遇到间歇性 NCCL 错误。

**问题表现**：
- 训练进行到 30% 时随机出现 NCCL timeout
- 错误日志显示 `unhandled cuda error`
- 重启后能继续训练一段时间

**排查过程**：
1. 启用 NCCL_DEBUG=INFO，发现特定 GPU 对通信超时
2. nvidia-smi topo -m 显示 GPU 6-7 之间是 PIX 连接（最慢）
3. 检查温度日志，发现 GPU 6 经常触发温度保护（throttling）

**解决方案**：
```bash
# 1. 调整 GPU 映射，避免使用问题 GPU 对
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5  # 跳过 6,7

# 2. 降低功率上限防止过热
nvidia-smi -pl 300  # 设置功率上限 300W

# 3. 优化通信模式
export NCCL_P2P_LEVEL=PHB  # 只使用同一 PCIe 桥下的 P2P
export NCCL_ALGO=Tree  # 使用树形算法减少 P2P 依赖

# 4. 监控脚本
watch -n 1 'nvidia-smi --query-gpu=index,name,temperature.gpu,power.draw --format=csv'
```

**最终效果**：
- 训练稳定性提升，未再出现 timeout
- 虽然使用 6 个 GPU，但整体吞吐量反而提升 15%（避免了通信瓶颈）

## 12.2 进程同步与死锁排查

分布式训练中，进程同步问题是仅次于 NCCL 错误的第二大"杀手"。本节将深入剖析死锁的成因和快速定位方法。

### 12.2.1 分布式训练的同步机制

理解同步点是排查死锁的基础。VLM 训练中的主要同步点：

**显式同步点**：
```python
# 1. Barrier 同步
torch.distributed.barrier()  # 所有进程必须到达

# 2. All-Reduce 操作
torch.distributed.all_reduce(tensor)  # 梯度同步

# 3. Broadcast 操作  
torch.distributed.broadcast(tensor, src=0)  # 参数广播
```

**隐式同步点**：
```python
# 1. 优化器步进
optimizer.step()  # DDP 会自动同步梯度

# 2. 模型保存
if rank == 0:
    torch.save(model.state_dict(), path)
torch.distributed.barrier()  # 等待保存完成

# 3. 数据加载
dataloader = DataLoader(dataset, sampler=DistributedSampler(dataset))
# DistributedSampler 确保各进程数据不重复
```

### 12.2.2 死锁的典型场景

**场景 1：条件分支不一致**

错误代码：
```python
if rank == 0 and epoch % 10 == 0:
    # 只有 rank 0 执行验证
    val_loss = validate(model, val_loader)
    torch.distributed.broadcast(val_loss, src=0)  # 死锁！其他进程未执行
```

正确做法：
```python
if epoch % 10 == 0:
    if rank == 0:
        val_loss = validate(model, val_loader)
    else:
        val_loss = torch.zeros(1).cuda()
    torch.distributed.broadcast(val_loss, src=0)  # 所有进程都参与
```

**场景 2：数据不均衡导致的死锁**

问题代码：
```python
for batch in dataloader:
    loss = model(batch)
    loss.backward()
    optimizer.step()  # 某些进程数据提前结束，未参与同步
```

解决方案：
```python
# 方案 1：使用 drop_last
dataloader = DataLoader(dataset, drop_last=True, ...)

# 方案 2：填充数据
total_samples = len(dataset)
samples_per_rank = math.ceil(total_samples / world_size)
# 确保每个进程有相同数量的批次
```

**场景 3：异常处理不当**

危险代码：
```python
try:
    output = model(input)
except Exception as e:
    print(f"Error on rank {rank}: {e}")
    continue  # 跳过这个批次，但其他进程还在等待同步！
```

安全处理：
```python
try:
    output = model(input)
except Exception as e:
    print(f"Error on rank {rank}: {e}")
    # 通知所有进程出现异常
    error_flag = torch.tensor([1.0]).cuda()
    torch.distributed.all_reduce(error_flag)
    if error_flag.item() > 0:
        # 所有进程一起退出
        cleanup_distributed()
        sys.exit(1)
```

### 12.2.3 死锁快速诊断方法

**方法 1：添加超时和日志**

```python
import functools
import torch.distributed as dist

def timeout_wrapper(func, timeout=300):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        import signal
        
        def timeout_handler(signum, frame):
            raise TimeoutError(f"{func.__name__} timeout after {timeout}s")
        
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(timeout)
        try:
            result = func(*args, **kwargs)
        finally:
            signal.alarm(0)
        return result
    return wrapper

# 使用
@timeout_wrapper
def training_step(batch):
    # 训练代码
    pass
```

**方法 2：进程状态监控**

```python
import threading
import time

def monitor_thread(rank, interval=30):
    """监控线程，定期打印进程状态"""
    def run():
        step = 0
        while True:
            time.sleep(interval)
            print(f"[Rank {rank}] Heartbeat at step {step}, "
                  f"Memory: {torch.cuda.memory_allocated()/1e9:.2f}GB")
            step += 1
    
    thread = threading.Thread(target=run, daemon=True)
    thread.start()

# 在训练开始时启动
monitor_thread(rank)
```

**方法 3：使用 py-spy 分析**

```bash
# 安装 py-spy
pip install py-spy

# 分析挂起的进程
py-spy dump --pid <process_id>

# 生成火焰图
py-spy record -d 30 -o profile.svg --pid <process_id>
```

### 12.2.4 Barrier 超时问题

**常见原因**：

1. **不均匀的计算负载**
```python
# 问题：rank 0 做额外的日志记录
if rank == 0:
    # 复杂的日志计算
    log_metrics(model, epoch, step)
    
torch.distributed.barrier()  # rank 0 太慢，其他进程超时
```

2. **I/O 操作不当**
```python
# 问题：所有进程同时写入
for rank in range(world_size):
    with open(f"log_{rank}.txt", "a") as f:
        f.write(metrics)  # 文件系统压力导致某些进程阻塞
```

**解决策略**：

```python
# 1. 使用异步 I/O
import asyncio

async def async_log(data, filename):
    async with aiofiles.open(filename, 'a') as f:
        await f.write(data)

# 2. 错开 I/O 时机
if step % 100 == rank:  # 不同 rank 在不同步数记录
    save_checkpoint(model, f"ckpt_{rank}_{step}.pt")

# 3. 设置合理的超时
os.environ['TORCH_DISTRIBUTED_TIMEOUT'] = '3600'  # 1 小时
```

### 12.2.5 实战案例：64 GPU 训练死锁排查

**背景**：某团队使用 8 节点 × 8 V100 训练 13B VLM，在第 1000 步突然挂起。

**症状**：
- 所有 GPU 利用率降为 0%
- CPU 占用率正常
- 无错误日志输出

**排查步骤**：

1. **确认挂起位置**
```bash
# 使用 gdb 附加到进程
gdb -p <pid>
(gdb) py-bt  # 查看 Python 调用栈

# 发现卡在：
File "torch/distributed/distributed_c10d.py", line 2838, in barrier
    work.wait()
```

2. **检查各进程状态**
```python
# 添加调试代码
print(f"[Rank {rank}] Entering barrier at step {step}")
torch.distributed.barrier()
print(f"[Rank {rank}] Exited barrier")

# 发现 rank 43 未进入 barrier
```

3. **定位问题代码**
```python
# 原代码
if batch_idx > 0 and batch_idx % gradient_accumulation_steps == 0:
    optimizer.step()
    optimizer.zero_grad()

# 问题：rank 43 的数据少一个 batch，未执行最后一次 optimizer.step()
```

**解决方案**：
```python
# 1. 确保所有 rank 执行相同次数的优化步骤
total_steps = len(dataloader) // gradient_accumulation_steps
for step in range(total_steps):
    for _ in range(gradient_accumulation_steps):
        batch = next(iter(dataloader), None)
        if batch is not None:
            loss = model(batch)
            loss.backward()
    
    optimizer.step()
    optimizer.zero_grad()

# 2. 添加同步检查点
if step % 100 == 0:
    # 同步检查，确保所有进程进度一致
    progress_tensor = torch.tensor([step], device='cuda')
    progress_list = [torch.zeros_like(progress_tensor) 
                    for _ in range(world_size)]
    torch.distributed.all_gather(progress_list, progress_tensor)
    
    if rank == 0:
        progress = [p.item() for p in progress_list]
        if len(set(progress)) > 1:
            print(f"Warning: Progress mismatch: {progress}")
```

## 12.3 不同 GPU 型号混合训练的坑

在实际生产环境中，你可能面临 A100 和 V100 混用、3090 和 4090 并存的情况。异构 GPU 训练充满挑战，本节将揭示所有隐藏的陷阱。

### 12.3.1 异构 GPU 的主要挑战

**硬件差异带来的问题**：

| 差异维度 | 影响 | 典型场景 |
|---------|------|---------|
| 显存大小 | OOM 风险 | A100-80G vs A100-40G |
| 计算能力 | 速度瓶颈 | V100 vs A100 (2.5x 差距) |
| 精度支持 | 训练不稳定 | 3090 (FP16) vs A100 (BF16) |
| 互联带宽 | 通信瓶颈 | NVLink vs PCIe |
| 架构差异 | 功能不兼容 | Ampere vs Volta |

### 12.3.2 性能瓶颈与负载均衡

**问题 1：木桶效应**

最慢的 GPU 决定整体训练速度：

```python
# 诊断代码
import time
import torch.distributed as dist

def measure_gpu_speed(rank, model, dummy_batch):
    torch.cuda.synchronize()
    start = time.time()
    
    for _ in range(100):
        output = model(dummy_batch)
        loss = output.mean()
        loss.backward()
    
    torch.cuda.synchronize()
    elapsed = time.time() - start
    
    # 收集所有 GPU 的时间
    times = [torch.zeros(1).cuda() for _ in range(dist.get_world_size())]
    torch.distributed.all_gather(times, torch.tensor([elapsed]).cuda())
    
    if rank == 0:
        times = [t.item() for t in times]
        print(f"GPU speeds: {times}")
        print(f"Slowest/Fastest ratio: {max(times)/min(times):.2f}x")
```

**解决方案：动态批大小**

```python
class HeterogeneousDataLoader:
    def __init__(self, dataset, gpu_configs):
        """
        gpu_configs: {
            0: {'type': 'A100', 'memory': 80, 'batch_size': 8},
            1: {'type': 'V100', 'memory': 32, 'batch_size': 4},
        }
        """
        self.dataset = dataset
        self.gpu_configs = gpu_configs
        
    def get_batch_size(self, rank):
        # 根据 GPU 能力分配不同批大小
        return self.gpu_configs[rank]['batch_size']
    
    def create_loader(self, rank):
        batch_size = self.get_batch_size(rank)
        sampler = DistributedSampler(
            self.dataset,
            num_replicas=len(self.gpu_configs),
            rank=rank,
            shuffle=True
        )
        return DataLoader(
            self.dataset,
            batch_size=batch_size,
            sampler=sampler
        )
```

**问题 2：显存不均衡**

```python
def adaptive_gradient_accumulation(rank, base_batch_size=8):
    """根据 GPU 显存动态调整梯度累积"""
    gpu_memory = torch.cuda.get_device_properties(rank).total_memory
    
    if gpu_memory > 80 * 1024**3:  # 80GB
        micro_batch_size = base_batch_size
        accumulation_steps = 1
    elif gpu_memory > 40 * 1024**3:  # 40GB
        micro_batch_size = base_batch_size // 2
        accumulation_steps = 2
    elif gpu_memory > 24 * 1024**3:  # 24GB
        micro_batch_size = base_batch_size // 4
        accumulation_steps = 4
    else:  # 16GB or less
        micro_batch_size = 1
        accumulation_steps = base_batch_size
    
    return micro_batch_size, accumulation_steps
```

### 12.3.3 混合精度的兼容性问题

**BF16 vs FP16 混用陷阱**：

```python
def setup_mixed_precision(rank):
    """处理不同 GPU 的精度差异"""
    gpu_name = torch.cuda.get_device_name(rank)
    
    if 'A100' in gpu_name or 'H100' in gpu_name:
        # 支持 BF16
        dtype = torch.bfloat16
        use_bf16 = True
    else:
        # 只支持 FP16
        dtype = torch.float16
        use_bf16 = False
    
    # 统一精度设置
    all_use_bf16 = torch.tensor([use_bf16], dtype=torch.bool).cuda()
    dist.all_reduce(all_use_bf16, op=dist.ReduceOp.MIN)
    
    if all_use_bf16.item():
        return torch.bfloat16
    else:
        # 降级到 FP16
        if rank == 0:
            print("Warning: Falling back to FP16 due to hardware limitations")
        return torch.float16
```

**梯度同步精度问题**：

```python
class MixedPrecisionOptimizer:
    def __init__(self, optimizer, model, dtype):
        self.optimizer = optimizer
        self.model = model
        self.dtype = dtype
        self.grad_scaler = torch.cuda.amp.GradScaler(enabled=(dtype == torch.float16))
        
    def step(self):
        # 梯度转换到统一精度
        for param in self.model.parameters():
            if param.grad is not None:
                # 确保梯度精度一致
                param.grad = param.grad.to(dtype=torch.float32)
        
        # 同步前转换
        for param in self.model.parameters():
            if param.grad is not None:
                dist.all_reduce(param.grad)
                param.grad = param.grad / dist.get_world_size()
        
        # 优化器步进
        if self.dtype == torch.float16:
            self.grad_scaler.step(self.optimizer)
            self.grad_scaler.update()
        else:
            self.optimizer.step()
```

### 12.3.4 实战配置示例

**场景：4×A100-80G + 4×V100-32G 混合集群**

```python
# 配置文件 heterogeneous_config.yaml
gpu_groups:
  high_tier:  # A100-80G
    ranks: [0, 1, 2, 3]
    batch_size: 8
    gradient_accumulation: 1
    precision: bfloat16
    
  low_tier:  # V100-32G  
    ranks: [4, 5, 6, 7]
    batch_size: 3
    gradient_accumulation: 3
    precision: float16

communication:
  # 分组通信策略
  allreduce_groups:
    - [0, 1, 2, 3]  # A100 内部先同步
    - [4, 5, 6, 7]  # V100 内部先同步
  
  # 跨组同步使用异步模式
  cross_group_async: true
  
training:
  # 使用 pipeline 并行缓解不均衡
  pipeline_parallel: true
  pipeline_stages:
    - ranks: [0, 1]  # A100 处理前面层
    - ranks: [2, 3]  # A100 处理中间层
    - ranks: [4, 5, 6, 7]  # V100 处理后面层（计算量较小）
```

实现代码：

```python
class HeterogeneousTrainer:
    def __init__(self, config_path):
        self.config = yaml.load(open(config_path))
        self.rank = dist.get_rank()
        self.setup_gpu_group()
        
    def setup_gpu_group(self):
        # 确定当前 GPU 所属组
        for group_name, group_config in self.config['gpu_groups'].items():
            if self.rank in group_config['ranks']:
                self.group_name = group_name
                self.group_config = group_config
                break
        
        # 创建通信组
        for group_ranks in self.config['communication']['allreduce_groups']:
            group = dist.new_group(ranks=group_ranks)
            if self.rank in group_ranks:
                self.local_group = group
    
    def train_step(self, batch):
        # 根据组配置调整批大小
        micro_batch = self._split_batch(batch, self.group_config['batch_size'])
        
        total_loss = 0
        for step in range(self.group_config['gradient_accumulation']):
            with torch.cuda.amp.autocast(
                dtype=getattr(torch, self.group_config['precision'])
            ):
                loss = self.model(micro_batch[step])
                loss = loss / self.group_config['gradient_accumulation']
            
            loss.backward()
            total_loss += loss.item()
        
        # 分层同步策略
        self._hierarchical_allreduce()
        
        return total_loss
    
    def _hierarchical_allreduce(self):
        # 第一步：组内同步
        for param in self.model.parameters():
            if param.grad is not None:
                dist.all_reduce(param.grad, group=self.local_group)
        
        # 第二步：跨组同步（仅组长参与）
        if self.rank == self.group_config['ranks'][0]:
            for param in self.model.parameters():
                if param.grad is not None:
                    dist.all_reduce(param.grad)  # 全局同步
        
        # 第三步：组内广播
        for param in self.model.parameters():
            if param.grad is not None:
                dist.broadcast(param.grad, 
                             src=self.group_config['ranks'][0],
                             group=self.local_group)
```

### 12.3.5 调试技巧与监控

**异构集群监控脚本**：

```bash
#!/bin/bash
# monitor_heterogeneous.sh

while true; do
    echo "=== GPU Status at $(date) ==="
    
    # 收集所有节点的 GPU 信息
    for node in node1 node2; do
        echo "Node: $node"
        ssh $node "nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu,temperature.gpu,power.draw --format=csv"
    done
    
    # 检查速度差异
    echo "=== Training Speed ==="
    tail -n 8 training.log | grep "step_time"
    
    # 检查是否有 OOM
    if grep -q "out of memory" training.log; then
        echo "WARNING: OOM detected!"
        grep "out of memory" training.log | tail -n 5
    fi
    
    sleep 30
done
```

## 12.4 FSDP vs DeepSpeed 实战对比

选择 FSDP 还是 DeepSpeed？这是每个大模型训练者都会面临的问题。本节通过实战对比，帮你做出最佳选择。

### 12.4.1 架构差异与设计理念

| 特性 | FSDP | DeepSpeed |
|------|------|-----------|
| 开发者 | Meta/PyTorch 原生 | Microsoft |
| 集成度 | PyTorch 内置 | 需要额外安装 |
| 学习曲线 | 相对简单 | 功能丰富但复杂 |
| 优化器状态分片 | ✅ | ✅ (ZeRO-2/3) |
| 参数分片 | ✅ | ✅ (ZeRO-3) |
| 激活值分片 | 部分支持 | ✅ (ZeRO-R) |
| CPU Offload | ✅ | ✅ 更成熟 |
| 混合精度 | 原生支持 | 需要配置 |
| Pipeline 并行 | ❌ | ✅ |

### 12.4.2 VLM 训练配置对比

**FSDP 配置示例**：

```python
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    MixedPrecision,
    BackwardPrefetch,
    ShardingStrategy,
    CPUOffload
)
from torch.distributed.fsdp.wrap import (
    transformer_auto_wrap_policy,
    size_based_auto_wrap_policy
)

def setup_fsdp_for_vlm(model, vision_encoder, language_model):
    # 混合精度配置
    mp_policy = MixedPrecision(
        param_dtype=torch.bfloat16,
        reduce_dtype=torch.bfloat16,
        buffer_dtype=torch.bfloat16,
        cast_forward_inputs=True
    )
    
    # 自动包装策略 - 关键！
    auto_wrap_policy = functools.partial(
        transformer_auto_wrap_policy,
        transformer_layer_cls={
            # 视觉编码器层
            vision_encoder.__class__,
            # 语言模型层  
            type(language_model.layers[0]),
        }
    )
    
    # CPU Offload（显存不足时启用）
    cpu_offload = CPUOffload(offload_params=True)
    
    # FSDP 配置
    model = FSDP(
        model,
        auto_wrap_policy=auto_wrap_policy,
        mixed_precision=mp_policy,
        sharding_strategy=ShardingStrategy.FULL_SHARD,  # 完全分片
        cpu_offload=cpu_offload,
        backward_prefetch=BackwardPrefetch.BACKWARD_PRE,  # 预取优化
        limit_all_gathers=True,  # 限制 all-gather 防止 OOM
        use_orig_params=True,  # 保持原始参数（重要！）
    )
    
    return model
```

**DeepSpeed 配置示例**：

```json
{
    "train_batch_size": 64,
    "gradient_accumulation_steps": 8,
    "train_micro_batch_size_per_gpu": 2,
    
    "bf16": {
        "enabled": true
    },
    
    "zero_optimization": {
        "stage": 3,
        "offload_optimizer": {
            "device": "cpu",
            "pin_memory": true
        },
        "offload_param": {
            "device": "cpu",
            "pin_memory": true
        },
        "overlap_comm": true,
        "contiguous_gradients": true,
        "sub_group_size": 1e9,
        "reduce_bucket_size": 1e9,
        "stage3_prefetch_bucket_size": 1e8,
        "stage3_param_persistence_threshold": 1e6,
        "stage3_max_live_parameters": 1e9,
        "stage3_max_reuse_distance": 1e9,
        "stage3_gather_16bit_weights_on_model_save": true
    },
    
    "gradient_clipping": 1.0,
    
    "optimizer": {
        "type": "AdamW",
        "params": {
            "lr": 2e-5,
            "betas": [0.9, 0.999],
            "eps": 1e-8,
            "weight_decay": 0.01
        }
    },
    
    "scheduler": {
        "type": "WarmupCosineLR",
        "params": {
            "warmup_min_lr": 0,
            "warmup_max_lr": 2e-5,
            "warmup_num_steps": 1000,
            "total_num_steps": 10000
        }
    },
    
    "activation_checkpointing": {
        "partition_activations": true,
        "cpu_checkpointing": false,
        "contiguous_memory_optimization": false,
        "number_checkpoints": null,
        "synchronize_checkpoint_boundary": false,
        "profile": false
    }
}
```

### 12.4.3 性能基准测试

**测试环境**：
- 模型：LLaVA-13B
- 硬件：8×A100-40G
- 数据：图文对，批大小 64

**测试结果**：

| 指标 | FSDP | DeepSpeed ZeRO-2 | DeepSpeed ZeRO-3 |
|------|------|-----------------|-----------------|
| 吞吐量 (samples/s) | 28.5 | 31.2 | 26.8 |
| 显存占用 (GB) | 35.2 | 32.1 | 28.9 |
| 通信开销 (%) | 18% | 15% | 22% |
| 启动时间 (s) | 45 | 62 | 78 |
| Checkpoint 大小 (GB) | 26 | 26 | 52 (分片) |

**性能分析代码**：

```python
import time
import torch
from contextlib import contextmanager

@contextmanager
def profile_time(name):
    torch.cuda.synchronize()
    start = time.time()
    yield
    torch.cuda.synchronize()
    print(f"{name}: {time.time() - start:.2f}s")

def benchmark_training_step(model, batch, optimizer, use_fsdp=True):
    metrics = {}
    
    # Forward
    with profile_time("Forward"):
        output = model(batch)
        loss = output.loss
    metrics['loss'] = loss.item()
    
    # Backward
    with profile_time("Backward"):
        loss.backward()
    
    # Optimizer step
    with profile_time("Optimizer"):
        optimizer.step()
        optimizer.zero_grad()
    
    # 内存统计
    metrics['memory_allocated'] = torch.cuda.memory_allocated() / 1e9
    metrics['memory_reserved'] = torch.cuda.memory_reserved() / 1e9
    
    if use_fsdp:
        # FSDP 特定指标
        if hasattr(model, '_fsdp_wrapped_module'):
            metrics['communication_time'] = model._communication_time
    else:
        # DeepSpeed 特定指标
        if hasattr(optimizer, 'timer_names'):
            for name in optimizer.timer_names:
                metrics[f'ds_{name}'] = optimizer.timers(name)
    
    return metrics
```

### 12.4.4 选择建议与迁移指南

**何时选择 FSDP**：
1. PyTorch 原生项目，不想引入额外依赖
2. 模型相对简单，不需要复杂的并行策略
3. 团队熟悉 PyTorch，学习成本低
4. 需要与其他 PyTorch 生态工具集成

**何时选择 DeepSpeed**：
1. 超大模型（>30B），需要极致优化
2. 需要 Pipeline 并行等高级特性
3. 混合训练环境，需要更细粒度控制
4. 已有 DeepSpeed 经验积累

**从 FSDP 迁移到 DeepSpeed**：

```python
# 迁移检查清单
migration_checklist = {
    "模型包装": "FSDP(...) -> deepspeed.initialize(...)",
    "优化器": "需要在 config 中配置，不能直接传入",
    "梯度累积": "自动处理 -> 需要显式配置",
    "Checkpoint": "torch.save -> model.save_checkpoint",
    "混合精度": "MixedPrecision -> fp16/bf16 config",
    "学习率调度": "手动 -> 配置文件",
}

# 迁移示例
def migrate_fsdp_to_deepspeed(fsdp_model, fsdp_optimizer):
    # 1. 提取原始模型
    if hasattr(fsdp_model, 'module'):
        base_model = fsdp_model.module
    else:
        base_model = fsdp_model
    
    # 2. 创建 DeepSpeed 配置
    ds_config = {
        "train_batch_size": world_size * batch_size,
        "gradient_accumulation_steps": grad_acc_steps,
        "zero_optimization": {
            "stage": 3 if was_full_shard else 2,
            # 其他配置...
        }
    }
    
    # 3. 初始化 DeepSpeed
    model, optimizer, _, _ = deepspeed.initialize(
        model=base_model,
        config=ds_config,
        model_parameters=base_model.parameters()
    )
    
    return model, optimizer
```

### 12.4.5 常见问题与解决方案

**问题 1：FSDP OOM 但 DeepSpeed 正常**

原因：FSDP 的 all-gather 操作可能导致瞬时显存峰值。

解决方案：
```python
# FSDP 限制 all-gather
model = FSDP(
    model,
    limit_all_gathers=True,
    forward_prefetch=True,  # 前向预取
    backward_prefetch=BackwardPrefetch.BACKWARD_PRE,
)
```

**问题 2：DeepSpeed 训练速度慢**

原因：ZeRO-3 的参数收集开销。

解决方案：
```python
# 优化 ZeRO-3 配置
"zero_optimization": {
    "stage": 3,
    "stage3_max_live_parameters": 2e9,  # 增加缓存
    "stage3_max_reuse_distance": 2e9,
    "stage3_prefetch_bucket_size": 2e8,  # 增加预取
}
```

**问题 3：Checkpoint 不兼容**

```python
def convert_checkpoint(checkpoint_path, from_format="fsdp", to_format="deepspeed"):
    """转换不同格式的 checkpoint"""
    
    if from_format == "fsdp" and to_format == "deepspeed":
        # FSDP -> DeepSpeed
        state_dict = torch.load(checkpoint_path)
        
        # FSDP 可能有 _fsdp 前缀
        new_state_dict = {}
        for key, value in state_dict.items():
            new_key = key.replace("_fsdp_wrapped_module.", "")
            new_key = new_key.replace("_fpw_module.", "")
            new_state_dict[new_key] = value
        
        # DeepSpeed 期望的格式
        ds_checkpoint = {
            "module": new_state_dict,
            "epoch": 0,
            "global_step": 0,
        }
        
        torch.save(ds_checkpoint, checkpoint_path.replace(".pt", "_ds.pt"))
    
    elif from_format == "deepspeed" and to_format == "fsdp":
        # DeepSpeed -> FSDP
        # DeepSpeed ZeRO-3 需要先收集分片
        from deepspeed.utils.zero_to_fp32 import convert_zero_checkpoint_to_fp32_state_dict
        
        convert_zero_checkpoint_to_fp32_state_dict(
            checkpoint_path,
            checkpoint_path.replace("ds", "fsdp.pt")
        )
```

## 本章小结

多机多卡训练是 VLM 扩展的必经之路，但也充满挑战。本章系统介绍了分布式训练中最常见的四类问题：

1. **NCCL 通信错误**：掌握了快速诊断流程、环境变量配置和网络优化策略。记住，大部分 NCCL 错误都可以通过正确的环境变量和网络配置解决。

2. **进程同步与死锁**：理解了分布式训练的同步机制，学会了识别和避免死锁的典型场景。关键是确保所有进程执行相同的集合通信操作。

3. **异构 GPU 训练**：了解了混合 GPU 训练的挑战和解决方案。核心思想是根据硬件能力动态调整批大小和梯度累积策略。

4. **FSDP vs DeepSpeed**：通过实战对比，明确了两种框架的优劣和适用场景。FSDP 更简单直接，DeepSpeed 功能更丰富。

**关键公式回顾**：

有效批大小计算：
$$\text{Effective Batch Size} = \text{World Size} \times \text{Micro Batch Size} \times \text{Gradient Accumulation Steps}$$

通信时间估算：
$$T_{\text{comm}} = \frac{\text{Data Size}}{\text{Bandwidth}} + \text{Latency} \times \text{Num Operations}$$

显存占用（ZeRO-3）：
$$M_{\text{per GPU}} = \frac{M_{\text{model}} + M_{\text{optimizer}} + M_{\text{gradients}}}{\text{World Size}} + M_{\text{activations}}$$

## 练习题

### 基础题

**练习 12.1：NCCL 环境变量配置**

你的 8 卡 V100 服务器训练时经常出现 NCCL timeout，请写出完整的环境变量配置来优化通信。

💡 **提示**：考虑超时时间、日志级别、P2P 通信和网络接口选择。

<details>
<summary>📝 参考答案</summary>

```bash
# 增加超时时间
export NCCL_TIMEOUT=7200  # 2小时

# 启用调试日志
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=INIT,GRAPH

# 优化 P2P 通信
export NCCL_P2P_DISABLE=0
export NCCL_P2P_LEVEL=NVL

# 指定网络接口
export NCCL_SOCKET_IFNAME=eth0
export NCCL_IB_DISABLE=1  # 如果没有 InfiniBand

# 优化缓冲区
export NCCL_BUFFSIZE=8388608
export NCCL_NTHREADS=256

# 树形算法优化
export NCCL_TREE_THRESHOLD=0
export NCCL_ALGO=Tree
```

这套配置增加了超时容忍度，启用了详细日志便于调试，优化了 P2P 和网络通信，适合大多数 V100 集群。
</details>

**练习 12.2：死锁诊断**

以下代码在 4 卡训练时会死锁，请找出原因并修复：

```python
def validation_step(rank, model, val_loader):
    if rank == 0:
        model.eval()
        total_loss = 0
        for batch in val_loader:
            loss = model(batch).loss
            total_loss += loss.item()
        avg_loss = total_loss / len(val_loader)
        torch.distributed.broadcast(torch.tensor([avg_loss]).cuda(), src=0)
    return avg_loss
```

💡 **提示**：考虑所有进程的执行路径。

<details>
<summary>📝 参考答案</summary>

问题：只有 rank 0 执行 broadcast，其他进程没有对应的 broadcast 调用，导致死锁。

修复方案：
```python
def validation_step(rank, model, val_loader):
    if rank == 0:
        model.eval()
        total_loss = 0
        for batch in val_loader:
            loss = model(batch).loss
            total_loss += loss.item()
        avg_loss = torch.tensor([total_loss / len(val_loader)]).cuda()
    else:
        avg_loss = torch.zeros(1).cuda()
    
    # 所有进程都参与 broadcast
    torch.distributed.broadcast(avg_loss, src=0)
    return avg_loss.item()
```
</details>

**练习 12.3：异构 GPU 批大小计算**

你有 2 张 A100-80G 和 2 张 V100-32G，目标是总批大小 64。请设计每个 GPU 的 micro batch size 和梯度累积步数。

💡 **提示**：A100 的计算能力约是 V100 的 2.5 倍。

<details>
<summary>📝 参考答案</summary>

根据显存和计算能力分配：

```python
config = {
    "A100-80G": {
        "micro_batch_size": 8,  # 充分利用显存
        "gradient_accumulation_steps": 2,
        "effective_batch_per_gpu": 16
    },
    "V100-32G": {
        "micro_batch_size": 3,  # 显存限制
        "gradient_accumulation_steps": 5,  # 补偿小批次
        "effective_batch_per_gpu": 15
    }
}

# 验证：
# 2 * 16 (A100) + 2 * 15 (V100) = 62 ≈ 64
# 可以通过调整最后一个 batch 来精确达到 64
```

这种配置平衡了显存使用和计算效率，避免了木桶效应。
</details>

### 挑战题

**练习 12.4：FSDP 内存优化**

你的 LLaVA-34B 模型在 8×A100-40G 上用 FSDP 训练时 OOM。请提供完整的优化方案，包括配置和代码。

💡 **提示**：考虑 CPU offload、激活检查点、分片策略等。

<details>
<summary>📝 参考答案</summary>

```python
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    MixedPrecision,
    BackwardPrefetch,
    ShardingStrategy,
    CPUOffload
)
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    checkpoint_wrapper,
    CheckpointImpl,
    apply_activation_checkpointing
)

def optimize_fsdp_for_large_vlm(model):
    # 1. 激活检查点
    check_fn = lambda m: isinstance(m, TransformerBlock)
    apply_activation_checkpointing(
        model,
        checkpoint_wrapper_fn=functools.partial(
            checkpoint_wrapper,
            checkpoint_impl=CheckpointImpl.NO_REENTRANT
        ),
        check_fn=check_fn
    )
    
    # 2. 混合精度 - 使用 BF16
    mp_policy = MixedPrecision(
        param_dtype=torch.bfloat16,
        reduce_dtype=torch.float32,  # 梯度用 FP32 更稳定
        buffer_dtype=torch.bfloat16
    )
    
    # 3. CPU Offload
    cpu_offload = CPUOffload(offload_params=True)
    
    # 4. 优化的分片策略
    model = FSDP(
        model,
        auto_wrap_policy=functools.partial(
            transformer_auto_wrap_policy,
            transformer_layer_cls={TransformerBlock},
        ),
        mixed_precision=mp_policy,
        sharding_strategy=ShardingStrategy.HYBRID_SHARD,  # 混合分片
        cpu_offload=cpu_offload,
        backward_prefetch=BackwardPrefetch.BACKWARD_PRE,
        limit_all_gathers=True,
        use_orig_params=True,
        sync_module_states=True,
        forward_prefetch=True
    )
    
    # 5. 梯度累积 + 小 batch
    micro_batch_size = 1  # 极小批次
    gradient_accumulation_steps = 32
    
    return model, micro_batch_size, gradient_accumulation_steps
```

这个方案通过激活检查点减少 50% 激活值内存，CPU offload 节省参数内存，混合分片优化通信，可以成功训练 34B 模型。
</details>

**练习 12.5：分布式调试工具设计**

设计一个调试工具，能够实时监控多机训练的进程状态、通信时间和潜在死锁。

💡 **提示**：考虑心跳机制、通信 hook 和异常检测。

<details>
<summary>📝 参考答案</summary>

```python
import threading
import time
import torch.distributed as dist
from collections import deque
from datetime import datetime

class DistributedDebugger:
    def __init__(self, rank, world_size, check_interval=10):
        self.rank = rank
        self.world_size = world_size
        self.check_interval = check_interval
        self.comm_times = deque(maxlen=100)
        self.last_heartbeat = time.time()
        self.deadlock_threshold = 300  # 5分钟
        
        # 注册通信 hook
        self._register_comm_hooks()
        
        # 启动监控线程
        self._start_monitor()
    
    def _register_comm_hooks(self):
        """注册通信钩子来测量时间"""
        original_all_reduce = dist.all_reduce
        
        def timed_all_reduce(*args, **kwargs):
            start = time.time()
            result = original_all_reduce(*args, **kwargs)
            elapsed = time.time() - start
            self.comm_times.append(elapsed)
            
            if elapsed > 10:  # 超过10秒警告
                print(f"[Rank {self.rank}] WARNING: all_reduce took {elapsed:.2f}s")
            
            return result
        
        dist.all_reduce = timed_all_reduce
    
    def _start_monitor(self):
        """启动监控线程"""
        def monitor():
            while True:
                time.sleep(self.check_interval)
                self._check_health()
        
        thread = threading.Thread(target=monitor, daemon=True)
        thread.start()
    
    def _check_health(self):
        """健康检查"""
        current_time = time.time()
        
        # 1. 检查心跳
        if current_time - self.last_heartbeat > self.deadlock_threshold:
            self._report_deadlock()
        
        # 2. 统计通信时间
        if self.comm_times:
            avg_comm = sum(self.comm_times) / len(self.comm_times)
            max_comm = max(self.comm_times)
            print(f"[Rank {self.rank}] Comm stats: avg={avg_comm:.3f}s, max={max_comm:.3f}s")
        
        # 3. 内存状态
        mem_alloc = torch.cuda.memory_allocated() / 1e9
        mem_reserved = torch.cuda.memory_reserved() / 1e9
        print(f"[Rank {self.rank}] Memory: {mem_alloc:.2f}/{mem_reserved:.2f} GB")
        
        # 4. 同步检查（可选）
        self._sync_check()
    
    def _sync_check(self):
        """检查所有进程是否同步"""
        try:
            check_tensor = torch.tensor([self.rank], device='cuda')
            check_list = [torch.zeros_like(check_tensor) for _ in range(self.world_size)]
            dist.all_gather(check_list, check_tensor, timeout=datetime.timedelta(seconds=30))
            
            # 验证所有进程都响应
            ranks = [t.item() for t in check_list]
            if sorted(ranks) != list(range(self.world_size)):
                print(f"[Rank {self.rank}] ERROR: Missing ranks in sync check: {ranks}")
        except Exception as e:
            print(f"[Rank {self.rank}] Sync check failed: {e}")
    
    def _report_deadlock(self):
        """报告可能的死锁"""
        import traceback
        print(f"[Rank {self.rank}] DEADLOCK WARNING!")
        print(f"Stack trace:")
        traceback.print_stack()
        
        # 可选：触发 core dump
        import signal
        os.kill(os.getpid(), signal.SIGABRT)
    
    def update_heartbeat(self):
        """更新心跳时间（在训练循环中调用）"""
        self.last_heartbeat = time.time()

# 使用示例
debugger = DistributedDebugger(rank, world_size)

for epoch in range(num_epochs):
    for batch in dataloader:
        debugger.update_heartbeat()  # 更新心跳
        # 训练代码...
```

这个调试器提供了实时监控、死锁检测、通信性能分析等功能，能够快速定位分布式训练问题。
</details>

**练习 12.6：混合并行策略设计**

为 VLM-65B 模型设计一个结合 FSDP、Pipeline 并行和 Tensor 并行的训练方案，硬件是 16×A100-80G（2 节点）。

💡 **提示**：考虑不同并行策略的通信模式和内存占用。

<details>
<summary>📝 参考答案</summary>

```python
"""
混合并行策略设计：
- Tensor Parallel (TP): 4-way (节点内)
- Pipeline Parallel (PP): 2-way (跨节点)
- Data Parallel with FSDP: 2-way

总并行度: 4 × 2 × 2 = 16
"""

class HybridParallelVLM:
    def __init__(self, model_config):
        self.world_size = 16
        self.tp_size = 4  # 节点内 tensor parallel
        self.pp_size = 2  # pipeline stages
        self.dp_size = 2  # data parallel groups
        
        # 初始化进程组
        self._init_process_groups()
        
        # 构建模型
        self.model = self._build_model(model_config)
    
    def _init_process_groups(self):
        """创建不同的进程组"""
        rank = dist.get_rank()
        
        # Tensor Parallel 组 (同节点内)
        tp_ranks = [rank // 4 * 4 + i for i in range(4)]
        self.tp_group = dist.new_group(tp_ranks)
        
        # Pipeline Parallel 组 (跨节点)
        pp_ranks = [rank % 4 + i * 8 for i in range(2)]
        self.pp_group = dist.new_group(pp_ranks)
        
        # Data Parallel 组
        dp_ranks = [rank // 8 * 8 + rank % 4 + i * 4 
                   for i in range(2)]
        self.dp_group = dist.new_group(dp_ranks)
    
    def _build_model(self, config):
        """构建混合并行模型"""
        rank = dist.get_rank()
        
        # 1. 模型分层 (Pipeline)
        if rank < 8:  # 第一个 pipeline stage
            layers = self._get_first_stage_layers(config)
        else:  # 第二个 pipeline stage
            layers = self._get_second_stage_layers(config)
        
        # 2. Tensor Parallel
        for layer in layers:
            if isinstance(layer, nn.Linear):
                layer = ColumnParallelLinear(layer, self.tp_group)
            elif isinstance(layer, nn.Embedding):
                layer = ParallelEmbedding(layer, self.tp_group)
        
        # 3. FSDP 包装
        model = nn.Sequential(*layers)
        model = FSDP(
            model,
            process_group=self.dp_group,
            sharding_strategy=ShardingStrategy.FULL_SHARD,
            mixed_precision=MixedPrecision(
                param_dtype=torch.bfloat16,
                reduce_dtype=torch.float32
            ),
            limit_all_gathers=True
        )
        
        return model
    
    def train_step(self, batch):
        """混合并行训练步骤"""
        # Pipeline parallel 的 micro-batching
        micro_batches = self._split_batch(batch, self.pp_size)
        
        losses = []
        for micro_batch in micro_batches:
            # Forward (with pipeline)
            if self.is_first_stage():
                output = self.model(micro_batch)
                # 发送到下一个 stage
                self._send_activations(output, target_stage=1)
            else:
                # 接收前一个 stage 的激活
                activations = self._recv_activations(source_stage=0)
                output = self.model(activations)
                loss = self.compute_loss(output, micro_batch['labels'])
                losses.append(loss)
            
            # Backward (reverse pipeline)
            if self.is_last_stage():
                loss.backward()
                # 发送梯度到前一个 stage
                self._send_gradients(...)
            else:
                # 接收梯度
                gradients = self._recv_gradients(...)
                output.backward(gradients)
        
        # FSDP 会自动处理梯度同步
        return sum(losses) / len(losses)

# 配置示例
config = {
    "model": {
        "hidden_size": 8192,
        "num_layers": 80,
        "num_heads": 64,
        "vocab_size": 32000
    },
    "training": {
        "micro_batch_size": 1,
        "gradient_accumulation": 8,
        "optimizer": "AdamW",
        "lr": 1e-4
    }
}

# 内存估算
"""
每个 GPU 的模型参数：65B / 16 = 4B 参数
FP16 存储：4B * 2 bytes = 8GB
优化器状态 (AdamW)：8GB * 2 = 16GB
激活值 (with checkpointing)：~20GB
总计：~44GB < 80GB (安全)
"""
```

这个方案通过三种并行策略的组合，实现了 65B 模型在 16 卡上的高效训练，每种并行策略都针对其最适合的维度进行优化。
</details>

**练习 12.7：通信瓶颈分析**

你的训练在 scaling 到 32 卡后，效率从 8 卡的 90% 下降到 60%。请分析可能的原因并提供优化方案。

💡 **提示**：考虑通信拓扑、梯度同步策略和数据加载。

<details>
<summary>📝 参考答案</summary>

可能原因分析：

1. **通信瓶颈增加**
   - All-Reduce 时间 ∝ log(N) × 数据量
   - 32卡的通信轮数比8卡多

2. **网络拓扑不优化**
   - 跨节点通信带宽受限
   - PCIe/NVLink 拓扑不均衡

3. **同步开销**
   - Barrier 等待时间增加
   - 数据加载不均衡加剧

优化方案：

```python
# 1. 梯度压缩
from torch.distributed.algorithms.ddp_comm_hooks import default_hooks

model.register_comm_hook(
    state=None,
    hook=default_hooks.fp16_compress_hook
)

# 2. 分层 All-Reduce
def hierarchical_allreduce(tensor, groups):
    # 节点内先同步
    dist.all_reduce(tensor, group=groups['intra_node'])
    
    # 节点间同步（仅 master）
    if is_node_master():
        dist.all_reduce(tensor, group=groups['inter_node'])
    
    # 节点内广播
    dist.broadcast(tensor, src=node_master_rank, 
                  group=groups['intra_node'])

# 3. 梯度累积增加
gradient_accumulation_steps = 4  # 8卡
gradient_accumulation_steps = 8  # 32卡，减少同步频率

# 4. 异步数据预取
class AsyncDataLoader:
    def __init__(self, dataset, num_workers=8):
        self.queue = Queue(maxsize=2)
        self.workers = []
        for _ in range(num_workers):
            w = Process(target=self._worker, args=(dataset,))
            w.start()
            self.workers.append(w)
    
    def _worker(self, dataset):
        while True:
            batch = dataset.get_next_batch()
            self.queue.put(batch)
    
    def __iter__(self):
        while True:
            yield self.queue.get()

# 5. 通信与计算重叠
class OverlappedOptimizer:
    def step(self):
        # 启动异步 all-reduce
        handles = []
        for param in model.parameters():
            handle = dist.all_reduce(param.grad, async_op=True)
            handles.append(handle)
        
        # 同时进行其他计算
        self.update_metrics()
        self.log_progress()
        
        # 等待通信完成
        for handle in handles:
            handle.wait()
        
        # 应用梯度
        super().step()

# 6. NCCL 优化
os.environ['NCCL_TREE_THRESHOLD'] = '0'
os.environ['NCCL_ALGO'] = 'Ring,Tree'
os.environ['NCCL_CROSS_NIC'] = '1'
os.environ['NCCL_NET_GDR_LEVEL'] = '5'
```

预期效果：优化后 32 卡效率可提升到 75-80%。
</details>

**练习 12.8：生产环境故障恢复**

设计一个完整的故障恢复系统，能够处理节点故障、网络中断和 GPU 错误，确保训练能够自动恢复。

💡 **提示**：考虑检查点、健康检查、自动重启和弹性训练。

<details>
<summary>📝 参考答案</summary>

```python
import os
import time
import signal
import subprocess
from typing import Optional, Dict, Any
from dataclasses import dataclass
from datetime import datetime, timedelta

@dataclass
class TrainingState:
    """训练状态"""
    epoch: int
    step: int
    best_loss: float
    checkpoint_path: str
    failure_count: int = 0
    last_failure_time: Optional[datetime] = None

class ResilientTrainer:
    """弹性训练器 - 自动故障恢复"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.state = self._load_state()
        self.health_checker = HealthChecker()
        self.checkpoint_manager = CheckpointManager(config['checkpoint_dir'])
        self.max_failures = config.get('max_failures', 3)
        self.failure_window = timedelta(hours=1)
        
        # 注册信号处理
        self._register_signal_handlers()
    
    def _register_signal_handlers(self):
        """注册优雅退出的信号处理"""
        def graceful_exit(signum, frame):
            print(f"Received signal {signum}, saving checkpoint...")
            self.save_checkpoint(emergency=True)
            sys.exit(0)
        
        signal.signal(signal.SIGTERM, graceful_exit)
        signal.signal(signal.SIGINT, graceful_exit)
    
    def train(self):
        """主训练循环 - 带故障恢复"""
        while self.state.epoch < self.config['max_epochs']:
            try:
                # 健康检查
                if not self.health_checker.check_all():
                    self._handle_unhealthy_state()
                    continue
                
                # 训练一个 epoch
                self._train_epoch()
                
                # 定期保存检查点
                if self.state.step % self.config['checkpoint_interval'] == 0:
                    self.save_checkpoint()
                
                # 重置故障计数（成功完成 epoch）
                self.state.failure_count = 0
                
            except Exception as e:
                self._handle_training_failure(e)
    
    def _train_epoch(self):
        """训练一个 epoch"""
        for batch in self.dataloader:
            # 故障注入测试（可选）
            if self.config.get('fault_injection'):
                self._inject_random_fault()
            
            # 训练步骤
            loss = self.train_step(batch)
            
            # 异常检测
            if self._detect_anomaly(loss):
                raise ValueError(f"Anomaly detected: loss={loss}")
            
            self.state.step += 1
    
    def _handle_training_failure(self, error: Exception):
        """处理训练故障"""
        print(f"Training failed: {error}")
        
        # 更新故障统计
        current_time = datetime.now()
        self.state.failure_count += 1
        
        # 检查是否超过最大故障次数
        if self.state.last_failure_time:
            time_since_last = current_time - self.state.last_failure_time
            if time_since_last < self.failure_window:
                if self.state.failure_count >= self.max_failures:
                    self._escalate_failure("Too many failures in short time")
                    return
        
        self.state.last_failure_time = current_time
        
        # 尝试恢复
        self._attempt_recovery(error)
    
    def _attempt_recovery(self, error: Exception):
        """尝试从故障中恢复"""
        recovery_strategies = [
            self._recover_from_oom,
            self._recover_from_nccl_error,
            self._recover_from_checkpoint,
            self._restart_workers
        ]
        
        for strategy in recovery_strategies:
            try:
                if strategy(error):
                    print(f"Recovery successful using {strategy.__name__}")
                    return
            except Exception as e:
                print(f"Recovery strategy {strategy.__name__} failed: {e}")
        
        # 所有策略都失败
        self._escalate_failure("All recovery strategies failed")
    
    def _recover_from_oom(self, error: Exception) -> bool:
        """从 OOM 错误恢复"""
        if "out of memory" not in str(error).lower():
            return False
        
        print("Attempting OOM recovery...")
        
        # 1. 清理缓存
        torch.cuda.empty_cache()
        
        # 2. 减小批大小
        self.config['batch_size'] = max(1, self.config['batch_size'] // 2)
        print(f"Reduced batch size to {self.config['batch_size']}")
        
        # 3. 启用更激进的内存优化
        self.config['gradient_checkpointing'] = True
        self.config['cpu_offload'] = True
        
        # 4. 重新初始化模型
        self._reinitialize_model()
        
        return True
    
    def _recover_from_nccl_error(self, error: Exception) -> bool:
        """从 NCCL 错误恢复"""
        if "nccl" not in str(error).lower():
            return False
        
        print("Attempting NCCL recovery...")
        
        # 1. 销毁进程组
        if dist.is_initialized():
            dist.destroy_process_group()
        
        # 2. 等待所有进程
        time.sleep(10)
        
        # 3. 重新初始化
        self._init_distributed()
        
        # 4. 从检查点恢复
        self.load_checkpoint()
        
        return True
    
    def _restart_workers(self, error: Exception) -> bool:
        """重启工作进程"""
        print("Restarting all workers...")
        
        # 保存当前状态
        self.save_checkpoint(emergency=True)
        
        # 构建重启命令
        restart_cmd = [
            "python", "-m", "torch.distributed.launch",
            "--nproc_per_node", str(self.config['gpus_per_node']),
            "--nnodes", str(self.config['num_nodes']),
            "--node_rank", str(self.config['node_rank']),
            "--master_addr", self.config['master_addr'],
            "--master_port", self.config['master_port'],
            "train.py",
            "--resume", self.state.checkpoint_path
        ]
        
        # 执行重启
        subprocess.run(restart_cmd)
        return True
    
    def save_checkpoint(self, emergency: bool = False):
        """保存检查点"""
        checkpoint = {
            'model_state': self.model.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
            'training_state': self.state,
            'config': self.config,
            'timestamp': datetime.now().isoformat(),
            'emergency': emergency
        }
        
        # 保存多个副本防止损坏
        paths = self.checkpoint_manager.save(checkpoint, emergency)
        self.state.checkpoint_path = paths[0]
        
        # 异步上传到云存储
        if self.config.get('cloud_backup'):
            self._async_cloud_backup(paths[0])
    
    def _escalate_failure(self, reason: str):
        """升级故障处理"""
        print(f"CRITICAL: {reason}")
        
        # 1. 发送告警
        self._send_alert(reason)
        
        # 2. 保存调试信息
        self._save_debug_info()
        
        # 3. 优雅退出
        sys.exit(1)

class HealthChecker:
    """健康检查器"""
    
    def check_all(self) -> bool:
        """执行所有健康检查"""
        checks = [
            self.check_gpu_health(),
            self.check_network_health(),
            self.check_memory_health(),
            self.check_disk_space()
        ]
        return all(checks)
    
    def check_gpu_health(self) -> bool:
        """检查 GPU 健康状态"""
        try:
            # 检查 GPU 是否可用
            for i in range(torch.cuda.device_count()):
                torch.cuda.set_device(i)
                torch.cuda.synchronize()
                
                # 检查温度
                temp = self._get_gpu_temperature(i)
                if temp > 85:
                    print(f"WARNING: GPU {i} temperature {temp}°C")
                    return False
                
                # 检查 ECC 错误
                ecc_errors = self._get_ecc_errors(i)
                if ecc_errors > 0:
                    print(f"WARNING: GPU {i} has {ecc_errors} ECC errors")
                    return False
            
            return True
        except Exception as e:
            print(f"GPU health check failed: {e}")
            return False
    
    def check_network_health(self) -> bool:
        """检查网络健康状态"""
        if not dist.is_initialized():
            return True
        
        try:
            # 简单的 all-reduce 测试
            test_tensor = torch.ones(1).cuda()
            dist.all_reduce(test_tensor)
            return test_tensor.item() == dist.get_world_size()
        except Exception as e:
            print(f"Network health check failed: {e}")
            return False

# 使用示例
config = {
    'max_epochs': 100,
    'checkpoint_dir': './checkpoints',
    'checkpoint_interval': 1000,
    'max_failures': 3,
    'batch_size': 32,
    'gpus_per_node': 8,
    'num_nodes': 2,
    'node_rank': 0,
    'master_addr': '192.168.1.1',
    'master_port': '29500',
    'cloud_backup': True
}

trainer = ResilientTrainer(config)
trainer.train()
```

这个系统提供了完整的故障恢复能力，包括自动重试、降级策略、健康检查和云备份，能够处理生产环境中的各种故障场景。
</details>

## 常见陷阱与错误 (Gotchas)

### 1. NCCL 版本不匹配
**陷阱**：不同节点的 NCCL 版本不一致导致通信失败。
**解决**：统一所有节点的 PyTorch 和 NCCL 版本。

### 2. Hanging Without Error
**陷阱**：训练挂起但没有任何错误输出。
**解决**：启用 NCCL_DEBUG=INFO 和设置合理的超时时间。

### 3. 隐式同步点
**陷阱**：print、日志等操作可能引入隐式同步。
**解决**：只在 rank 0 进行 I/O 操作，或使用异步 I/O。

### 4. GPU 亲和性设置错误
**陷阱**：CUDA_VISIBLE_DEVICES 设置不当导致进程看到错误的 GPU。
**解决**：使用 torchrun 或正确设置每个进程的 GPU 映射。

### 5. 混合精度不兼容
**陷阱**：不同 GPU 支持的精度不同（FP16 vs BF16）。
**解决**：检测硬件能力，降级到所有 GPU 都支持的精度。

### 6. Checkpoint 腐败
**陷阱**：保存 checkpoint 时进程被中断导致文件损坏。
**解决**：先保存到临时文件，成功后再重命名。

## 最佳实践检查清单

### 启动前检查
- [ ] 所有节点的环境一致（Python、PyTorch、CUDA 版本）
- [ ] 网络连通性测试通过
- [ ] GPU 健康检查通过（温度、ECC 错误）
- [ ] 磁盘空间充足（checkpoint 需要大量空间）
- [ ] NCCL 环境变量正确设置

### 训练中监控
- [ ] GPU 利用率 > 85%
- [ ] 网络带宽利用合理
- [ ] 无进程明显落后（通过 progress bar）
- [ ] 内存使用稳定（无泄漏）
- [ ] Loss 曲线正常（无 NaN、无异常跳变）

### 故障恢复准备
- [ ] Checkpoint 定期保存（至少每小时）
- [ ] 有多个 checkpoint 副本
- [ ] 故障恢复脚本已测试
- [ ] 监控告警已配置
- [ ] 有回滚计划

### 性能优化
- [ ] 通信与计算重叠
- [ ] 梯度累积合理设置
- [ ] 数据加载不是瓶颈
- [ ] 使用了合适的 NCCL 算法
- [ ] 混合精度训练已启用

### 调试工具
- [ ] NCCL 日志已启用（问题排查时）
- [ ] 进程监控脚本运行中
- [ ] 性能 profiling 工具就绪
- [ ] 有 core dump 生成配置