# MLA RoPE KV Cache Write 海光平台失败分析与修复报告

日期：2026-08-29

## 1. 摘要

测试文件 tests/test_mla_rope_kv_cache_write.py 在天数平台能够通过，但在当前海光
DCU 平台初始运行结果为：

~~~text
7 passed, 2 failed
~~~

两个失败并非同一个问题：

1. 海光 PyTorch 暴露 HIP backend，算子会进入 output-anchored fallback。该回退
   只将伪 query 截断到实际 token 数，却把包含 CUDA Graph padding 行的完整
   kv_c 和 k_pe 继续传给融合算子，破坏了原 API 允许 source 行数大于实际 token
   数的契约，最终触发 token 维度校验失败。
2. 非连续 cache 测试使用 torch.empty 创建待比较张量。海光显存分配器返回的
   未初始化区域中出现 NaN，而 torch.allclose 默认不认为两个同位置的 NaN 相等，
   因此即使算子与 reference 对 cache 的实际修改一致，整张量比较仍然失败。

修复后，HIP fallback 会先按实际 token 数统一截断 kv_c 和 k_pe，再将相同视图用于
伪 query 和 cache source。测试侧将非连续 cache 的底层存储由 torch.empty 改为
torch.randn，在保留 stride 覆盖的同时消除未初始化数据。强制 fallback 的回归测试
也扩展为 source token 数大于实际 token 数，确保该缺陷可以在所有后端稳定复现。

最终结果：

~~~text
pytest -q tests/test_mla_rope_kv_cache_write.py
9 passed

pytest -q tests/test_mla_rope_concat_and_cache.py \
          tests/test_mla_rope_kv_cache_write.py
17 passed
~~~

## 2. 问题背景

mla_rope_kv_cache_write 将每个有效 token 的压缩 MLA latent 与旋转后的 RoPE
向量写入 paged KV cache：

~~~text
[kv_c[token] | RoPE(k_pe[token], positions[token])]
                    |
                    v
kv_cache[slot // cache_block_size, slot % cache_block_size]
~~~

接口支持以下重要语义：

- kv_c 形状为 [S, L]；
- k_pe 形状为 [S, R] 或 [S, 1, R]；
- slot_mapping 和 positions 的长度为实际 token 数 T；
- S 可以大于 T，以支持 CUDA Graph padding source；
- slot_mapping 中的 -1 表示跳过该 token；
- kv_cache 原地更新，函数返回 None；
- source 和 cache 可以是非连续张量。

因此，source tensor 第一维和实际写入 token 数不一定相等：

~~~text
S >= T
~~~

这项契约正是第一个海光失败场景的关键。

## 3. 复现环境

当前海光环境记录如下：

| 项目 | 值 |
| --- | --- |
| Python | 3.11.9 |
| PyTorch | 2.9.0 |
| torch.version.cuda | None |
| torch.version.hip | 6.3.26093 |
| Triton | 3.5.1 |
| pytest | 9.1.1 |
| NineToothed | /root/private_data/ninetoothed/src/ninetoothed/__init__.py |
| 设备名称 | BW |

复现命令：

~~~bash
pytest -vv -s tests/test_mla_rope_kv_cache_write.py
~~~

初始结果：

~~~text
collected 9 items
7 passed
2 failed
~~~

失败用例为：

~~~text
test_mla_rope_kv_cache_write_supports_padding_source_rows
test_mla_rope_kv_cache_write_strided_and_alias
~~~

## 4. 平台执行路径差异

wrapper 通过 torch.version.hip 判断是否需要 output-anchored cache write：

~~~python
def _requires_output_anchored_cache_write():
    return torch.version.hip is not None
~~~

两类平台由此进入不同路径：

| 平台 | backend 特征 | 执行路径 |
| --- | --- | --- |
| 天数 | torch.version.hip 为 None | 直接执行 cache-only kernel |
| 海光 DCU | torch.version.hip 非 None | 复用 mla_rope_concat_and_cache 的 output-anchored fallback |

路径关系如下：

~~~text
mla_rope_kv_cache_write
        |
        +-- 非 HIP --> cache-only kernel
        |
        +-- HIP ----> 构造单头伪 query
                      |
                      v
              mla_rope_concat_and_cache
                      |
                      +-- 生成真实 query 输出，供 SSA/backend 锚定
                      +-- 执行语义相同的 paged cache 写入
                      +-- wrapper 丢弃 query 输出并返回 None
~~~

采用 fallback 的原因是当前 HIP/AMD Triton 路径无法稳定编译只包含 cache 副作用、
没有真实输出根的 cache writer。mla_rope_concat_and_cache 已有真实输出张量，
其 cache 更新语义又与 cache-only 接口相同，因此被用于海光兼容路径。

天数平台没有进入该 fallback，所以不会触发其中的 token 维度不一致问题。这解释了
同一测试在两个平台上表现不同的直接原因。

## 5. 根因一：HIP fallback 破坏 padding-source 契约

### 5.1 失败现象

padding-source 测试使用：

~~~text
kv_c.shape          = [4, 8]
k_pe.shape          = [4, 1, 8]
slot_mapping.shape  = [2]
positions.shape     = [2]
~~~

即 source token 数 S=4，实际 token 数 T=2。

原 fallback 构造调用时，伪 query 被截断为 T 行，但 cache source 仍保留 S 行：

~~~python
num_tokens = slot_mapping.shape[0]
mla_rope_concat_and_cache(
    kv_c[:num_tokens].unsqueeze(1),  # [T, 1, L]
    k_pe[:num_tokens].unsqueeze(1),  # [T, 1, R]
    kv_c,                            # [S, L]
    k_pe,                            # [S, R]
    ...
)
~~~

mla_rope_concat_and_cache 的输入校验要求 query batch、kv_c 和 k_pe 的 token
维度完全相等：

~~~python
if kv_c.shape[0] != batch or k_pe.shape[0] != batch:
    raise ValueError("all token dimensions must match")
~~~

因此海光路径稳定报错：

~~~text
ValueError: all token dimensions must match
~~~

### 5.2 根本原因

cache-only API 的契约是 S >= T，只读取 source 的前 T 行；而被复用的
mla_rope_concat_and_cache 契约是所有 token 维度严格等于 batch。

兼容回退只转换了执行路径，没有完整转换接口契约。它对伪 query 做了 S 到 T 的
投影，却漏掉了传给融合算子的 kv_c/k_pe source 参数。

### 5.3 修复方案

进入融合算子前先生成统一的前 T 行视图，之后所有参数都使用同一批次：

~~~python
num_tokens = slot_mapping.shape[0]
kv_c = kv_c[:num_tokens]
k_pe = k_pe[:num_tokens]
mla_rope_concat_and_cache(
    kv_c.unsqueeze(1),
    k_pe.unsqueeze(1),
    kv_c,
    k_pe,
    ...
)
~~~

修复后的形状全部以 T 为 token 维：

~~~text
ql_nope.shape = [T, 1, L]
q_pe.shape    = [T, 1, R]
kv_c.shape    = [T, L]
k_pe.shape    = [T, R]
~~~

该修复具有以下性质：

- 只影响 torch.version.hip 非空的 fallback；
- 切片是视图操作，不复制 source 数据；
- 不改变原 tensor，也不改变公开接口；
- 只丢弃本来就不应参与当前 cache write 的 padding 行；
- rank-3 k_pe 已在进入 fallback 前规范化为 rank-2，因此 unsqueeze 后形状正确；
- 天数平台的 cache-only kernel 路径保持不变。

## 6. 根因二：测试依赖未初始化显存

### 6.1 失败现象

非连续张量测试原先使用：

~~~python
cache = torch.empty(
    4, 4, (latent + rope) * 2, device=device, dtype=torch.float16
)[..., ::2]
reference_cache = cache.clone()
~~~

该测试只写 slot 0 和 slot 5，其余 cache 区域应保持不变。测试最后比较整个 cache：

~~~python
torch.allclose(cache, reference_cache, rtol=2e-3, atol=2e-3)
~~~

海光失败输出中，未写入区域出现了 NaN。torch.empty 不初始化内存，其内容没有任何
数值保证；clone 会复制 NaN，但 torch.allclose 默认 equal_nan=False，因此：

~~~text
allclose(NaN, NaN) == False
~~~

这会把“未初始化内存中恰好含 NaN”错误地报告成算子数值不一致。

### 6.2 为何存在平台差异

不同设备、驱动、allocator 状态和之前运行的 kernel 都会改变 torch.empty 返回区域
中的残留位模式。天数平台上的残留数据在该次运行中没有触发 NaN 比较，而当前海光
平台出现了 NaN。

因此天数通过不是对该测试写法正确性的证明；原测试本质上依赖了未定义的初始状态。

### 6.3 修复方案

将底层存储改为有限随机值，同时继续通过步长切片构造非连续 view：

~~~python
cache = torch.randn(
    4, 4, (latent + rope) * 2, device=device, dtype=torch.float16
)[..., ::2]
~~~

该修改保留了测试的全部目标：

- cache 仍为非连续张量；
- wrapper 的 contiguous 临时张量和 copy-back 路径仍被覆盖；
- 写入 slot 的结果与 reference 对比；
- 未写入区域也继续参与整张量比较，可验证没有越界修改；
- 初始内容确定为正常数值，不再受 allocator 残留数据影响。

不建议简单添加 equal_nan=True，因为那会容忍算子在本应产生有限值的位置错误生成
NaN。使用有限初始化能更清楚地区分输入状态和算子行为。

## 7. 回归测试增强

强制 output-anchored fallback 的测试原先使用相同的 source token 数和实际 token
数，无法发现 S > T 时的契约转换遗漏。

修复后测试改为：

~~~python
source_tokens = 4
kv_c = torch.randn(source_tokens, latent, ...)
k_pe = torch.randn(source_tokens, rope, ...)
slots = torch.tensor((0, -1, 5), ...)
positions = torch.tensor((0, 1, 2), ...)
~~~

此时：

~~~text
S = 4
T = 3
~~~

测试仍通过 monkeypatch 强制执行 fallback，因此即使在天数/CUDA 平台上运行，也能
覆盖海光专用逻辑。若以后再次只截断伪 query 而遗漏 source，该测试会稳定触发
all token dimensions must match，而不需要依赖海光设备才能发现回归。

## 8. 修改范围

实现修改：

- src/ntops/torch/mla_rope_kv_cache_write.py
  - 在 output-anchored fallback 中统一截断 kv_c/k_pe；
  - 伪 query 和 cache source 共享相同的 T 行视图。

测试修改：

- tests/test_mla_rope_kv_cache_write.py
  - 非连续 cache 使用 torch.randn 初始化；
  - fallback 测试增加 padding source 行，覆盖 S > T。

未修改内容：

- 公开 Python API 和返回值；
- cache entry 布局；
- RoPE 数学计算；
- slot=-1 的跳过语义；
- 天数平台 cache-only kernel；
- mla_rope_concat_and_cache 的输入契约；
- autotuning 参数及默认值。

## 9. 验证结果

### 9.1 目标测试

~~~bash
pytest -q tests/test_mla_rope_kv_cache_write.py
~~~

结果：

~~~text
9 passed in 14.41s
~~~

目标测试随后在联合回归中再次执行并通过。

### 9.2 相邻算子联合回归

~~~bash
pytest -q tests/test_mla_rope_concat_and_cache.py \
          tests/test_mla_rope_kv_cache_write.py
~~~

结果：

~~~text
17 passed in 19.30s
~~~

这说明 fallback 所复用的融合算子及 cache-only wrapper 均未出现相邻回归。

### 9.3 补丁检查

~~~bash
git diff --check
~~~

结果：通过，没有空白符错误。

当前环境未安装 Ruff：

~~~text
/usr/local/bin/python: No module named ruff
~~~

因此 Ruff 静态检查未执行；目标模块已在 pytest 中完成导入、编译和实际 GPU 执行。

## 10. 兼容性与风险评估

| 项目 | 评估 |
| --- | --- |
| 天数路径 | 未修改，风险低 |
| 海光连续输入 | 语义不变，仅多两个 O(1) 切片视图 |
| 海光 padding source | 由失败修复为按前 T 行正确写入 |
| 非连续输入/cache | 原有连续化与 copy-back 逻辑不变 |
| 数值精度 | RoPE 和 cache store 计算未修改 |
| 性能 | 切片不复制；fallback 原有 query 输出开销不变 |
| API | 签名、返回值、异常边界均未扩大 |
| 测试稳定性 | 消除未初始化显存依赖，跨 allocator 更稳定 |

主要剩余风险是未来两个融合接口的输入契约继续演化时，fallback 适配层可能再次出现
语义漂移。建议让适配层的强制测试持续覆盖 S=T、S>T、rank-2/rank-3 k_pe、
slot=-1 和非连续 view。

## 11. 结论

本次问题不是海光 RoPE 数值精度不足，也不是 cache 写入算法本身错误，而是平台分支
暴露了两个可移植性缺陷：

1. HIP 兼容回退没有完整继承 cache-only API 的 padding-source 契约；
2. 测试使用未初始化显存作为全量数值比较的基线。

通过在 fallback 边界统一 token 视图，并让 stride 测试使用有限初值，当前海光平台
已从 7 passed、2 failed 恢复为 9 passed。联合回归 17 个用例全部通过，天数直接
kernel 路径未被修改。

本次修复也形成两条通用原则：

- 复用另一个 kernel 作为平台 fallback 时，必须显式转换完整接口契约，而不只是让
  张量形状足以进入 kernel；
- 需要验证“未写区域保持不变”时，应使用已知有限初值，不能把 torch.empty 的残留
  位模式当作稳定测试数据。
