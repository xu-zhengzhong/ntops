# NineToothed 跨 DCU 与天数平台算子实现注意事项

## 1. 文档目标

本文面向需要让同一份 NineToothed kernel 同时运行在以下两类环境中的开发者：

- 海光 DCU：PyTorch HIP + Triton AMD backend；
- 天数智芯：PyTorch CUDA 兼容接口 + CoreX/Triton backend。

这里的“跨平台”不是指 Python 接口能被导入，也不是指 NineToothed 能生成一份
Triton 源码，而是必须依次满足五个条件：

1. NineToothed frontend 能完成 lowering；
2. 对应 Triton backend 能完成编译；
3. kernel 能在设备上安全执行；
4. 数值和副作用与独立 PyTorch reference 一致；
5. 在真实输入上没有不可接受的性能回退。

本文结论主要来自本仓库 `rms_norm_gated`、两个 MLA cache writer 和
`scaled_grouped_mm` 的跨平台适配。性能优化的通用流程另见
[`ninetoothed_kernel_optimization_guide.md`](ninetoothed_kernel_optimization_guide.md)。

## 2. 先固定真实软件栈

### 2.1 不要只用“稳定版”描述编译器

同名 NineToothed 包可能来自 wheel、editable install、源码目录或仓库内软链接。
DCU 和天数机器也可能加载不同的 Triton backend。每次验证前至少记录：

```bash
python - <<'PY'
import inspect
import ninetoothed
import torch
import triton

print("ninetoothed:", inspect.getfile(ninetoothed))
print("torch:", torch.__version__)
print("torch cuda:", torch.version.cuda)
print("torch hip:", torch.version.hip)
print("triton:", triton.__version__)
print("device:", torch.cuda.get_device_name())
PY
```

测试报告中应保存输出，而不是仅写“使用稳定版”。尤其要确认：

- `ninetoothed.__file__` 没有意外指向开发源码；
- `PYTHONPATH` 没有把另一个 checkout 的 `src` 放在前面；
- 执行 `pytest` 的 Python 与打印版本信息的 Python 是同一个解释器；
- DCU 上 `torch.cuda` 是兼容 API，判断 backend 应检查 `torch.version.hip`；
- 清理或更换环境后没有复用旧版本生成的 kernel artifact。

### 2.2 稳定版与开发版必须物理隔离

推荐分别使用独立虚拟环境和独立 benchmark 进程：

```text
/opt/venvs/nt-stable/bin/python
/opt/venvs/nt-development/bin/python
```

不要在同一进程内修改 `sys.path` 后先后导入两个版本。Python 模块缓存、Triton
JIT cache 和 NineToothed artifact cache 都可能让结果失真。建议：

- 使用绝对路径调用目标环境的 `python -m pytest`；
- 设置 `PYTHONNOUSERSITE=1`，避免用户目录包抢占；
- 给不同编译器版本设置不同的 `TRITON_CACHE_DIR`；
- 修改 DSL 或 compiler 后启动新进程；
- 必须清缓存时只清理已确认属于本次测试的目录，不执行宽泛的递归删除。

开发版源码检查可以提前暴露问题，但不能替代部署所用稳定版的端到端运行。

## 3. 只使用两条 lowering 路径的公共子集

### 3.1 NineToothed frontend 也属于兼容性边界

当前两类机器曾分别走 legacy 代码生成路径和 SSA backend path。相同 DSL 表达式
可能在两条路径中产生不同 IR。因此算子不仅要适配 CUDA/HIP backend，还要适配
项目实际使用的 NineToothed frontend。

避免依赖以下实现细节：

- monkey patch 私有 compiler class 或 emitter；
- 导入 `ninetoothed.backends.*` 等非公开内部模块；
- 假设某个 Python 对象在 lowering 中一定是编译期常量；
- 依赖生成变量的名称或 artifact 目录结构；
- 只在一个 frontend 上检查生成源码。

私有 patch 在一台机器上“修好”编译，往往会在另一稳定版本中直接导入失败，或把
真正应由 compiler 处理的问题隐藏起来。kernel 应优先改写为公共 DSL 能表达的形式。

### 3.2 dtype 转换需要查看最终 Triton 源码

本仓库实际遇到过以下写法在 legacy frontend 可用，但经 SSA lowering 后生成：

```python
v7 = ntl.int32
```

最终 Triton JIT 中没有 `ntl` 命名空间，因此报 `NameError: ntl is not defined`。
对普通数值转换，当前两条路径已验证的写法是：

```python
# 跨当前 legacy/SSA 路径验证过的普通数值转换
packed = (packed + 0).to(ntl.int32)
value_f32 = (value + 0).to(ntl.float32)
output = result.to(ntl.bfloat16)
```

`+ 0` 用于把某些 load/别名表达式规范化为可转换的 tensor value。SSA 路径会把
上述 dtype 正确生成为 `tl.int32`、`tl.float32` 等。

这不是要求机械替换所有 `ntl.cast`：

- `bitcast=True` 是位级重解释，不能等价替换为普通 `.to()`；
- scalar constant、tensor、层级 tensor 的 dtype lowering 可能不同；
- compiler 升级后仍要重新检查生成代码和数值结果。

至少对目标 kernel 的可执行 `.py` artifact 做一次检查：生成源码中不应残留
`ntl.`，也不应只有 metadata/JSON 通过。JSON 中出现 DSL 名称不等于可执行源码有错。

### 3.3 区分 scalar dtype 与层级 tensor dtype

NineToothed arrangement 会产生层级 tensor。此时 `tensor.dtype` 可能描述一层
Tensor，而不是 application 中当前 scalar/block value 的基础 dtype。把它直接传给
cast，legacy 路径可能把 Tensor 当成可迭代对象递归展开，表现为编译极慢或内存暴涨。

更稳妥的做法是：

- 已知计算精度时显式使用 `ntl.float32` 等标量 dtype；
- 必须跟随输入 dtype 时，在进入正确层级后读取 `tensor[i].dtype`；
- arrangement 修改后重新确认 application 实参所处层级；
- 不根据一台机器上的偶然 lowering 结果推断另一条 frontend 路径。

## 4. 索引、shape 与整数类型

### 4.1 避免混合 `index` 和 `i64`

DCU SSA 路径曾报告：

```text
Cannot promote unsupported dtypes `i64` and `index`
```

典型原因是把 program/tile offset 产生的 compiler `index` 与从 tensor 加载的
`int64` slot 或 position 放在同一算术链中。推荐：

- 尽量使用 NineToothed 的 `source[...]`、tile 和 `.offsets()` 表达索引；
- 从 tensor 读取的 slot/position 在明确边界转换为 `ntl.int64`；
- 不要无必要地手写 `program_id * stride + arange` 指针算术；
- shape、stride、slot 和 position 的整数类型保持一致；
- block size 等不变参数通过 wrapper 校验并作为静态语义传入；
- 生成 Triton 后检查地址表达式中是否存在意外的 `i64/index` 混合。

例如 MLA 路径当前采用：

```python
position = positions.source[token_idx].to(ntl.int64)
slot = slot_mapping.source[token_idx].to(ntl.int64)
```

不要为了消除错误而把所有地址降为 `int32`。长上下文、较大 paged cache 和 byte
offset 乘法可能溢出；应消除类型混合，而不是牺牲地址范围。

### 4.2 无效索引必须在 store 之前被 mask

cache writer 常用负 slot 表示跳过 token。跨平台实现应保证：

- 负 slot 不参与实际地址解引用；
- store 的最终 mask 同时包含 token、feature 和 slot 有效性；
- 不先构造越界 view 再期望后续 Python 条件阻止访问；
- reference 明确验证无效 slot 对 cache 完全无修改。

不同 backend 对未使用的越界指针表达式容忍度可能不同。正确的 mask 和受控的地址
生成都需要检查，不能只依赖最终结果碰巧正确。

## 5. Arrangement 与 launch grid

### 5.1 先写清一个 program 负责什么

在写 DSL 之前，应明确：

```text
program grid 的每一维对应 token、expert、M tile 还是 N tile？
一个 program 内部的 reduction 轴是什么？
哪个 arranged tensor 决定外层 shape？
不同返回值的 outer shape 是否严格一致？
```

legacy 和 SSA 路径对无输出值、别名值及未使用 arrangement 结果的处理并不一定
相同。不能把第一项返回值“看起来正确”当成 grid 一定正确，应检查最终 launch grid
和边界 mask。

### 5.2 `floor_mode` 作用于整个 tile

`Tensor.tile(..., floor_mode=True)` 会对该 tile 的每个相关维度使用 floor 语义，
不是只裁剪开发者心里想处理的 reduction 维。本仓库的 grouped GEMM 曾因此在处理 K
打包时同时丢掉 M 尾块和 expert，留下未初始化输出。

需要精确裁剪某一维时，优先构造语义明确的 view，再使用正常 ceil tile。例如偶/奇
MXFP4 K 元素可分别使用 `[..., :-1]` 和 `[..., 1:]` 配合 dilation 表达，而不是给
完整三维 tile 开启 `floor_mode`。

必须测试：

- `M < BLOCK_M`；
- `M % BLOCK_M != 0`；
- `N % BLOCK_N != 0`；
- 最后一个 expert 或 token；
- reduction 尾块；
- 零 token expert。

### 5.3 不要依赖隐含广播与 stride

所有 arrangement 返回 tensor 的 outer shape 应可证明一致。对于广播数据，明确它
在哪些 grid 维共享。对 stride 的支持也应成为接口契约，而不是 accidental feature：

- 声明只支持 contiguous 时，在 wrapper 中验证或规范化；
- 声明支持 strided 时，测试每种允许的 stride pattern；
- in-place 输出若使用 contiguous 临时量，结束后必须 copy back；
- benchmark 的主要结果应包含 wrapper 中必要的复制成本；
- 已知 routed/jagged 内部切片的 stride 也要作为独立测试用例。

AMD backend 曾在某些复杂 strided specialization 的 `make_amdgcn` 阶段进程级
崩溃。这种问题无法由 Python `try/except` 或 autotuner 捕获。遇到此类问题时先将
输入规范化为已验证布局并缩小复现，再决定是放宽 kernel 还是收紧接口。

## 6. 仅副作用 kernel 的特殊风险

### 6.1 cache write 也需要稳定的数据根

NineToothed 根据 arrangement/application 的数据流建立输出图和 launch domain。
如果 application 只有对 cache 的 scatter store，而没有普通输出，某条 frontend
路径可能把关键值视为未使用、选择不稳定的 launch root，或生成异常庞大的索引式。

对只写 cache 的算子，建议按以下顺序处理：

1. 优先让 arrangement 返回一个真实、已消费的 driver/output；
2. 该输出与某个 source 值做等值写回，使编译器保留稳定数据流；
3. 明确验证 no-op 输出不会修改公开语义；
4. 如果 DCU 仍不稳定，复用一个已经通过验证且具有真实输出的融合 kernel 路径；
5. wrapper 负责丢弃内部锚定输出，但 cache 副作用仍须逐元素验证。

仅创建 driver 而不在 application 中使用通常不够。使用 stride-0 expand 伪造 driver
也可能生成 backend 不友好的索引；优先选取与真实 source 对齐的 scalar/view。

### 6.2 避免重复写共享 cache

RoPE query 可能有多个 head，而压缩 KV cache 对一个 token 只有一份 latent。若 grid
按 head 展开，每个 head 都写同一 cache 地址，会产生冗余带宽和潜在竞态。应让一个
明确的 lane/head 完成共享 cache 写入，其余 program 只处理各自输出。

即使多个 writer 写入相同值，在内存模型上也不能默认无害。测试既要验证数值，也要
检查生成 grid 中 writer 数量符合设计。

## 7. Triton CUDA/HIP 公共子集

### 7.1 从普通指令建立正确基线

跨平台主实现应先使用两个 backend 都稳定支持的指令：

- 普通 `tl.load` / `tl.store` 和 mask；
- FP32 reduction；
- 已验证 shape 的 BF16/FP16 `tl.dot`；
- 基础逐元素整数、位运算和数学函数。

不要在主路径中无条件依赖某个平台专用 intrinsic、私有 libdevice 名称或特定硬件
低精度指令。`scaled_grouped_mm` 的可移植方案采用 kernel 内 MXFP4 解码，再调用
普通 BF16 `dot`；这比假设 DCU 也支持相同 `tl.dot_scaled` lowering 更可靠。

若必须使用平台特化优化，应做到：

- wrapper 通过公开能力检测选择实现；
- 两条实现共享同一 PyTorch reference 和接口测试；
- 不支持的 backend 不会在导入阶段解析特化代码；
- 性能报告分别列出 portable 和 specialized 路径；
- portable fallback 不是用 `torch.matmul` 偷换 kernel 语义。

### 7.2 warp/wavefront 差异会改变资源占用

天数和 DCU 的 SIMD 执行宽度、寄存器分配和 occupancy 可能不同。AMD 目标常见
wavefront 64；相同 `num_warps=8` 可能意味着比 warp-32 平台多得多的活跃 lane 和
寄存器压力。因此：

- 先用 `num_stages=1` 和较小 `num_warps` 建立共同可运行配置；
- tile 要同时满足两端 `tl.dot` 的合法 shape；
- 不把本平台最快配置直接宣称为 DCU 最优；
- 共享固定配置优先选择两端都稳定的 Pareto 点；
- 允许平台分派时分别保存两端最佳配置，但数学接口保持一致。

编译器在 backend codegen 中崩溃时，优先降低 `BLOCK_M/BLOCK_N`、`num_warps` 和
`num_stages`，再检查 stride 与生成表达式复杂度。继续扩大 tile 通常只会增加定位
噪声。

## 8. 自动调优必须防止进程级失败

### 8.1 候选集合取两端能力的交集

Python autotuner 能记录普通编译异常，却无法从 Triton AMD compiler 的 segmentation
fault 中恢复。因此不能把未验证候选直接放进 DCU 进程内批量 autotune。

推荐流程：

1. 用一个两端已知可运行的固定配置完成正确性；
2. 每个候选配置放进独立子进程，设置超时；
3. 分别在天数和 DCU 上完成 compile、launch 和数值比较；
4. 排除导致崩溃、超时或数值错误的候选；
5. 仅将共同安全候选交给进程内 autotuner；
6. 分平台记录 winner，不把某一端的 winner 当成全局事实。

如果部署要求一份固定配置，应在两端安全候选中根据真实 workload 加权选择，而不是
只选择天数机器上延迟最低的一项。

### 8.2 正确区分调优耗时与稳态耗时

自动调优首次调用包含多次编译和候选 benchmark，不应计入稳态 kernel 加速比。正确
报告至少分为：

- 首次调用/调优总耗时；
- winner 缓存命中后的稳态耗时；
- wrapper 端到端耗时；
- 最佳配置及其 key。

固定为同一 winner 后，固定配置与 autotune 缓存命中的 kernel 稳态时间应接近；若
差异明显，应检查 benchmark 是否包含重新调优、不同 cache key、输入复制或同步方式。

autotune key 至少要覆盖会改变代码或性能的 shape、dtype、stride 和静态语义。对
非幂等 in-place 算子，还必须保证不同候选试跑前恢复输入状态，否则后续候选看到的
不是同一问题。

## 9. 跨平台正确性测试矩阵

### 9.1 “跳过”不是“通过”

测试汇总中的 `ssss` 只说明 skip 条件生效。曾经出现的 DCU `10 passed, 4 skipped`
没有执行四个核心数值 kernel，不能作为跨平台通过证据。跨平台数值测试不应按
`torch.version.hip` 整体跳过。

只有目标环境确实不存在某种公开 dtype 时，才可跳过该 dtype 的 API 测试；同时应
保留使用底层 `uint8` storage 的等价数值测试，确保核心 kernel 在 DCU 真正 launch。

完整报告必须列出：

- collected、passed、failed、skipped 的数量；
- 每个 skip 的原因；
- 是否实际进入目标 kernel；
- 失败发生在 frontend、Triton compile、backend codegen、launch 还是数值比较。

### 9.2 最小测试清单

| 维度 | 必测内容 |
| --- | --- |
| dtype | FP16、BF16、FP32，及算子所需原始低精度 storage |
| shape | 最小值、真实模型值、非 2 的幂、所有 tile 尾块 |
| layout | contiguous；只有接口承诺时才测允许的 strided layout |
| index | 第一个/最后一个 slot、负 slot、block 边界、长地址 |
| grouped | uniform、jagged、零 token expert、最后一个 expert |
| side effect | 目标位置正确，未命中 cache 保持不变 |
| alias | 接口允许的 alias；不允许的 alias 显式报错 |
| source | 可执行 artifact 无未解析 `ntl.`，关键 `tl.dot` 数量符合设计 |

PyTorch reference 应独立、直观，并明确 MXFP4 nibble、RoPE pairing、RMSNorm 归约
精度和 cache layout。不要用另一份相似 Triton kernel 作为唯一 reference。

### 9.3 编译崩溃时拆分测试进程

`Fatal Python error: Segmentation fault` 会终止整个 pytest 进程。此时日志中崩溃前
显示的若干点不等于整套验证成功。建议：

```bash
python -m pytest -vv tests/test_target.py
python -m pytest -vv tests/test_target.py::test_exact_case
```

定位到 shape/config 后，把它作为独立回归测试。风险较高的 stride 和 autotune 用例
可由外层测试启动子进程，这样一个 backend crash 不会掩盖其他结果。

## 10. 性能验收

### 10.1 两端分别测量

每个平台都应使用相同数学语义、相同 shape 和相同 benchmark 方法，分别测量：

- PyTorch reference 端到端时间；
- NineToothed wrapper 端到端时间；
- 有需要时再补充纯 kernel 时间；
- p50/p90 或 mean 的明确定义与迭代次数；
- warmup、repeat、同步方法；
- 固定配置或 autotune winner。

本仓库已有 benchmark 使用 `triton.testing.do_bench`，新增算子应沿用项目统一口径，
而不是为得到更高加速比自行更改统计方式。若 wrapper 为兼容 stride 引入
`.contiguous()`/copy-back，端到端结果必须包含这部分成本。

### 10.2 不把单平台数字外推

天数平台的最佳 `BLOCK_N`、warps 或 stages 只能说明该设备和输入 key。DCU 在没有
实测前只能标记“待验证”，不能沿用天数加速比。文档应记录：

- 设备型号与 backend；
- 软件版本和实际模块路径；
- 输入 shape/dtype/stride；
- winner 配置；
- reference 与 kernel 时间；
- 加速比公式；
- 未覆盖路径。

对 compiler 通用优化，只有在所有可正常通过的算子上跑 A/B，并且多数算子改善且
无显著回退，才适合合入。单个新算子则至少要覆盖其主要真实 workload，而不是只测
一个容易获益的小 shape。

## 11. 故障定位顺序

按照下面顺序定位，通常能减少无效尝试：

1. **wrapper 契约**：shape、dtype、device、contiguous、alias 校验是否正确；
2. **arrangement**：层级、outer shape、launch root、tile 尾块是否正确；
3. **NineToothed lowering**：是否出现 `i64/index`、不支持 dtype 或未使用值；
4. **生成 Triton Python**：是否残留 `ntl.`、grid 是否异常、mask 是否完整；
5. **Triton IR compile**：dot shape、dtype、constexpr 是否被 backend 接受；
6. **backend codegen**：LLVM/AMDGPU/CoreX 是否崩溃或资源超限；
7. **launch**：非法地址、竞态、错误 stream 或同步；
8. **数值**：reference、舍入位置、低精度编码、边界副作用；
9. **性能**：带宽、计算、launch、复制或重复工作哪个是瓶颈。

常见症状与优先检查项：

| 症状 | 优先检查 |
| --- | --- |
| `Cannot promote i64 and index` | loaded int64 与 tile/program index 的算术边界 |
| `NameError: ntl is not defined` | dtype/function对象泄漏到最终 Triton `.py` |
| `make_amdgcn` segmentation fault | tile/warps/stages、复杂 stride、巨型索引表达式 |
| 输出尾部未初始化 | grid、`floor_mode`、M/N/expert 尾块 mask |
| 全部 autotune candidate failed | 第一个根因；不要先增加候选数量 |
| 大量 `skipped` | 核心 kernel 是否根本没有执行 |
| cache 偶发错误 | 重复 writer、slot mask、alias 和 autotune 输入恢复 |

每个失败都应保存最小 shape、dtype、stride、配置、生成 artifact 路径和完整 traceback。
不要只保留 pytest 最后一行摘要。

## 12. 推荐开发流程

### 阶段 A：冻结语义

1. 写独立 PyTorch reference；
2. 明确支持和拒绝的 dtype/layout/alias；
3. 列出真实模型 shape；
4. 明确原地副作用和无效索引语义。

### 阶段 B：建立公共实现

1. 使用普通 load/store、FP32 reduction 和基础 `tl.dot`；
2. 选择较小固定 tile、`num_stages=1` 和保守 warps；
3. 显式处理 dtype 和 index 边界；
4. 为副作用 kernel 建立稳定输出锚点；
5. 检查两条 frontend 生成的可执行 Triton 源码。

### 阶段 C：两平台正确性

1. 天数运行完整数值测试，不只做 source test；
2. 推送相同 commit 到 DCU 运行相同测试；
3. 报告精确 pass/skip 数量；
4. 对 crash case 使用独立进程；
5. 两端均通过后才扩大 layout 和 shape 支持。

### 阶段 D：安全调优

1. 分阶段 benchmark 定位计算、带宽、launch 或复制瓶颈；
2. 独立子进程筛选每个候选；
3. 分平台记录安全候选和 winner；
4. 再启用 shape-keyed autotune；
5. 用同一 benchmark 口径记录稳态与首次调用成本。

## 13. 合入前检查表

- [ ] 实际导入的 NineToothed、PyTorch、Triton 路径和版本已记录；
- [ ] 稳定版与开发版环境、cache 和进程已隔离；
- [ ] 没有 monkey patch compiler 私有 API；
- [ ] dtype/index 写法在 legacy 与 SSA 路径均完成 lowering；
- [ ] 最终可执行 Triton `.py` 中没有未解析 `ntl.`；
- [ ] arrangement outer shape、grid 和所有尾块均有测试；
- [ ] side-effect-only kernel 具有稳定且实际消费的数据根；
- [ ] 无效 slot、零 token、最后一个 block 的副作用正确；
- [ ] strided/alias 支持范围由接口明确约束；
- [ ] 数值测试在 DCU/HIP 上真正执行，没有被平台 marker 跳过；
- [ ] autotune 候选逐个在独立进程完成 compile、launch 和数值验证；
- [ ] 两个平台分别报告最佳配置和性能，不互相外推；
- [ ] 端到端 benchmark 包含 wrapper 所需的数据复制；
- [ ] 文档列明暂不支持的路径，不用隐式 fallback 掩盖。

## 14. 核心原则

跨 DCU 和天数实现的关键不是堆叠更多 backend 分支，而是先把 kernel 收敛到两条
NineToothed frontend 与两个 Triton backend 的稳定公共子集。兼容性判断必须基于
相同 commit 的真实设备执行和数值比较；生成源码、compile-only、skip 或单平台
性能都只是中间证据。

只有公共实现经过完整验证后，才应通过清晰的能力分派增加平台专用优化。这样既能
保留可移植基线，也能让后续性能回归和 compiler 问题有可比较的参照。
