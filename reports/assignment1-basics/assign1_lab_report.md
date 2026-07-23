# Assignment 1 实验报告

## 可复现说明

所有已经序列化的 BPE 训练结果都按数据集保存在 `artifacts/tinystories/` 和 `artifacts/owt/` 中；每个目录下都有 `run.jsonl`、`vocab.jsonl` 和 `merges.jsonl`。`scripts/train_bpe_datasets.py` 只负责训练数据集并写入 JSONL 记录；本报告是根据这些记录和对应日志手写整理的，不由脚本自动生成。

训练 TinyStories tokenizer 的命令：

```bash
uv run python scripts/train_bpe_datasets.py train --dataset tinystories --input data/TinyStoriesV2-GPT4-train.txt --vocab-size 10000
```

训练 OpenWebText tokenizer 的命令：

```bash
uv run python scripts/train_bpe_datasets.py train --dataset owt --input data/owt_train.txt --vocab-size 32000
```

## Preparation

下载 TinyStories 和 OpenWebText 数据：

```bash
mkdir -p data
cd data
wget https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-train.txt
wget https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-valid.txt
wget https://huggingface.co/datasets/stanford-cs336/owt-sample/resolve/main/owt_train.txt.gz
wget https://huggingface.co/datasets/stanford-cs336/owt-sample/resolve/main/owt_valid.txt.gz
gunzip owt_train.txt.gz
gunzip owt_valid.txt.gz
```

## Problem: unicode1

### (a)

`chr(0)` 返回 Unicode 空字符，也就是 `U+0000`。

### (b)

它的 `repr` 表示是转义字符串 `\x00`，而直接 `print` 时会输出一个不可见的空控制字符。

### (c)

当空字符出现在文本中时，它仍然是字符串的一部分，也会计入字符串长度；只是通常没有可见的打印字形。

## Problem: unicode2

### (a)

在 byte-level tokenizer 中更适合使用 UTF-8，因为它是互联网上最主流的文本编码，对以 ASCII 为主的英文文本也更紧凑，同时仍然可以把任意 Unicode 字符串表示成字节序列。相比之下，UTF-16 和 UTF-32 会在常见英文文本中引入更多额外的零字节或固定宽度模式，浪费序列长度，也会让基于网页语料学到的 byte merge 不那么自然。

### (b)

例如，`"こんにちは".encode("utf-8")` 会让题目中的错误函数失败，因为这个函数试图逐个字节单独解码；但每个日文字符在 UTF-8 中都由多个字节共同表示，单独拆开的字节并不是合法的独立 UTF-8 字符。

### (c)

`b"\xff\xff"` 是一个无法解码成 Unicode 字符的两字节序列，因为 `0xff` 既不是合法的 UTF-8 起始字节，也不是合法的 UTF-8 continuation byte。

## Problem: train_bpe

我在 `cs336_basics/train_bpe.py` 中实现了 `train_bpe(input_path, vocab_size, special_tokens)`：它返回 `dict[int, bytes]` 形式的 vocab 和按生成顺序排列的 `list[tuple[bytes, bytes]]` merges；训练时会把 special tokens 当作硬边界处理，使用 GPT-2 风格的正则表达式做 pre-tokenization，并支持给实验脚本使用的进度回调。当前针对 `tests/test_train_bpe.py` 的三个测试均已通过，包括 `test_train_bpe_speed`、`test_train_bpe` 和 `test_train_bpe_special_tokens`。

## Problem: train_bpe_tinystories

### (a)

我在 TinyStories 上训练了一个 byte-level BPE tokenizer，最大词表大小为 10,000，并把 `<|endoftext|>` 加入为 special token。训练摘要、词表和 merges 已经分别序列化到 `artifacts/tinystories/run.jsonl`、`artifacts/tinystories/vocab.jsonl` 和 `artifacts/tinystories/merges.jsonl`；优化版训练总耗时为 `18:23.29`，峰值常驻内存为 `3459356 KB`（约 `3.3 GiB`），词表中最长的 token 是 `b' accomplishment'`，长度为 15 bytes，解码后是 `' accomplishment'`。这个结果是合理的，因为 TinyStories 中常见英文词片段，尤其是带前导空格的词，会因为频率较高而被 BPE 合并成单个 token。

### (b)

当前 tokenizer 训练中最慢的部分仍然是 BPE merge 阶段；优化版已经避免每轮完全重建 `pair_counts`，但每一轮仍需要扫描当前所有 pre-token、应用最佳 merge 并重建 `counts` 表，所以主要时间仍花在 merge loop 上。

## Problem: train_bpe_expts_owt

### (a)

我在 OpenWebText 上训练了一个 byte-level BPE tokenizer，最大词表大小为 32,000，并把结果序列化到 `artifacts/owt/run.jsonl`、`artifacts/owt/vocab.jsonl` 和 `artifacts/owt/merges.jsonl`。这次训练总耗时为 `11:02:13.77`，低于题目要求的 12 小时；峰值常驻内存为 `16202152 KB`（约 `15.45 GiB`），低于 100 GB；最终词表有 32,000 个 token，merges 有 31,743 条。

词表中最长的 token 是 `b'\xc3\x83\xc3\x82\xc3\x83\xc3\x82\xc3\x83\xc3\x82\xc3\x83\xc3\x82\xc3\x83\xc3\x82\xc3\x83\xc3\x82\xc3\x83\xc3\x82\xc3\x83\xc3\x82\xc3\x83\xc3\x82\xc3\x83\xc3\x82\xc3\x83\xc3\x82\xc3\x83\xc3\x82\xc3\x83\xc3\x82\xc3\x83\xc3\x82\xc3\x83\xc3\x82\xc3\x83\xc3\x82'`，长度为 64 bytes，按 UTF-8 解码后是 `ÃÂÃÂÃÂÃÂÃÂÃÂÃÂÃÂÃÂÃÂÃÂÃÂÃÂÃÂÃÂÃÂ`。这个结果是合理的，因为 OpenWebText 来自网页文本，里面会有编码 mojibake、重复片段、URL、格式残留和其他网页噪声；BPE 会把这些高频重复字节序列合并成长 token。

### (b)

TinyStories tokenizer 学到的 token 更偏儿童故事语料中的常见英文词和叙事表达，例如带前导空格的普通词片段，最长 token 只有 `b' accomplishment'`（15 bytes）。OpenWebText tokenizer 的词表更大、语料更杂，除了常见英文片段外还会学到网页噪声、编码异常、URL/标点模式和领域词，因此更适合压缩一般网页文本，但在 TinyStories 这种干净、风格单一的语料上不一定有同样针对性的词表分配。

## Problem: tokenizer

我实现了 `Tokenizer` 类来加载给定的 vocab 和 merges，并支持 `encode`、`encode_iterable` 和 `decode`。`encode` 会先保留 special tokens，再用 GPT-2 风格正则做 pre-tokenization，并按 merges 的 rank 顺序在每个 pre-token 内部合并；`decode` 则把 token id 对应的 bytes 拼接后用 UTF-8 解码，非法字节用 replacement character 处理。当前 `tests/test_tokenizer.py` 的结果为 `24 passed, 1 xfailed`，其中 xfail 是测试预期的 `encode` 大文件内存限制。

## Problem: tokenizer_experiments

### (a)

我从 TinyStories 和 OpenWebText 各取了 10 个文档，并分别用对应语料训练出的 tokenizer 编码。TinyStories tokenizer 在 TinyStories 样本上的压缩率为 `4.112 bytes/token`；OpenWebText tokenizer 在 OpenWebText 样本上的压缩率为 `4.691 bytes/token`。

### (b)

如果用 TinyStories tokenizer 编码 OpenWebText 样本，压缩率下降到 `3.189 bytes/token`，明显低于 OpenWebText tokenizer 的 `4.691 bytes/token`。这说明 TinyStories tokenizer 学到的词表更适合儿童故事语料，遇到网页文本中的长词、格式残留、URL 和领域词时会切得更碎。

### (c)

在 1,000,000 bytes 的前缀上估算吞吐量，TinyStories tokenizer 约为 `1.006 MB/s`，OpenWebText tokenizer 约为 `0.813 MB/s`。按 825GB 估算，tokenize Pile 大约分别需要 `227.8` 小时和 `281.9` 小时。

### (d)

我已经把训练集和验证集都编码成 `uint16` 的 NumPy array：TinyStories train 有 `540,796,778` 个 token，TinyStories valid 有 `5,461,210` 个 token；OpenWebText train 有 `2,727,120,452` 个 token，OpenWebText valid 有 `66,401,098` 个 token。`uint16` 合适是因为这里最大的词表大小是 32,000，所有 token id 都小于 `2^16 = 65,536`，因此 `uint16` 足够表示全部 token id，同时比 `int32` 或 `int64` 更省磁盘和内存。

## Problem: linear

实现无 bias 的 `Linear`（权重形状 `(d_out, d_in)`，截断正态初始化）。`test_linear` 通过。

## Problem: embedding

实现 `Embedding` 查表。`test_embedding` 通过。

## Problem: rmsnorm

实现 pre-norm 用的 `RMSNorm`（先升到 fp32 再算）。`test_rmsnorm` 通过。

## Problem: positionwise_feedforward

实现 `SwiGLU`：`w2(silu(w1(x)) * w3(x))`。`test_swiglu`、`test_silu_matches_pytorch` 通过。

## Problem: rope

实现 `RoPE`（奇偶维旋转，cos/sin buffer 预计算）。`test_rope` 通过。

## Problem: softmax

实现数值稳定的 `softmax`（减 max）。`test_softmax_matches_pytorch` 通过。

## Problem: scaled_dot_product_attention

实现 SDPA；课程约定 `mask=True` 保留、`False` 屏蔽。`test_scaled_dot_product_attention`、`test_4d_scaled_dot_product_attention` 通过。

## Problem: multihead_self_attention

实现因果 MHA（分头 QKV、可选 RoPE、`tril` causal mask、`output_proj`）。`test_multihead_self_attention`、`test_multihead_self_attention_with_rope` 通过。

## Problem: transformer_block

实现 pre-norm block：`x + attn(ln1(x))`，再 `x + ffn(ln2(x))`。`test_transformer_block` 通过。

## Problem: transformer_lm

实现完整 LM：`token_embeddings → N×TransformerBlock → ln_final → lm_head`。`test_transformer_lm`、`test_transformer_lm_truncated_input` 通过。

## Problem: transformer_accounting

约定：作业架构（无 bias、SwiGLU、RoPE 无绝对位置表、embedding 与 `lm_head` 不共享）；矩阵乘 FLOPs 用 \(2mnp\)；embedding 查表不计 matmul。小/中/大模型的 \(d_{ff}\) 取最接近 \(\frac{8}{3}d_{model}\) 的 64 倍数（2048 / 2752 / 3392），\(V=50257\)，\(T=1024\)（除非另行说明）。

参数量：
\[
2VD + L(4D^{2} + 3D\,d_{ff} + 2D) + D.
\]

前向 matmul（序列长 \(T\)）：每层 Q/K/V/O 投影 \(8TD^{2}\)，\(QK^{\top}\) 与 Attn\(V\) 各 \(2T^{2}D\)，SwiGLU \(6TD\,d_{ff}\)；最后 `lm_head` \(2TDV\)。

### (a)

GPT-2 XL（\(V{=}50257,\,T{=}1024,\,L{=}48,\,D{=}1600,\,d_{ff}{=}4288\)）：**1,640,452,800** 参数；fp32 加载约 **6.56 GB**（\(\times 4\) bytes）。

### (b)

| matmul | FLOPs（\(T{=}1024\)） |
|--------|----------------------|
| 每层 Q/K/V 投影 | \(3\cdot 2TD^{2}=1.573\times 10^{10}\)（\(\times 48\)） |
| 每层 O 投影 | \(2TD^{2}=5.243\times 10^{9}\)（\(\times 48\)） |
| 每层 \(QK^{\top}\) | \(2T^{2}D=3.355\times 10^{9}\)（\(\times 48\)） |
| 每层 Attn\(V\) | \(2T^{2}D=3.355\times 10^{9}\)（\(\times 48\)） |
| 每层 SwiGLU \(w_{1,2,3}\) | \(6TD\,d_{ff}=4.215\times 10^{10}\)（\(\times 48\)） |
| `lm_head` | \(2TDV=1.647\times 10^{11}\) |
| **合计** | **\(3.517\times 10^{12}\) FLOPs** |

### (c)

FLOPs 主要在 **SwiGLU（约 57.5%）**，其次 attention 投影（约 28.6%）；SDPA 的 \(T^{2}\) 项与 `lm_head` 更小。

### (d)

\(T{=}1024\) 时各部件占比（占前向 matmul 总量）：

| 模型 | 总 FLOPs | attn 投影 | SDPA \(T^{2}\) | FFN | lm_head |
|------|----------|-----------|----------------|-----|---------|
| small（12L, 768d, 12H） | \(2.92\times 10^{11}\) | 19.9% | 13.3% | 39.8% | 27.1% |
| medium（24L, 1024d, 16H） | \(8.30\times 10^{11}\) | 24.8% | 12.4% | 50.1% | 12.7% |
| large（36L, 1280d, 20H） | \(1.77\times 10^{12}\) | 27.3% | 10.9% | 54.3% | 7.5% |
| XL | \(3.52\times 10^{12}\) | 28.6% | 9.2% | 57.5% | 4.7% |

模型变大时：**FFN / attn 投影占比上升**，**`lm_head`（\(\propto VD\)）占比下降**；\(T\) 固定时 SDPA 占比略降。

### (e)

XL 将 \(T\) 提到 16,384：总 FLOPs 约 **\(1.336\times 10^{14}\)**（约为 \(T{=}1024\) 时的 **38×**）。SDPA 的 \(T^{2}\) 项升至约 **62%**，投影与 FFN（\(\propto T\)）占比下降；长上下文下 attention score/value 乘法成为主导。
