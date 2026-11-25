# -*- coding: utf-8 -*-
"""
Transformer Attention 教学演示

用法：
  python3 transformer_demo.py                    # 训练模式：训练并保存权重
  python3 transformer_demo.py X can              # 推理模式：加载权重并预测

数据文件：./tr_data（词汇表和训练语料）
权重文件：./model_weights（训练后的模型参数）
"""

import numpy as np
import sys
import os

np.set_printoptions(precision=4, suppress=True, linewidth=120)

# ================================================================ 0. 配置
DATA_FILE = "./tr_data"
WEIGHTS_FILE = "./model_weights"

# 模型维度配置
D = 5      # 嵌入维度
DK = 5     # QKV 投影维度
D_FF = 10  # FFN 中间层维度
MAX_LEN = 4
MAX_LEN = 4

# ================================================================ 1. 工具函数
def softmax(z, axis=-1):
    e = np.exp(z - z.max(axis=axis, keepdims=True)) 
    return e / e.sum(axis=axis, keepdims=True)

def positional_encoding(max_len, d):
    P = np.zeros((max_len, d))
    for pos in range(max_len):
        for i in range(0, d, 2):
            denom = 10000 ** (i / d)
            P[pos, i] = np.sin(pos / denom)
            if i + 1 < d:
                P[pos, i + 1] = np.cos(pos / denom)
    return P


# ================================================================ 2. 加载数据
def load_data():
    """从 tr_data 文件加载词汇表和训练语料"""
    if not os.path.exists(DATA_FILE):
        print(f"错误：找不到数据文件 {DATA_FILE}")
        print("请先创建该文件，格式：")
        print("""
VOCAB = ["I", "love", "you", "sky", "dog", "sea", "beautiful", "big"]
TRAINING_DATA = [
    (["big", "dog", "love"], "you"),
    (["I", "love"], "you"),
]
""")
        sys.exit(1)

    with open(DATA_FILE, 'r', encoding='utf-8') as f:
        content = f.read()

    local_vars = {}
    exec(content, {}, local_vars)

    VOCAB = local_vars['VOCAB']
    TRAINING_DATA_RAW = local_vars['TRAINING_DATA']

    ID = {w: i for i, w in enumerate(VOCAB)}

    TRAINING_DATA = []
    for input_words, target_word in TRAINING_DATA_RAW:
        input_ids = [ID[w] for w in input_words]
        target_id = ID[target_word]
        TRAINING_DATA.append((input_ids, target_id, input_words, target_word))

    return VOCAB, ID, TRAINING_DATA


VOCAB, ID, TRAINING_DATA = load_data()
VOCAB_SIZE = len(VOCAB)
SCALE = np.sqrt(DK)
PE = positional_encoding(MAX_LEN, D)


# ================================================================ 3. 参数打包/解包
NUM_LAYERS = 1  # 单层 Transformer

def get_param_shapes():
    SHAPES = {
        "E":  (VOCAB_SIZE, D),
        "Wq": (D, DK),
        "Wk": (D, DK),
        "Wv": (D, DK),
        "W1": (D, D_FF),    # FFN 第一层：升维
        "W2": (D_FF, D),    # FFN 第二层：降维
        "WU": (VOCAB_SIZE, D),
    }
    KEYS = list(SHAPES)
    SIZES = [int(np.prod(SHAPES[k])) for k in KEYS]
    return SHAPES, KEYS, SIZES


SHAPES, KEYS, SIZES = get_param_shapes()
NPARAM = sum(SIZES)


def unpack(theta):
    d, i = {}, 0
    for k, n in zip(KEYS, SIZES):
        d[k] = theta[i:i + n].reshape(SHAPES[k])
        i += n
    return d


def pack(params):
    """将参数字典打包成一维向量"""
    return np.concatenate([params[k].flatten() for k in KEYS])


# ================================================================ 4. 保存/加载权重
def save_weights(theta, filename=WEIGHTS_FILE):
    """保存权重到文件"""
    np.save(filename, theta)
    print(f"\n权重已保存到: {filename}")


def load_weights(filename=WEIGHTS_FILE):
    """从文件加载权重"""
    if not os.path.exists(filename + ".npy"):
        print(f"错误：找不到权重文件 {filename}.npy")
        print("请先运行训练模式：python3 transformer_demo.py")
        sys.exit(1)

    theta = np.load(filename + ".npy")
    print(f"权重已从 {filename}.npy 加载")
    return theta


# ================================================================ 5. 前向传播（单层 Transformer）
def forward(theta, ids, target, need_cache=True, verbose=False):
    p = unpack(theta)
    seq_len = len(ids)

    if verbose:
        print(f"\n  [前向] 输入: {[VOCAB[i] for i in ids]} (ids={ids})")

    # 初始嵌入
    X = p["E"][list(ids)] + PE[:seq_len]
    if verbose:
        print(f"  [Step 1] X = E[ids] + PE:\n{X}")

    # 因果掩码
    causal_mask = np.triu(np.ones((seq_len, seq_len)) * -np.inf, k=1)

    # Attention
    Q = X @ p["Wq"]
    K = X @ p["Wk"]
    V = X @ p["Wv"]

    S = Q @ K.T / SCALE + causal_mask
    if verbose:
        print(f"  [Step 2] S = Q @ K^T / sqrt({DK}) + mask:\n{S}")

    A = softmax(S, axis=1)
    if verbose:
        print(f"  [Step 3] A = softmax(S):\n{A}")

    O = A @ V
    if verbose:
        print(f"  [Step 4] O = A @ V:\n{O}")

    # FFN: 升维 -> ReLU -> 降维
    hidden = O @ p["W1"]
    hidden = np.maximum(0, hidden)  # ReLU
    O = hidden @ p["W2"]
    if verbose:
        print(f"  [Step 5] FFN: hidden = ReLU(O @ W1) @ W2:\n{O}")

    # 取最后一个位置
    h = O[-1]
    if verbose:
        print(f"  [Step 6] h = O[-1]:\n{h}")

    logits = h @ p["WU"].T
    if verbose:
        print(f"  [Step 7] logits = h @ WU^T:\n{logits}")

    prob = softmax(logits)
    if verbose:
        print(f"  [Output] prob = softmax(logits):")
        for i, w in enumerate(VOCAB):
            print(f"           {w:10s}: {prob[i]:.4f}")

    loss = -np.log(prob[target] + 1e-12)
    if verbose:
        print(f"  [损失] -log(prob['{VOCAB[target]}']) = {loss:.4f}")

    cache = {
        "X": X, "h": h,
        "logits": logits, "prob": prob, "params": p
    }
    return loss, cache


# ================================================================ 6. 损失函数
LAMBDA = 0.001


def compute_loss_single(theta, ids, target):
    loss, _ = forward(theta, ids, target, need_cache=False)
    return loss


def compute_loss_all(theta):
    losses = []
    for ids, target, _, _ in TRAINING_DATA:
        loss = compute_loss_single(theta, ids, target)
        losses.append(loss)
    return float(np.mean(losses)), losses


def total_loss(theta):
    avg_loss, _ = compute_loss_all(theta)
    return avg_loss + LAMBDA * float(theta @ theta)


def numerical_grad(theta, eps=1e-5):
    g = np.empty_like(theta)
    for i in range(len(theta)):
        a = theta.copy(); a[i] += eps
        b = theta.copy(); b[i] -= eps
        g[i] = (total_loss(a) - total_loss(b)) / (2 * eps)
    return g


# ================================================================ 7. 推理函数
def predict_next(input_words, theta, verbose=True):
    """给定输入词，预测下一个词"""
    input_ids = [ID[w] for w in input_words]

    if verbose:
        print(f"\n{'='*80}")
        print(f"【推理】输入: {' '.join(input_words)}")
        print(f"{'='*80}")

    _, cache = forward(theta, input_ids, 0, need_cache=True, verbose=verbose)

    pred_idx = int(cache['prob'].argmax())
    pred_word = VOCAB[pred_idx]
    pred_prob = cache['prob'][pred_idx]

    if verbose:
        print(f"\n预测结果: {pred_word} (概率: {pred_prob:.4f})")
        print("\n概率分布:")
        for i, w in enumerate(VOCAB):
            bar = "█" * int(cache["prob"][i] * 50)
            print(f"  {w:10s}: {cache['prob'][i]:.4f} {bar}")

    return pred_word, pred_prob, cache


# ================================================================ 8. 训练模式
def train_mode():
    print("=" * 80)
    print("【训练模式】")
    print("=" * 80)
    print(f"  数据文件: {DATA_FILE}")
    print(f"  权重文件: {WEIGHTS_FILE}.npy")
    print(f"  词表: {VOCAB}")
    print(f"  嵌入维度: {D}, QKV维度: {DK}")

    print("\n【训练语料】")
    for i, (ids, target, words, target_word) in enumerate(TRAINING_DATA):
        print(f"  {i+1}. {' '.join(words)} -> {target_word}")

    # 初始化参数
    rng = np.random.default_rng(42)
    theta = rng.normal(0.0, 0.3, NPARAM)

    print(f"\n【初始权重】共 {NPARAM} 个参数")
    p0 = unpack(theta)
    for k in KEYS:
        print(f"\n{k} {SHAPES[k]}:")
        print(p0[k])

    # 训练前的预测
    print("\n【训练前】每条语料的预测")
    avg_loss, losses = compute_loss_all(theta)
    print(f"平均损失: {avg_loss:.4f}\n")

    for i, (ids, target, words, target_word) in enumerate(TRAINING_DATA):
        loss, cache = forward(theta, ids, target, need_cache=True)
        pred_word = VOCAB[int(cache['prob'].argmax())]
        target_prob = cache['prob'][target]
        print(f"  {i+1}. {' '.join(words):20s} -> {target_word:10s} | P={target_prob:.4f} | pred={pred_word}")

    # 训练过程
    print("\n" + "=" * 80)
    print("【训练过程】")

    STEPS = 200
    LR = 0.05

    m = np.zeros_like(theta)
    v = np.zeros_like(theta)

    for step in range(1, STEPS + 1):
        old_theta = theta.copy()

        g = numerical_grad(theta)
        g_p = unpack(g)

        m = 0.9 * m + 0.1 * g
        v = 0.999 * v + 0.001 * g * g
        mh = m / (1 - 0.9 ** step)
        vh = v / (1 - 0.999 ** step)

        update = LR * mh / (np.sqrt(vh) + 1e-8)
        theta -= update

        new_avg_loss, new_losses = compute_loss_all(theta)
        old_avg_loss, old_losses = compute_loss_all(old_theta)

        # 每条语料的预测
        preds = []
        for ids, target, words, target_word in TRAINING_DATA:
            _, cache = forward(theta, ids, target, need_cache=True)
            pred_word = VOCAB[int(cache['prob'].argmax())]
            target_prob = cache['prob'][target]
            preds.append((pred_word, target_prob))

        print(f"\nStep {step:3d} | 平均Loss: {old_avg_loss:.4f} → {new_avg_loss:.4f} ({new_avg_loss-old_avg_loss:+.4f})")

        for i, (ids, target, words, target_word) in enumerate(TRAINING_DATA):
            pred_word, target_prob = preds[i]
            loss_change = new_losses[i] - old_losses[i]
            status = "✓" if pred_word == target_word else "✗"
            print(f"  {i+1}. {' '.join(words):20s} -> {target_word:5s} | P={target_prob:.4f} | pred={pred_word:5s} {status} | loss={new_losses[i]:.4f}({loss_change:+.4f})")

        # 每隔一段时间打印详细的前向传播过程和梯度
        if step <= 3 or step % 20 == 0:
            # 打印第一个样本的详细前向传播
            sample_idx = 0
            ids, target, words, target_word = TRAINING_DATA[sample_idx]
            print(f"\n  === 详细前向传播: {' '.join(words)} -> {target_word} ===")
            _, cache = forward(theta, ids, target, need_cache=True, verbose=True)

            print(f"\n  梯度详情:")
            for k in ['E', 'Wq', 'Wk', 'Wv', 'W1', 'W2', 'WU']:
                g_norm = np.linalg.norm(g_p[k])
                print(f"    {k}: 范数={g_norm:.4f}")

        # 收敛检查：预测正确且概率都大于 0.9
        all_correct = all(
            pred[0] == VOCAB[target] and pred[1] > 0.9
            for (ids, target, _, _), pred in zip(TRAINING_DATA, preds)
        )
        if all_correct and step > 50:
            print(f"\n  *** 所有语料预测正确且概率 > 0.9，提前停止 ***")
            break

    # 训练后的权重
    print("\n" + "=" * 80)
    print("【训练后】修正后的权重")

    p = unpack(theta)
    for k in KEYS:
        print(f"\n{k} {SHAPES[k]}:")
        print(p[k])

    # 保存权重
    save_weights(theta)

    print("\n" + "=" * 80)
    print("训练完成！")
    print("现在可以进行推理：python3 transformer_demo.py <token1> <token2> ...")
    print("例如：python3 transformer_demo.py X can")


# ================================================================ 9. 推理模式
def inference_mode(input_words):
    print("=" * 80)
    print("【推理模式】")
    print("=" * 80)
    print(f"  数据文件: {DATA_FILE}")
    print(f"  权重文件: {WEIGHTS_FILE}.npy")
    print(f"  词表: {VOCAB}")

    # 检查词汇
    invalid = [w for w in input_words if w not in ID]
    if invalid:
        print(f"\n错误：以下词汇不在词表中: {invalid}")
        print(f"可用词汇: {VOCAB}")
        sys.exit(1)

    # 加载权重
    theta = load_weights()

    # 显示加载的权重
    p = unpack(theta)
    print("\n【加载的权重】")
    for k in KEYS:
        print(f"\n{k} {SHAPES[k]}:")
        print(p[k])

    # 进行推理
    predict_next(input_words, theta, verbose=True)


# ================================================================ 10. 主程序
if __name__ == "__main__":
    if len(sys.argv) > 1:
        # 有参数：推理模式
        inference_mode(sys.argv[1:])
    else:
        # 无参数：训练模式
        train_mode()
