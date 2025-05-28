import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import font_manager

from sklearn.manifold import TSNE

font_path = '/share/nas169/yuchunliu/miniconda3/envs/espnet_context1/lib/python3.10/site-packages/matplotlib/mpl-data/fonts/ttf/MicrosoftJhengHei.ttf'

if os.path.isfile(font_path):
    print(f"Font file found at {font_path}")
else:
    print(f"Font file not found at {font_path}")

font_manager.fontManager.addfont(font_path)
prop = font_manager.FontProperties(fname=font_path)
plt.rcParams['font.family'] = prop.get_name()
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei']
plt.rcParams['axes.unicode_minus'] = False

def plot_attention_map(
    frame2align,
    attention,
    text,
    labels,
    debug_path,
    uttid='test',
    w_t=None,
):
    attention = torch.flip(attention, [0, 2])
    attention = attention.squeeze(0).T.detach().cpu().resolve_conj().resolve_neg().numpy()
    xlabels   = [
        f'{frame2align[i]} {i}' if i in frame2align else f'{i}' for i in range(attention.shape[1])
    ]

    labels = [f'{labels[len(labels) - i - 1]}' for i in range(len(labels))]
    # plt.rcParams.update({'font.size': 8})
    plt.rcParams.update({'font.size': 30})

    # draw attention map
    # fig, axes = plt.subplots(1, 1, figsize=(45, 10))
    # fig, axes = plt.subplots(1, 1, figsize=(45, 30))
    fig, axes = plt.subplots(2, 1, figsize=(60, 40), gridspec_kw={'height_ratios': [4, 1]})
    attention_axes, wt_axes = axes
    attention_axes.xaxis.set_ticks(np.arange(0, attention.shape[1], 1))
    attention_axes.yaxis.set_ticks(np.arange(0, attention.shape[0], 1))
    attention_axes.set_xticks(np.arange(-.5, attention.shape[1], 10), minor=True)
    attention_axes.set_yticks(np.arange(-.5, attention.shape[0], 1), minor=True)
    attention_axes.set_xticklabels(xlabels, rotation=90)
    attention_axes.set_yticklabels(labels)

    attention_axes.imshow(attention, aspect='auto')
    attention_axes.grid(which='minor', color='w', linewidth=0.5, alpha=0.3)
    attention_axes.set_title(text)
    
    if w_t is not None:
        w_t = w_t.squeeze(0).cpu().numpy()
        wt_axes.plot(w_t, marker='o', color='b')
        wt_axes.set_xlim(0, len(w_t) - 1)
        wt_axes.set_ylim(0, 1)
        wt_axes.set_xticks(np.arange(0, len(w_t), 10))
        wt_axes.set_yticks(np.arange(0, 1.1, 0.1))
        wt_axes.set_xlabel("Frame")
        wt_axes.set_ylabel("w_t Value")
    else:
        wt_axes.plot(w_t, marker='o', color='b')
        wt_axes.set_xlim(0, len(w_t) - 1)
        wt_axes.set_ylim(0, 1)
        wt_axes.set_xticks(np.arange(0, len(w_t), 10))
        wt_axes.set_yticks(np.arange(0, 1.1, 0.1))
        wt_axes.set_xlabel("Frame")
        wt_axes.set_ylabel("w_t Value")
    output_path = os.path.join(debug_path, f'{uttid}_attention_map.pdf')
    plt.savefig(output_path, format="pdf", bbox_inches="tight")
    plt.clf()

def plot_tsne(
        X,
        label,
        debug_path,
        uttid='test',
    ):
    # tsne = TSNE(n_components=2, verbose=1, perplexity=5)
    tsne = TSNE(n_components=2, verbose=1)
    X    = tsne.fit_transform(X.detach())

    fig, axes = plt.subplots(1, 1, figsize=(20, 20))
    plt.scatter(x=X[:, 0], y=X[:, 1], s=10)

    if label is not None:
        texts = []
        # plt.rcParams.update({'font.size': 8})
        plt.rcParams.update({'font.size': 12})
        for i in range(X.shape[0]):
            x, y = X[i, 0], X[i, 1]
            texts.append(plt.text(x, y, label[i]))

    output_path = os.path.join(debug_path, f'{uttid}_tsne.pdf')
    plt.savefig(output_path, format="pdf", bbox_inches="tight")
    plt.clf()

def plot_gate(
    gate_prob,
    debug_path,
    uttid='test',
):
    plt.plot(gate_prob)
    output_path = os.path.join(debug_path, f'{uttid}_gate.pdf')
    plt.savefig(output_path, format="pdf", bbox_inches="tight")
    plt.clf()