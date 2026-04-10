# visualize.py
# called in main via main.py --viz
# draws one network panel per generation step in a grid

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec

# colors
BG       = "#1e1e2e"
PANEL    = "#181825"
SURFACE  = "#313244"
OVERLAY  = "#6c7086"
SUBTEXT  = "#a6adc8"
TEXT     = "#cdd6f4"
LAVENDER = "#b4befe"
BLUE     = "#89b4fa"
SAPPHIRE = "#74c7ec"
TEAL     = "#94e2d5"
GREEN    = "#a6e3a1"
YELLOW   = "#f9e2af"
PEACH    = "#fab387"
MAUVE    = "#cba6f7"
RED      = "#f38ba8"
PINK     = "#f5c2e7"

NEURON_COLORS = [BLUE, MAUVE, PEACH, GREEN, SAPPHIRE, TEAL, YELLOW, PINK]

def neuron_color(j):
    return NEURON_COLORS[j % len(NEURON_COLORS)]

plt.rcParams.update({
    "font.family":      "monospace",
    "text.color":       TEXT,
    "axes.facecolor":   PANEL,
    "figure.facecolor": BG,
    "axes.edgecolor":   SURFACE,
    "axes.labelcolor":  SUBTEXT,
    "xtick.color":      OVERLAY,
    "ytick.color":      OVERLAY,
    "grid.color":       SURFACE,
    "grid.linewidth":   0.4,
})

SHOW = 6

def layer_positions(x, n, y_start=0.10, y_end=0.82):
    ys = np.linspace(y_end, y_start, n)
    return [(x, y) for y in ys]


def draw_step(ax, network, vocab_size, idx_to_word,
              inp_indices, pred_idx, step_num, total_steps):
    
    # Draw one generation step onto ax.
    #  inp_indices : list of N word indices (context window)
    #  pred_idx    : index of the word that was sampled/predicted
    
    hidden_size = network.hidden_size

    pos_in  = layer_positions(0.15, SHOW)
    pos_hid = layer_positions(0.50, hidden_size)
    pos_out = layer_positions(0.85, SHOW)

    w1_rows = np.linspace(0, network.w1.shape[0] - 1, SHOW, dtype=int)
    w2_cols = np.linspace(0, network.w2.shape[1] - 1, SHOW, dtype=int)

    ax.clear()
    ax.set_facecolor(PANEL)
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.20, 1.06)
    ax.axis("off")
    for spine in ax.spines.values():
        spine.set_edgecolor(SURFACE)

    # layer header pills
    for lx, label, color in [
        (0.15, "EMB",  BLUE),
        (0.50, "HID", MAUVE),
        (0.85, "OUT", PEACH),
    ]:
        ax.add_patch(mpatches.FancyBboxPatch(
            (lx - 0.09, 0.875), 0.18, 0.055,
            boxstyle="round,pad=0.006",
            facecolor=SURFACE, edgecolor=color,
            linewidth=0.7, zorder=2, clip_on=False
        ))
        ax.text(lx, 0.910, label, ha="center", va="center",
                fontsize=5.5, color=color,
                fontfamily="monospace", fontweight="bold")

    # connections
    w1_max = np.abs(network.w1).max() + 1e-9
    w2_max = np.abs(network.w2).max() + 1e-9

    for i, row_idx in enumerate(w1_rows):
        for j in range(hidden_size):
            w        = network.w1[row_idx, j]
            strength = float(np.clip(abs(w) / w1_max, 0.02, 1.0))
            alpha    = strength * (0.55 if w > 0 else 0.22)
            ax.plot([pos_in[i][0],  pos_hid[j][0]],
                    [pos_in[i][1],  pos_hid[j][1]],
                    color=neuron_color(j), alpha=alpha,
                    linewidth=strength * 1.2,
                    solid_capstyle="round", zorder=1)

    for j in range(hidden_size):
        for k, col_idx in enumerate(w2_cols):
            w        = network.w2[j, col_idx]
            strength = float(np.clip(abs(w) / w2_max, 0.02, 1.0))
            alpha    = strength * (0.55 if w > 0 else 0.22)
            ax.plot([pos_hid[j][0], pos_out[k][0]],
                    [pos_hid[j][1], pos_out[k][1]],
                    color=neuron_color(j), alpha=alpha,
                    linewidth=strength * 1.2,
                    solid_capstyle="round", zorder=1)

    # neurons
    inp_active = int(np.argmin(np.abs(w1_rows - inp_indices[0])))
    out_active = int(np.argmin(np.abs(w2_cols - pred_idx)))

    def draw_layer(positions, base_color, active_idx, colors=None):
        for i, (x, y) in enumerate(positions):
            active = (i == active_idx)
            c      = colors[i] if colors else base_color
            r_out  = 0.020 if active else 0.013
            r_in   = 0.011 if active else 0.007
            ax.add_patch(plt.Circle((x, y), r_out,
                facecolor="none", edgecolor=c,
                linewidth=1.1 if active else 0.5,
                alpha=0.95 if active else 0.45, zorder=4))
            ax.add_patch(plt.Circle((x, y), r_in,
                facecolor=c,
                alpha=1.0 if active else 0.55, zorder=5))

    draw_layer(pos_in,  BLUE,  inp_active)
    draw_layer(pos_hid, MAUVE, -1, colors=[neuron_color(j) for j in range(hidden_size)])
    draw_layer(pos_out, PEACH, out_active)

    # truncation dots
    for lx in [0.15, 0.85]:
        ax.text(lx, 0.045, "· · ·", ha="center", va="center",
                fontsize=6, color=OVERLAY, fontfamily="monospace")

    # step counter
    ax.text(0.50, 0.995, f"step {step_num}/{total_steps}",
            ha="center", va="top", fontsize=5,
            color=OVERLAY, fontfamily="monospace",
            transform=ax.transAxes)

    # context -> prediction label
    probs     = network.forward(inp_indices)
    pred      = int(np.argmax(probs))
    conf      = float(probs[0, pred])
    ctx_str   = " ".join(idx_to_word[i] for i in inp_indices)
    pred_word = idx_to_word[pred_idx]
    col       = GREEN if pred == pred_idx else YELLOW

    ax.text(0.50, -0.04,
            f"[{ctx_str}]",
            ha="center", va="top", fontsize=5,
            color=SUBTEXT, fontfamily="monospace",
            transform=ax.transAxes)
    ax.text(0.50, -0.10,
            f"-> {pred_word}  ({conf:.0%})",
            ha="center", va="top", fontsize=5.5,
            color=col, fontfamily="monospace", fontweight="bold",
            transform=ax.transAxes)


def show_generation(seed_words, generated_words, network,
                    word_to_idx, idx_to_word,
                    loss_history, epoch_history, n, epochs):
    
    # seed_words      : list of N seed word strings  e.g. ["say","wallahi","bro"]
    # generated_words : list of predicted word strings
    # network         : trained net instance (weights already loaded)

    num_steps  = len(generated_words)
    vocab_size = network.vocab_size
    cols       = min(num_steps, 5)
    rows       = (num_steps + cols - 1) // cols

    fig_w = max(16, cols * 3.4)
    fig_h = 3.5 + rows * 3.8
    fig   = plt.figure(figsize=(fig_w, fig_h), facecolor=BG)

    seed_str = " ".join(seed_words)
    fig.text(0.02, 0.988, "ROT'NN SCRATCH",
             fontsize=12, color=TEXT, fontfamily="monospace",
             fontweight="bold", va="top")
    fig.text(0.02, 0.962, f'seed: "{seed_str}"',
             fontsize=8, color=PEACH, fontfamily="monospace", va="top")

    for i, c in enumerate(NEURON_COLORS):
        fig.add_artist(plt.Rectangle(
            (0.02 + i * 0.011, 0.935), 0.009, 0.013,
            facecolor=c, edgecolor="none",
            transform=fig.transFigure
        ))

    fig.add_artist(plt.Line2D(
        [0, 1], [0.927, 0.927],
        color=SURFACE, linewidth=0.8,
        transform=fig.transFigure
    ))

    # loss curve occupies top strip
    loss_height = 2.8 / fig_h 
    ax_loss = fig.add_axes([0.06, 1 - loss_height - 0.01, 0.90, loss_height - 0.08])
    ax_loss.set_facecolor(PANEL)
    for spine in ax_loss.spines.values():
        spine.set_edgecolor(SURFACE)
        spine.set_linewidth(0.7)

    if len(loss_history) > 1:
        x = epoch_history
        y = loss_history
        ax_loss.fill_between(x, y, alpha=0.10, color=LAVENDER)
        ax_loss.plot(x, y, color=LAVENDER, linewidth=1.2, alpha=0.95)
        ax_loss.plot(x[-1], y[-1], "o", color=GREEN, markersize=5, zorder=5)
        yrange = max(y) - min(y) + 1e-9
        ax_loss.text(x[-1], y[-1] + yrange * 0.12,
                     f"{y[-1]:.4f}", fontsize=6,
                     color=GREEN, fontfamily="monospace", ha="right")

    ax_loss.set_title("LOSS CURVE", fontsize=7, color=TEXT,
                      fontfamily="monospace", fontweight="bold",
                      loc="left", pad=4)
    ax_loss.set_xlabel("epoch", fontsize=6, color=SUBTEXT)
    ax_loss.set_ylabel("avg loss", fontsize=6, color=SUBTEXT)
    ax_loss.tick_params(labelsize=5, colors=OVERLAY)
    ax_loss.grid(True, alpha=0.15)

    # panel grid occupies bottom portion
    panel_top    = 0.04 + (rows * 3.8) / fig_h
    panel_bottom = 0.04
    gs = GridSpec(rows, cols, figure=fig,
                  left=0.02, right=0.98,
                  top=panel_top, bottom=panel_bottom,
                  wspace=0.06, hspace=0.55)

    context = list(seed_words)

    for step, pred_word in enumerate(generated_words):
        r  = step // cols
        c  = step  % cols
        ax = fig.add_subplot(gs[r, c])

        ctx_indices = [word_to_idx[w] for w in context[-n:]]
        pred_idx    = word_to_idx[pred_word]

        draw_step(ax, network, vocab_size, idx_to_word,
                  ctx_indices, pred_idx,
                  step_num=step + 1, total_steps=num_steps)

        context.append(pred_word)

    plt.show()