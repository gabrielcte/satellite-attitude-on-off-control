
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import find_peaks

# Garante que a pasta de saída exista
os.makedirs("docs", exist_ok=True)


def pole_plot(poles, save_path="docs/poles.png"):
    fig, ax = plt.subplots(figsize=(6, 5))

    ax.scatter(np.real(poles), np.imag(poles), marker="x")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Parte real")
    ax.set_ylabel("Parte imaginária")
    ax.set_title("Polos do modelo linear em malha aberta")
    ax.grid(True)
    fig.tight_layout()

    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"Gráfico salvo em '{save_path}'")

    plt.show()
    plt.close(fig)


def time_plots(
    time_hist,
    thet_rk_hist,
    thet_s_rk_hist,
    theta_c,
    thet_d_rk_hist,
    u_rk_hist,
    f_rk_hist,
    save_path="docs/satellite_simulation_results.png",
):
    fig, axes = plt.subplots(4, 1, figsize=(12, 10))

    # Ângulo
    axes[0].plot(time_hist, thet_rk_hist, color="red", label="theta RK4")
    axes[0].plot(
        time_hist,
        thet_s_rk_hist,
        "--",
        color="red",
        label="theta sensor RK4",
    )
    axes[0].axhline(
        theta_c, linestyle=":", color="black", label="theta_c"
    )
    axes[0].set_ylabel("Angle (rad)")
    axes[0].grid(True)
    axes[0].legend()

    # Velocidade angular
    axes[1].plot(time_hist, thet_d_rk_hist, color="red", label="RK4")
    axes[1].set_ylabel("Rate (rad/s)")
    axes[1].grid(True)
    axes[1].legend()

    # Controle
    axes[2].plot(time_hist, u_rk_hist, color="red", label="RK4")
    axes[2].set_ylabel("Control")
    axes[2].grid(True)
    axes[2].legend()

    # Torque
    axes[3].plot(time_hist, f_rk_hist, color="red", label="RK4")
    axes[3].set_ylabel("Torque (N.m)")
    axes[3].set_xlabel("Time (s)")
    axes[3].grid(True)
    axes[3].legend()

    fig.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"Gráfico salvo em '{save_path}'")

    plt.show()
    plt.close(fig)

def plot_phase_flow_satellite(
    theta_hist,
    rate_hist,
    theta_c,
    save_path="docs/phase_flow.png",
    prominence=None,
):
    theta_hist = np.asarray(theta_hist)
    rate_hist = np.asarray(rate_hist)

    if len(theta_hist) != len(rate_hist):
        raise ValueError(
            "theta_hist e rate_hist devem ter o mesmo tamanho."
        )

    if len(theta_hist) == 0:
        raise ValueError("As séries históricas não podem estar vazias.")

    if (
        not np.all(np.isfinite(theta_hist))
        or not np.all(np.isfinite(rate_hist))
    ):
        raise ValueError("Os dados devem conter apenas valores finitos.")

    fig, ax = plt.subplots(figsize=(10, 7))

    # Trajetória simulada
    ax.plot(
        theta_hist,
        rate_hist,
        color="tab:blue",
        linewidth=1.5,
        label="Trajetória simulada",
    )

    # Condições inicial e final
    ax.plot(
        theta_hist[0],
        rate_hist[0],
        "s",
        color="tab:green",
        markersize=7,
        label="Início",
    )

    ax.plot(
        theta_hist[-1],
        rate_hist[-1],
        "X",
        color="tab:red",
        markersize=8,
        label="Fim",
    )

    # Identificação dos extremos locais
    peaks, _ = find_peaks(
        rate_hist,
        prominence=prominence,
    )

    valleys, _ = find_peaks(
        -rate_hist,
        prominence=prominence,
    )

    ax.scatter(
        theta_hist[peaks],
        rate_hist[peaks],
        color="darkorange",
        marker="^",
        s=65,
        zorder=5,
        label="Máximos locais",
    )

    ax.scatter(
        theta_hist[valleys],
        rate_hist[valleys],
        color="purple",
        marker="v",
        s=65,
        zorder=5,
        label="Mínimos locais",
    )

    # Extremos globais
    idx_max = np.argmax(rate_hist)
    idx_min = np.argmin(rate_hist)

    ax.scatter(
        theta_hist[idx_max],
        rate_hist[idx_max],
        color="red",
        marker="*",
        s=200,
        zorder=8,
        label="Máximo global",
    )

    ax.scatter(
        theta_hist[idx_min],
        rate_hist[idx_min],
        color="black",
        marker="*",
        s=200,
        zorder=8,
        label="Mínimo global",
    )

    # Anotações dos extremos globais
    ax.annotate(
        f"Máx. global\nθ={theta_hist[idx_max]:.3f}\n"
        f"θ̇={rate_hist[idx_max]:.3f}",
        (theta_hist[idx_max], rate_hist[idx_max]),
        xytext=(10, 12),
        textcoords="offset points",
        fontsize=9,
        color="red",
    )

    ax.annotate(
        f"Mín. global\nθ={theta_hist[idx_min]:.3f}\n"
        f"θ̇={rate_hist[idx_min]:.3f}",
        (theta_hist[idx_min], rate_hist[idx_min]),
        xytext=(10, -35),
        textcoords="offset points",
        fontsize=9,
        color="black",
    )

    # Seleciona extremos locais distintos dos globais
    peaks_valid = peaks[peaks != idx_max]
    valleys_valid = valleys[valleys != idx_min]

    # Intervalo horizontal para prolongar as retas
    theta_min_plot = np.min(theta_hist)
    theta_max_plot = np.max(theta_hist)

    theta_range = theta_max_plot - theta_min_plot

    if np.isclose(theta_range, 0):
        theta_line = None
    else:
        margin = 0.02 * theta_range

        theta_line = np.linspace(
            theta_min_plot - margin,
            theta_max_plot + margin,
            500,
        )

    # Reta prolongada: mínimo local -> mínimo global
    if len(valleys_valid) > 0 and theta_line is not None:
        idx_valley = valleys_valid[
            np.argmin(
                np.abs(
                    rate_hist[valleys_valid] - rate_hist[idx_min]
                )
            )
        ]

        x1 = theta_hist[idx_valley]
        y1 = rate_hist[idx_valley]
        x2 = theta_hist[idx_min]
        y2 = rate_hist[idx_min]

        if not np.isclose(x2, x1):
            slope = (y2 - y1) / (x2 - x1)
            rate_line = y1 + slope * (theta_line - x1)

            ax.plot(
                theta_line,
                rate_line,
                color="magenta",
                linestyle="--",
                linewidth=2,
                zorder=4,
                label="Reta propagada: mínimo",
            )

    # Reta prolongada: máximo local -> máximo global
    if len(peaks_valid) > 0 and theta_line is not None:
        idx_peak = peaks_valid[
            np.argmin(
                np.abs(
                    rate_hist[peaks_valid] - rate_hist[idx_max]
                )
            )
        ]

        x1 = theta_hist[idx_peak]
        y1 = rate_hist[idx_peak]
        x2 = theta_hist[idx_max]
        y2 = rate_hist[idx_max]

        if not np.isclose(x2, x1):
            slope = (y2 - y1) / (x2 - x1)
            rate_line = y1 + slope * (theta_line - x1)

            ax.plot(
                theta_line,
                rate_line,
                color="cyan",
                linestyle="--",
                linewidth=2,
                zorder=4,
                label="Reta propagada: máximo",
            )

    # Ângulo de referência
    ax.axvline(
        theta_c,
        color="black",
        linestyle=":",
        linewidth=1.2,
        label=r"$\theta_c$",
    )

    # Formatação
    ax.set_xlabel(r"Ângulo $\theta$ (rad)")
    ax.set_ylabel(r"Velocidade angular $\dot{\theta}$ (rad/s)")
    ax.set_title(
        "Trajetória no espaço de fase e retas prolongadas"
    )
    ax.grid(True)
    ax.legend(fontsize=9)
    fig.tight_layout()

    # Salva o gráfico
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    fig.savefig(
        save_path,
        dpi=300,
        bbox_inches="tight",
    )

    print(f"Gráfico salvo em '{save_path}'")

    plt.show()
    plt.close(fig)
