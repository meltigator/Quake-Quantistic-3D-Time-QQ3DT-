# notebook_1_alpha_comparison.ipynb

import numpy as np
import matplotlib.pyplot as plt
from predictive_feedback import TemporalFeedbackSystem, harmonic_oscillator, noisy_predictor
from scipy.signal import savgol_filter

def run_experiment(alpha, steps=300, dt=0.05, seed=42):
    """Esegue un esperimento con alpha specificato"""
    np.random.seed(seed)
    system = TemporalFeedbackSystem(
        initial_state=[1.0, 0.0],
        system_model=harmonic_oscillator,
        predictor=noisy_predictor,
        alpha=alpha,
        noise_std=0.03
    )
    results = system.run(steps=steps, dt=dt)
    return results

# Configurazione
plt.figure(figsize=(15, 10))

# 1. Evoluzione temporale - Confronto diretto
alphas = [0.0, 0.3, 0.6, 0.9]
colors = ['blue', 'green', 'orange', 'red']

plt.subplot(2, 2, 1)
for alpha, color in zip(alphas, colors):
    results = run_experiment(alpha, steps=200)
    positions = results['history'][:, 0]
    time = np.arange(len(positions)) * 0.05
    plt.plot(time, positions, label=f'α={alpha}', color=color, alpha=0.8)
plt.xlabel('Tempo')
plt.ylabel('Posizione (x)')
plt.title('Evoluzione della Posizione - Diversi α')
plt.legend()
plt.grid(True, alpha=0.3)

# 2. Stabilità nel tempo
plt.subplot(2, 2, 2)
for alpha, color in zip(alphas, colors):
    results = run_experiment(alpha, steps=200)
    stability = results['stability']
    smoothed = savgol_filter(stability, 21, 3)  # Smooth per visualizzazione
    time = np.arange(len(stability)) * 0.05
    plt.plot(time, smoothed, label=f'α={alpha}', color=color, alpha=0.8)
plt.xlabel('Tempo')
plt.ylabel('Stabilità (1/Δ)')
plt.title('Evoluzione della Stabilità')
plt.legend()
plt.grid(True, alpha=0.3)

# 3. Piano delle fasi
plt.subplot(2, 2, 3)
for alpha, color in zip(alphas, colors):
    results = run_experiment(alpha, steps=400)
    x = results['history'][:, 0]
    v = results['history'][:, 1]
    plt.plot(x, v, label=f'α={alpha}', color=color, alpha=0.6, linewidth=1)
plt.xlabel('Posizione (x)')
plt.ylabel('Velocità (v)')
plt.title('Piano delle Fasi - Attrattori')
plt.legend()
plt.grid(True, alpha=0.3)
plt.axis('equal')

# 4. Energia del sistema (E = ½kx² + ½mv²)
plt.subplot(2, 2, 4)
time_energy = np.arange(200) * 0.05
for alpha, color in zip(alphas, colors):
    results = run_experiment(alpha, steps=200)
    x = results['history'][:, 0]
    v = results['history'][:, 1]
    energy = 0.5 * x**2 + 0.5 * v**2  # k=1, m=1
    plt.plot(time_energy, energy, label=f'α={alpha}', color=color, alpha=0.8)
plt.xlabel('Tempo')
plt.ylabel('Energia totale')
plt.title('Dissipazione Energetica')
plt.legend()
plt.grid(True, alpha=0.3)
plt.yscale('log')

plt.tight_layout()
plt.show()

# Analisi quantitativa
print("ANALISI COMPARATIVA")
print("-" * 40)
for alpha in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
    results = run_experiment(alpha, steps=300)
    
    # Tempo per raggiungere stabilità (varianza < soglia)
    variance = np.var(results['history'][-50:, 0])  # Varianza ultimi 50 punti
    stable_time = np.argmax(np.abs(results['history'][:, 0]) < 0.01) * 0.05
    
    # Energia residua
    final_energy = 0.5 * results['history'][-1, 0]**2 + 0.5 * results['history'][-1, 1]**2
    
    print(f"α={alpha:.1f}: "
          f"Varianza={variance:.6f}, "
          f"Tempo_stabile={stable_time:.2f}s, "
          f"Energia_finale={final_energy:.6f}")