# notebook_2_stability_vs_alpha.ipynb

import numpy as np
import matplotlib.pyplot as plt
from predictive_feedback import TemporalFeedbackSystem, harmonic_oscillator, noisy_predictor
from tqdm import tqdm

def compute_stability_metrics(alpha, trials=20, steps=200):
    """Calculate stability metrics for a given alpha (averaged over multiple trials)"""
    stabilities = []
    convergence_times = []
    final_energies = []
    
    for trial in range(trials):
        system = TemporalFeedbackSystem(
            initial_state=[1.0, 0.0],
            system_model=harmonic_oscillator,
            predictor=noisy_predictor,
            alpha=alpha,
            noise_std=0.02
        )
        results = system.run(steps=steps, dt=0.05)
        
        # Stabilità media (escludi transiente iniziale)
        stabilities.append(np.mean(results['stability'][50:]))
        
        # Tempo di convergenza (quando |x| < 0.01)
        x_values = results['history'][:, 0]
        converged = np.where(np.abs(x_values) < 0.01)[0]
        if len(converged) > 0:
            convergence_times.append(converged[0] * 0.05)
        else:
            convergence_times.append(steps * 0.05)
        
        # Energia finale
        final_energies.append(
            0.5 * results['history'][-1, 0]**2 + 0.5 * results['history'][-1, 1]**2
        )
    
    return {
        'mean_stability': np.mean(stabilities),
        'std_stability': np.std(stabilities),
        'mean_convergence': np.mean(convergence_times),
        'mean_energy': np.mean(final_energies)
    }

# Scansione di alpha
alphas = np.linspace(0, 1, 21)
metrics = []

print("Scansione in corso...")
for alpha in tqdm(alphas):
    metrics.append(compute_stability_metrics(alpha, trials=15))

# Estrai dati
mean_stab = [m['mean_stability'] for m in metrics]
std_stab = [m['std_stability'] for m in metrics]
conv_times = [m['mean_convergence'] for m in metrics]
energies = [m['mean_energy'] for m in metrics]

# Visualizzazione
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 1. Stabilità media ± deviazione standard
ax1 = axes[0, 0]
ax1.errorbar(alphas, mean_stab, yerr=std_stab, 
             fmt='o-', capsize=5, capthick=2, 
             color='darkblue', ecolor='lightblue', linewidth=2)
ax1.set_xlabel('Coefficiente di Feedback (α)')
ax1.set_ylabel('Stabilità Media')
ax1.set_title('Stabilità vs Feedback Strength')
ax1.grid(True, alpha=0.3)
ax1.fill_between(alphas, 
                 np.array(mean_stab) - np.array(std_stab),
                 np.array(mean_stab) + np.array(std_stab),
                 alpha=0.2, color='blue')

# 2. Tempo di convergenza
ax2 = axes[0, 1]
ax2.plot(alphas, conv_times, 's-', color='darkgreen', linewidth=2, markersize=6)
ax2.set_xlabel('Coefficiente di Feedback (α)')
ax2.set_ylabel('Tempo di Convergenza (s)')
ax2.set_title('Velocità di Stabilizzazione')
ax2.grid(True, alpha=0.3)
ax2.invert_yaxis()  # Tempo minore = migliore

# 3. Energia residua
ax3 = axes[1, 0]
ax3.semilogy(alphas, energies, '^-', color='darkred', linewidth=2, markersize=8)
ax3.set_xlabel('Coefficiente di Feedback (α)')
ax3.set_ylabel('Energia Finale (log)')
ax3.set_title('Energia Residua vs α')
ax3.grid(True, alpha=0.3)

# 4. Diagramma di efficienza (stabilità vs convergenza)
ax4 = axes[1, 1]
scatter = ax4.scatter(conv_times, mean_stab, c=alphas, 
                      cmap='viridis', s=100, alpha=0.8)
ax4.set_xlabel('Tempo di Convergenza (s)')
ax4.set_ylabel('Stabilità Media')
ax4.set_title('Trade-off: Velocità vs Stabilità')
ax4.grid(True, alpha=0.3)

# Aggiungi etichette per alcuni punti significativi
for i, alpha in enumerate([0, 0.2, 0.4, 0.6, 0.8, 1.0]):
    idx = np.argmin(np.abs(alphas - alpha))
    ax4.annotate(f'α={alpha}', 
                 (conv_times[idx], mean_stab[idx]),
                 xytext=(10, 10), textcoords='offset points',
                 fontsize=9, bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.7))

# Barra dei colori
plt.colorbar(scatter, ax=ax4, label='Valore di α')

plt.tight_layout()
plt.show()

# Trova ottimi
optimal_stab_idx = np.argmax(mean_stab)
optimal_speed_idx = np.argmin(conv_times)

print("\n" + "="*50)
print("ANALISI OTTIMI")
print("="*50)
print(f"\nMassima stabilità:")
print(f"  α = {alphas[optimal_stab_idx]:.2f}")
print(f"  Stabilità = {mean_stab[optimal_stab_idx]:.3f} ± {std_stab[optimal_stab_idx]:.3f}")
print(f"  Tempo convergenza = {conv_times[optimal_stab_idx]:.2f}s")

print(f"\nMassima velocità:")
print(f"  α = {alphas[optimal_speed_idx]:.2f}")
print(f"  Tempo convergenza = {conv_times[optimal_speed_idx]:.2f}s")
print(f"  Stabilità = {mean_stab[optimal_speed_idx]:.3f} ± {std_stab[optimal_speed_idx]:.3f}")

# Calcola efficienza composita (stabilità/velocità)
efficiency = np.array(mean_stab) / np.array(conv_times)
optimal_eff_idx = np.argmax(efficiency)
print(f"\nMiglior compromesso (stabilità/velocità):")
print(f"  α = {alphas[optimal_eff_idx]:.2f}")
print(f"  Efficienza = {efficiency[optimal_eff_idx]:.3f}")