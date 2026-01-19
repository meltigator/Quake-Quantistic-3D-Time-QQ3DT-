# notebook_3_3d_visualization.ipynb

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
from predictive_feedback import TemporalFeedbackSystem, harmonic_oscillator

# Modified version to track the three states separately
class DebugTemporalSystem(TemporalFeedbackSystem):
    """Version with extended logging for 3D visualization"""
    
    def step(self, dt=1.0):
        # Calcola i tre stati separatamente
        natural_state = self.model(self.state, dt)
        
        if self.noise_std > 0:
            natural_state = natural_state + np.random.normal(
                0, self.noise_std, size=self.state.shape
            )
        
        predicted_future = self.predictor(self.state, dt)
        new_state = ((1.0 - self.alpha) * natural_state + 
                     self.alpha * predicted_future)
        
        # Logging esteso
        if not hasattr(self, 'natural_history'):
            self.natural_history = [natural_state.copy()]
            self.predicted_history = [predicted_future.copy()]
        else:
            self.natural_history.append(natural_state.copy())
            self.predicted_history.append(predicted_future.copy())
        
        self.state = new_state
        self.history.append(new_state.copy())
        
        return new_state
    
    def run_debug(self, steps=100, dt=1.0):
        for _ in range(steps):
            self.step(dt)
        
        return {
            'natural': np.array(self.natural_history),
            'predicted': np.array(self.predicted_history),
            'result': np.array(self.history[1:])  # Escludi stato iniziale
        }

# Configura sistema
np.random.seed(42)
system = DebugTemporalSystem(
    initial_state=[1.0, 0.0],
    system_model=harmonic_oscillator,
    predictor=harmonic_oscillator,  # Predictor perfetto per visualizzazione chiara
    alpha=0.4,
    noise_std=0.01
)

# Esegui simulazione
results = system.run_debug(steps=100, dt=0.1)
time = np.arange(len(results['natural'])) * 0.1

# Creazione figura 3D
fig = plt.figure(figsize=(16, 12))

# 1. Visualizzazione 3D completa
ax1 = fig.add_subplot(221, projection='3d')

# Traiettorie nello spazio (x, v, tempo)
ax1.plot(results['natural'][:, 0], results['natural'][:, 1], time,
         label='Naturale', color='blue', alpha=0.7, linewidth=1.5)
ax1.plot(results['predicted'][:, 0], results['predicted'][:, 1], time,
         label='Predetto', color='green', alpha=0.7, linewidth=1.5)
ax1.plot(results['result'][:, 0], results['result'][:, 1], time,
         label='Risultante', color='red', alpha=0.9, linewidth=2)

# Collegamenti tra stati (per alcuni punti)
for i in range(0, len(time), 10):
    # Naturale -> Predetto
    ax1.plot([results['natural'][i, 0], results['predicted'][i, 0]],
             [results['natural'][i, 1], results['predicted'][i, 1]],
             [time[i], time[i]], 'k--', alpha=0.3, linewidth=0.5)
    
    # Predetto -> Risultante
    ax1.plot([results['predicted'][i, 0], results['result'][i, 0]],
             [results['predicted'][i, 1], results['result'][i, 1]],
             [time[i], time[i]], 'k:', alpha=0.3, linewidth=0.5)

ax1.set_xlabel('Posizione (x)')
ax1.set_ylabel('Velocità (v)')
ax1.set_zlabel('Tempo')
ax1.set_title('Spazio delle Fasi-Tempo 3D')
ax1.legend()
ax1.view_init(elev=20, azim=45)

# 2. Proiezioni 2D
ax2 = fig.add_subplot(222)
ax2.plot(results['natural'][:, 0], results['natural'][:, 1], 
         label='Naturale', color='blue', alpha=0.5, linewidth=1)
ax2.plot(results['predicted'][:, 0], results['predicted'][:, 1], 
         label='Predetto', color='green', alpha=0.5, linewidth=1)
ax2.plot(results['result'][:, 0], results['result'][:, 1], 
         label='Risultante', color='red', alpha=0.8, linewidth=2)

# Punti iniziali e finali
ax2.scatter(results['natural'][0, 0], results['natural'][0, 1], 
            s=100, c='blue', marker='o', edgecolors='black', label='Inizio')
ax2.scatter(results['result'][-1, 0], results['result'][-1, 1], 
            s=100, c='red', marker='s', edgecolors='black', label='Fine')

ax2.set_xlabel('Posizione (x)')
ax2.set_ylabel('Velocità (v)')
ax2.set_title('Proiezione Piano delle Fasi')
ax2.legend()
ax2.grid(True, alpha=0.3)
ax2.axis('equal')

# 3. Differenze tra stati
ax3 = fig.add_subplot(223)
diff_nat_pred = np.linalg.norm(results['natural'] - results['predicted'], axis=1)
diff_pred_res = np.linalg.norm(results['predicted'] - results['result'], axis=1)
diff_nat_res = np.linalg.norm(results['natural'] - results['result'], axis=1)

ax3.plot(time, diff_nat_pred, label='Naturale ↔ Predetto', color='orange')
ax3.plot(time, diff_pred_res, label='Predetto ↔ Risultante', color='purple')
ax3.plot(time, diff_nat_res, label='Naturale ↔ Risultante', color='brown', linewidth=2)

ax3.set_xlabel('Tempo')
ax3.set_ylabel('Distanza tra Stati')
ax3.set_title('Evoluzione delle Differenze')
ax3.legend()
ax3.grid(True, alpha=0.3)

# 4. Diagramma vettoriale per un istante specifico
ax4 = fig.add_subplot(224)
instant = 25  # Istante da visualizzare

# Punti
states = {
    'Naturale': results['natural'][instant],
    'Predetto': results['predicted'][instant],
    'Risultante': results['result'][instant]
}

colors = {'Naturale': 'blue', 'Predetto': 'green', 'Risultante': 'red'}

for name, state in states.items():
    ax4.scatter(state[0], state[1], s=200, c=colors[name], 
                label=name, alpha=0.8, edgecolors='black')
    
    # Annotazione
    ax4.annotate(name, (state[0], state[1]), 
                 xytext=(10, 10), textcoords='offset points',
                 fontsize=10, bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8))

# Vettori
ax4.arrow(states['Naturale'][0], states['Naturale'][1],
          states['Predetto'][0] - states['Naturale'][0],
          states['Predetto'][1] - states['Naturale'][1],
          head_width=0.05, head_length=0.1, fc='gray', ec='gray', 
          alpha=0.5, label='Predizione')
ax4.arrow(states['Predetto'][0], states['Predetto'][1],
          states['Risultante'][0] - states['Predetto'][0],
          states['Risultante'][1] - states['Predetto'][1],
          head_width=0.05, head_length=0.1, fc='black', ec='black', 
          alpha=0.7, label='Feedback')

ax4.set_xlabel('Posizione (x)')
ax4.set_ylabel('Velocità (v)')
ax4.set_title(f'Diagramma Vettoriale (t={instant*0.1:.1f}s)')
ax4.legend()
ax4.grid(True, alpha=0.3)
ax4.axis('equal')

plt.tight_layout()
plt.show()

# Animazione (opzionale - salva come GIF)
print("\nAnimation creation...")
fig_anim = plt.figure(figsize=(10, 8))
ax_anim = fig_anim.add_subplot(111, projection='3d')

def update(frame):
    ax_anim.cla()
    # Traccia fino al frame corrente
    ax_anim.plot(results['natural'][:frame, 0], results['natural'][:frame, 1], time[:frame],
                 color='blue', alpha=0.6, linewidth=1, label='Naturale')
    ax_anim.plot(results['predicted'][:frame, 0], results['predicted'][:frame, 1], time[:frame],
                 color='green', alpha=0.6, linewidth=1, label='Predetto')
    ax_anim.plot(results['result'][:frame, 0], results['result'][:frame, 1], time[:frame],
                 color='red', alpha=0.8, linewidth=2, label='Risultante')
    
    # Punto corrente
    ax_anim.scatter(results['natural'][frame, 0], results['natural'][frame, 1], time[frame],
                    s=50, c='blue', edgecolors='black')
    ax_anim.scatter(results['predicted'][frame, 0], results['predicted'][frame, 1], time[frame],
                    s=50, c='green', edgecolors='black')
    ax_anim.scatter(results['result'][frame, 0], results['result'][frame, 1], time[frame],
                    s=80, c='red', edgecolors='black')
    
    ax_anim.set_xlabel('Posizione')
    ax_anim.set_ylabel('Velocità')
    ax_anim.set_zlabel('Tempo')
    ax_anim.set_title(f'QQ3DT - Evoluzione (t={time[frame]:.1f}s)')
    ax_anim.legend()
    ax_anim.view_init(elev=20, azim=frame/2)  # Camera rotante
    
    return ax_anim,

# Crea animazione (ridotta a 30 frame per performance)
anim = FuncAnimation(fig_anim, update, frames=range(0, len(time), 3), 
                     interval=50, blit=False)

# Salva (richiede pillow)
# anim.save('qq3dt_evolution.gif', writer='pillow', fps=20)
print("Animation created. Use anim.save() to export GIF.")