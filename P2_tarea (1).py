"""
Problema 2
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import os

carpeta_descargas = os.path.join(os.path.expanduser("~"), "Downloads")
# Parámetros
ri   = 0.030   # [m] radio interior constante del conducto hueco
ro1  = 0.075   # [m] radio exterior extremo caliente (x=0)
ro2  = 0.035   # [m] radio exterior extremo enfriado (x=L)
L    = 0.150   # [m] longitud axial de la sección bajo estudio

# Condiciones de borde
Th   = 1160.0  # [K] temperatura extremo caliente
Tinf = 300.0   # [K] temperatura del aire ambiente

# Propiedades del aire
# T [K], rho [kg/m3], mu [Pa·s], kf [W/mK], Pr [-]
air_data = np.array([
    #  T      rho       mu          kf        Pr
    [250,   1.3947,  15.62e-6,  22.3e-3,  0.720],
    [300,   1.1614,  18.46e-6,  26.3e-3,  0.707],
    [350,   0.9950,  20.92e-6,  30.0e-3,  0.700],
    [400,   0.8711,  23.01e-6,  33.8e-3,  0.689],
    [450,   0.7740,  25.07e-6,  37.3e-3,  0.683],
    [500,   0.6964,  27.01e-6,  40.7e-3,  0.680],
    [550,   0.6329,  28.84e-6,  43.9e-3,  0.680],
    [600,   0.5804,  30.71e-6,  46.9e-3,  0.680],
    [700,   0.4975,  34.18e-6,  52.4e-3,  0.684],
    [800,   0.4354,  37.73e-6,  57.3e-3,  0.689],
    [900,   0.3868,  40.61e-6,  62.0e-3,  0.696],
    [1000,  0.3482,  44.50e-6,  66.7e-3,  0.700],
    [1100,  0.3166,  48.08e-6,  71.5e-3,  0.704],
    [1200,  0.2902,  51.34e-6,  76.3e-3,  0.707],
])

T_tab  = air_data[:, 0]
rho_tab = air_data[:, 1]
mu_tab  = air_data[:, 2]
kf_tab  = air_data[:, 3]
Pr_tab  = air_data[:, 4]

def get_air_props(Tf):
    """Interpolación lineal de propiedades del aire a temperatura de película Tf [K]."""
    Tf = np.clip(Tf, T_tab[0], T_tab[-1])
    rho = np.interp(Tf, T_tab, rho_tab)
    mu  = np.interp(Tf, T_tab, mu_tab)
    kf  = np.interp(Tf, T_tab, kf_tab)
    Pr  = np.interp(Tf, T_tab, Pr_tab)
    return rho, mu, kf, Pr

def calc_resistances(h, k):
    """Calcula resistencias térmica conductiva y convectiva."""
    # Resistencia conductiva
    num = (ro1 - ri) * (ro2 + ri)
    den = (ro1 + ri) * (ro2 - ri)
    Rcond = L / (2 * np.pi * k * ri * (ro1 - ro2)) * np.log(num / den)
    # Resistencia convectiva
    Rconv = 1.0 / (h * np.pi * (ro2**2 - ri**2))
    Rtot  = Rcond + Rconv
    return Rcond, Rconv, Rtot

def calc_Ts(h, k):
    """Calcula temperatura superficial Ts dado h y k."""
    Rcond, Rconv, Rtot = calc_resistances(h, k)
    Ts = Tinf + (Th - Tinf) / Rtot * Rconv
    return Ts

def calc_h(Ts, U):
    """Calcula coeficiente convectivo h dado Ts y U."""
    Tf = (Ts + Tinf) / 2.0
    rho, mu, kf, Pr = get_air_props(Tf)
    D   = 2 * ro2
    ReD = rho * U * D / mu
    NuD = 0.664 * ReD**0.5 * Pr**(1/3)
    h   = NuD * kf / D
    return h, Tf, ReD, NuD

def iterative_solver(U, k, tol=1e-4, max_iter=100, label=""):
    """
    Resuelve iterativamente el sistema para encontrar Ts y h convergidos.
    Retorna historial de iteraciones y resultados finales.
    """
    # Valor inicial: asumir Ts = promedio entre Th y Tinf
    Ts = Tinf  # arrancar desde T∞ para visualizar mejor la convergencia

    history = []

    for j in range(1, max_iter + 1):
        h, Tf, ReD, NuD = calc_h(Ts, U)
        Rcond, Rconv, Rtot = calc_resistances(h, k)
        Ts_new = calc_Ts(h, k)
        Qdot   = (Th - Tinf) / Rtot

        history.append({
            'j': j, 'Ts': Ts_new, 'Tf': Tf,
            'ReD': ReD, 'NuD': NuD, 'h': h,
            'Rconv': Rconv, 'Rcond': Rcond,
            'Rtot': Rtot, 'Qdot': Qdot
        })

        if abs(Ts_new - Ts) < tol:
            break
        Ts = Ts_new

    return history

# Escenarios de diseño
escenarios = {
    'Caso base':           {'U': 35, 'k': 15},
    'Mejora aerodinámica': {'U': 50, 'k': 15},
    'Cambio de material':  {'U': 35, 'k': 32},
}
costos = {
    'Caso base':           0,
    'Mejora aerodinámica': 300,
    'Cambio de material':  700,
}

resultados = {}
historiales = {}

for nombre, params in escenarios.items():
    hist = iterative_solver(params['U'], params['k'], label=nombre)
    historiales[nombre] = hist
    final = hist[-1]
    resultados[nombre] = final

# Impresión de la tabla de iteraciones (Caso base)
print("=" * 90)
print("TABLA DE ITERACIONES - CASO BASE")
print("=" * 90)
print(f"{'j':>4}  {'Ts [K]':>10}  {'Tf [K]':>10}  {'ReD':>12}  {'NuD':>8}  {'h [W/m²K]':>12}  {'Rconv [K/W]':>14}")
print("-" * 90)
for row in historiales['Caso base']:
    print(f"{row['j']:>4}  {row['Ts']:>10.3f}  {row['Tf']:>10.3f}  "
          f"{row['ReD']:>12.1f}  {row['NuD']:>8.3f}  {row['h']:>12.4f}  {row['Rconv']:>14.6f}")


# Tabla comparativa - tres escenarios

print("\n" + "=" * 100)
print("Tabla comparativa - tres escenarios")
print("=" * 100)
print(f"{'Escenario':<25} {'Ts [K]':>9} {'h [W/m²K]':>11} {'Rcond [K/W]':>13} "
      f"{'Rconv [K/W]':>13} {'Rtot [K/W]':>12} {'Q [W]':>9} {'Iter':>6}")
print("-" * 100)
for nombre, r in resultados.items():
    print(f"{nombre:<25} {r['Ts']:>9.3f} {r['h']:>11.4f} {r['Rcond']:>13.6f} "
          f"{r['Rconv']:>13.6f} {r['Rtot']:>12.6f} {r['Qdot']:>9.3f} {r['j']:>6}")

# Análisis costo-beneficio

print("\n" + "=" * 60)
print("Análisis costo-beneficio")
print("=" * 60)
Ts_base = resultados['Caso base']['Ts']
for nombre in ['Mejora aerodinámica', 'Cambio de material']:
    Ts_alt = resultados[nombre]['Ts']
    delta  = Ts_base - Ts_alt
    costo  = costos[nombre]
    indic  = delta / costo if costo > 0 else float('inf')
    print(f"\n{nombre}:")
    print(f"  ΔTs = {delta:.3f} K")
    print(f"  Costo = USD ${costo}")
    print(f"  Indicador ΔTs/USD = {indic:.6f} K/USD")

# Resistencia dominante
r_base = resultados['Caso base']
print(f"\nResistencia dominante (Caso base):")
print(f"  Rcond = {r_base['Rcond']:.6f} K/W")
print(f"  Rconv = {r_base['Rconv']:.6f} K/W")
if r_base['Rcond'] > r_base['Rconv']:
    print("  → Domina la resistencia CONDUCTIVA")
else:
    print("  → Domina la resistencia CONVECTIVA")

# Gráficos

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle('Convergencia Iterativa - Caso Base', fontsize=13, fontweight='bold')

hist_base = historiales['Caso base']
iters = [r['j'] for r in hist_base]
Ts_iter = [r['Ts'] for r in hist_base]
h_iter  = [r['h']  for r in hist_base]

ax1 = axes[0]
ax1.plot(iters, Ts_iter, 'o-', color='firebrick', linewidth=2, markersize=6)
ax1.axhline(Ts_iter[-1], linestyle='--', color='gray', linewidth=1, label=f'Convergido: {Ts_iter[-1]:.3f} K')
ax1.set_xlabel('Iteración j', fontsize=11)
ax1.set_ylabel('$T_s$ [K]', fontsize=11)
ax1.set_title('Convergencia de $T_s$', fontsize=11)
ax1.legend()
ax1.grid(True, alpha=0.4)

ax2 = axes[1]
ax2.plot(iters, h_iter, 's-', color='steelblue', linewidth=2, markersize=6)
ax2.axhline(h_iter[-1], linestyle='--', color='gray', linewidth=1, label=f'Convergido: {h_iter[-1]:.4f} W/m²K')
ax2.set_xlabel('Iteración j', fontsize=11)
ax2.set_ylabel('$h$ [W/m²K]', fontsize=11)
ax2.set_title('Convergencia de $h$', fontsize=11)
ax2.legend()
ax2.grid(True, alpha=0.4)

plt.tight_layout()
plt.savefig(os.path.join(carpeta_descargas, 'convergencia_caso_base.png'), dpi=150, bbox_inches='tight')
plt.close()

# Gráfico comparativo de los tres escenarios
fig2, ax = plt.subplots(figsize=(9, 5))
colores = ['firebrick', 'steelblue', 'seagreen']
for (nombre, hist), color in zip(historiales.items(), colores):
    iters_e = [r['j'] for r in hist]
    Ts_e    = [r['Ts'] for r in hist]
    ax.plot(iters_e, Ts_e, 'o-', color=color, linewidth=2, markersize=6, label=nombre)

ax.set_xlabel('Iteración j', fontsize=11)
ax.set_ylabel('$T_s$ [K]', fontsize=11)
ax.set_title('Convergencia de $T_s$ — Tres escenarios', fontsize=12, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.4)
plt.tight_layout()
plt.savefig(os.path.join(carpeta_descargas, 'convergencia_escenarios.png'), dpi=150, bbox_inches='tight')
plt.close()

print("\nGráficos guardados exitosamente.")
