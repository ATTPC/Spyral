import polars
from typing import Optional
from matplotlib import pyplot, widgets, colormaps
from matplotlib.colors import LinearSegmentedColormap, rgb2hex
from spyral.core.config import load_config, Config
from spyral.core.workspace import Workspace
from spyral.plot.cut import load_cut_json, write_cut_json, CutHandler
from spyral.plot.histogram import Histogrammer
import numpy as np
import sys

RAD2DEG = 180.0/np.pi

#Additional colormaps with white backgrounds
cmap_jet = colormaps.get_cmap("jet")
white_jet = LinearSegmentedColormap.from_list('white_jet', [
    (0, '#ffffff'),
    (1e-20, rgb2hex(cmap_jet(1e-20))),
    (0.2, rgb2hex(cmap_jet(0.2))),
    (0.4, rgb2hex(cmap_jet(0.4))),
    (0.6, rgb2hex(cmap_jet(0.6))),
    (0.8, rgb2hex(cmap_jet(0.8))),
    (1, rgb2hex(cmap_jet(0.9))),  
], N = 256)

white_viridis = LinearSegmentedColormap.from_list('white_viridis', [
    (0, '#ffffff'),
    (1e-20, '#440053'),
    (0.2, '#404388'),
    (0.4, '#2a788e'),
    (0.6, '#21a784'),
    (0.8, '#78d151'),
    (1, '#fde624'),
], N=256)

def help_string() -> str:
    return\
'''--- PC-Utils plotter ---
Usage: python plotter.py <operation> <configuration.json>
Operations:
    --gate: Load the data specified by the configuration and generate a 2D particle ID histogram. The user then draws an appropriate gate on the histogram. The gate
        is saved to the workspace. The gate has the default name pid_gate and is saved to a default pid_gate.json file in the cuts directory of the workspace.
    --plot: Load the data specified by the workspace and generate a set of plots and histograms defined by the plot function in this script.
'''

# Example of plotting using histogrammer and a gate. Useful for testing gates, but
# real analysis would need a custom solution.
def plot(run_min: int, run_max: int, ws: Workspace, pid_file):
    ede_cut = load_cut_json(ws.get_gate_file_path(pid_file))
    if ede_cut is None:
        print('Cut is invalid, plot failed')
        return

    grammer = Histogrammer()
    grammer.add_hist2d('ede_gated', (200, 200), ((-100.0, 5000.0), (0.0, 2.0)))
    grammer.add_hist2d('ede', (200, 200), ((-100.0, 5000.0), (0.0, 2.0)))
    grammer.add_hist2d('theta_brho_gated', (360, 200), ((0.0, 180.0), (0.0, 2.0)))
    grammer.add_hist2d('theta_brho', (360, 200), ((0.0, 180.0), (0.0, 2.0)))

    for run in range(run_min, run_max+1):
        run_path = ws.get_estimate_file_path_parquet(run)
        if not run_path.exists():
            continue
        df = polars.read_parquet(run_path)
        df_ede = df.filter(polars.struct(['dEdx', 'brho']).map(ede_cut.is_cols_inside))
        grammer.fill_hist2d('ede', df.select('dEdx').to_numpy(), df.select('brho').to_numpy())
        grammer.fill_hist2d('ede_gated', df_ede.select('dEdx').to_numpy(), df_ede.select('brho').to_numpy())
        grammer.fill_hist2d('theta_brho', df.select('polar').to_numpy() * RAD2DEG, df.select('brho').to_numpy())
        grammer.fill_hist2d('theta_brho_gated', df_ede.select('polar').to_numpy() * RAD2DEG, df_ede.select('brho').to_numpy())

    fig, ax = pyplot.subplots(1,2)
    fig.suptitle(f'Runs {run_min} to {run_max} Gated')
    mesh_1 = grammer.draw_hist2d('ede_gated', ax[0], log_z=True)
    mesh_2 = grammer.draw_hist2d('theta_brho_gated', ax[1], log_z=True)
    ax[0].set_xlabel('Energy Loss (channels)')
    ax[0].set_ylabel(r'B$\rho$ (T*m)')
    ax[0].set_title('E-dE')
    pyplot.colorbar(mesh_1, ax=ax[0])
    ax[1].set_xlabel(r'$\theta_{lab}$ (deg)')
    ax[1].set_ylabel(r'B$\rho$ (T*m)')
    ax[1].set_title('Kinematics')
    pyplot.colorbar(mesh_2, ax=ax[1])

    fig2, ax2 = pyplot.subplots(1,2)
    fig2.suptitle(f'Runs {run_min} to {run_max}')
    mesh_12 = grammer.draw_hist2d('ede', ax2[0], log_z=True)
    mesh_22 = grammer.draw_hist2d('theta_brho', ax2[1], log_z=True)
    ax2[0].set_xlabel('Energy Loss (channels)')
    ax2[0].set_ylabel(r'B$\rho$ (T*m)')
    ax2[0].set_title('E-dE')
    pyplot.colorbar(mesh_12, ax=ax2[0])
    ax2[1].set_xlabel(r'$\theta_{lab}$ (deg)')
    ax2[1].set_ylabel(r'B$\rho$ (T*m)')
    ax2[1].set_title('Kinematics')
    pyplot.colorbar(mesh_22, ax=ax2[1])

    pyplot.tight_layout()
    pyplot.show()

# Example of scripted cut generation.
# You have to close the plot window to save the cut
def draw_gate(run_min: int, run_max: int, ws: Workspace):
    handler = CutHandler()
    grammer = Histogrammer()
    grammer.add_hist2d('pid', (400, 300), ((-100.0, 5000.0), (0.0, 1.5)))
    for run in range(run_min, run_max+1):
        run_path = ws.get_estimate_file_path_parquet(run)
        if not run_path.exists():
            continue
        df = polars.read_parquet(ws.get_estimate_file_path_parquet(run))
        grammer.fill_hist2d('pid', df.select('dEdx').to_numpy(), df.select('brho').to_numpy())

    _fig, ax = pyplot.subplots(1,1)
    _selector = widgets.PolygonSelector(ax, handler.onselect)

    mesh = grammer.draw_hist2d(name = 'pid', axis = ax, cmap = white_viridis, log_z = True)

    pyplot.colorbar(mesh, ax=ax)
    pyplot.tight_layout()
    pyplot.show()

    try:
        handler.cuts['cut_0'].name = 'pid_gate'
        write_cut_json(handler.cuts['cut_0'], ws.get_gate_file_path('pid_gate.json'))
    except Exception:
        pass

def main_gate(config: Config):
    ws = Workspace(config.workspace)
    draw_gate(config.run.run_min, config.run.run_max, ws)

def main_plot(config: Config):
    ws = Workspace(config.workspace)
    plot(config.run.run_min, config.run.run_max, ws, config.solver.particle_id_filename)

if __name__ == '__main__':
    if len(sys.argv) < 3:
        print(help_string())
    elif sys.argv[1] == '--gate':
        main_gate(load_config(sys.argv[2]))
    elif sys.argv[1] == '--plot':
        main_plot(load_config(sys.argv[2]))
