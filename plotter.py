import polars
from matplotlib import pyplot, widgets
from pcutils.plot.cut import load_cut_json, write_cut_json, CutHandler
from pcutils.plot.histogram import Histogrammer
from pathlib import Path
from typing import Optional
import numpy as np

RAD2DEG = 180.0/np.pi
DATA_DIRECTORY: str = '/Volumes/Pattern/Analysis/a1975/estimates/'

#Merge a bunch of runs into one dataframe. This can be useful for doing one-shot analysis,
#but need to be mindful of memory limitations (and performance penalties)
def merge_runs_to_dataframe(run_min: int, run_max: int) -> polars.DataFrame:
    data_path = Path(DATA_DIRECTORY)
    path = data_path / f'run_{run_min:04d}.parquet'
    total_df = polars.read_parquet(path)
    for i in range(run_min+1, run_max+1):
        path = data_path / f'run_{i:04d}.parquet'
        if path.exists():
            total_df.vstack(polars.read_parquet(path), in_place=True)
    total_df.rechunk()
    return total_df

def get_dataframe(run_num: int) -> Optional[polars.DataFrame]:
    data_path = Path(DATA_DIRECTORY)
    path = data_path / f'run_{run_num:04d}.parquet'
    if path.exists():
        return polars.read_parquet(path)
    else:
        return None

#Example plotter making an xavg histogram with an ede gate
def plot(run_min: int, run_max: int):
    ede_cut = load_cut_json('ede_cut.json')
    if ede_cut is None:
        print('Cut is invalid, plot failed')
        return

    grammer = Histogrammer()
    grammer.add_hist2d('ede_gated', (200, 150), ((0.0, 20000.0), (0.0, 1.5)))
    grammer.add_hist2d('ede', (200, 150), ((0.0, 20000.0), (0.0, 1.5)))
    grammer.add_hist2d('theta_brho_gated', (180, 150), ((0.0, 180.0), (0.0, 1.5)))
    grammer.add_hist2d('theta_brho', (180, 150), ((0.0, 180.0), (0.0, 1.5)))

    for run in range(run_min, run_max+1):
        df = get_dataframe(run)
        if df is None:
            continue
        df_ede = df.filter(polars.struct(['dEdx', 'brho']).map(ede_cut.is_cols_inside))
        grammer.fill_hist2d('ede', df.select('dEdx').to_numpy(), df.select('brho').to_numpy())
        grammer.fill_hist2d('ede_gated', df_ede.select('dEdx').to_numpy(), df_ede.select('brho').to_numpy())
        grammer.fill_hist2d('theta_brho', df.select('polar').to_numpy() * RAD2DEG, df.select('brho').to_numpy())
        grammer.fill_hist2d('theta_brho_gated', df_ede.select('polar').to_numpy() * RAD2DEG, df_ede.select('brho').to_numpy())

    fig, ax = pyplot.subplots(1,2)
    fig.suptitle(f'Runs {run_min} to {run_max}')
    mesh_1 = grammer.draw_hist2d('ede', ax[0])
    mesh_2 = grammer.draw_hist2d('theta_brho', ax[1], log_z=True)
    ax[0].set_xlabel('Energy Loss (channels)')
    ax[0].set_ylabel(r'B$\rho$ (T*m)')
    ax[0].set_title('E-dE')
    pyplot.colorbar(mesh_1, ax=ax[0])
    ax[1].set_xlabel(r'$\theta_{lab}$ (deg)')
    ax[1].set_ylabel(r'B$\rho$ (T*m)')
    ax[1].set_title('Kinematics')
    pyplot.colorbar(mesh_2, ax=ax[1])

    pyplot.tight_layout()
    pyplot.show()

#Example of scripted cut generation. You have to close the plot window to save the cut
def draw_ede_cut(run_min: int, run_max: int):
    handler = CutHandler()
    grammer = Histogrammer()

    grammer.add_hist2d('ede', (200, 150), ((0.0, 20000.0), (0.0, 1.5)))
    for run in range(run_min, run_max+1):
        df = get_dataframe(run)
        grammer.fill_hist2d('ede', df.select('dEdx').to_numpy(), df.select('brho').to_numpy())

    fig, ax = pyplot.subplots(1,1)
    selector = widgets.PolygonSelector(ax, handler.onselect)

    mesh = grammer.draw_hist2d('ede', ax)
    pyplot.colorbar(mesh, ax=ax)
    pyplot.tight_layout()
    pyplot.show()

    try:
        handler.cuts['cut_0'].name = 'ede_cut'
        write_cut_json(handler.cuts['cut_0'], 'ede_cut.json')
    except:
        pass

if __name__ == '__main__':
    plot(4, 4)
    #draw_ede_cut(4, 4)