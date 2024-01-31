from spyral.core.config import load_config, Config
from spyral.core.workspace import Workspace

from spyral_utils.plot.cut import load_cut_json, write_cut_json, CutHandler
from spyral_utils.plot.histogram import Histogrammer

import polars
from matplotlib import pyplot, widgets
import numpy as np
from pathlib import Path
import click

RAD2DEG = 180.0 / np.pi


# Example of plotting using histogrammer and a gate. Useful for testing gates, but
# real analysis would need a custom solution.
def plot(config: Config):
    ws = Workspace(config.workspace)
    ede_cut = load_cut_json(ws.get_gate_file_path(config.solver.particle_id_filename))
    if ede_cut is None:
        print("Cut is invalid, plot failed")
        return

    grammer = Histogrammer()
    grammer.add_hist2d("ede_gated", (200, 200), ((-100.0, 5000.0), (0.0, 3.0)))
    grammer.add_hist2d("ede", (500, 300), ((-100.0, 5000.0), (0.0, 1.5)))
    grammer.add_hist2d("theta_brho_gated", (360, 200), ((0.0, 180.0), (0.0, 3.0)))
    grammer.add_hist2d("theta_brho", (360, 200), ((0.0, 180.0), (0.0, 3.0)))
    grammer.add_hist1d("ic_amp", 4096, (0.0, 4096.0))

    for run in range(config.run.run_min, config.run.run_max + 1):
        run_path = ws.get_estimate_file_path_parquet(run)
        if not run_path.exists():
            continue
        df = polars.read_parquet(run_path)
        # df = df.filter((polars.col('ic_amplitude') > config.solver.ic_min_val) & (polars.col('ic_amplitude') < config.solver.ic_max_val))
        df_ede = df.filter(
            polars.struct(["dEdx", "brho"]).map_batches(ede_cut.is_cols_inside)
        )
        grammer.fill_hist2d(
            "ede", df.select("dEdx").to_numpy(), df.select("brho").to_numpy()
        )
        grammer.fill_hist2d(
            "ede_gated",
            df_ede.select("dEdx").to_numpy(),
            df_ede.select("brho").to_numpy(),
        )
        grammer.fill_hist2d(
            "theta_brho",
            df.select("polar").to_numpy() * RAD2DEG,
            df.select("brho").to_numpy(),
        )
        grammer.fill_hist2d(
            "theta_brho_gated",
            df_ede.select("polar").to_numpy() * RAD2DEG,
            df_ede.select("brho").to_numpy(),
        )
        grammer.fill_hist1d(
            "ic_amp", df.unique(subset=["event"]).select("ic_amplitude").to_numpy()
        )

    fig, ax = pyplot.subplots(1, 2)
    fig.suptitle(f"Runs {config.run.run_min} to {config.run.run_max} Gated")
    mesh_1 = grammer.draw_hist2d("ede_gated", ax[0], log_z=False)
    mesh_2 = grammer.draw_hist2d("theta_brho_gated", ax[1], log_z=False)
    ax[0].set_xlabel("Energy Loss (channels)")
    ax[0].set_ylabel(r"B$\rho$ (T*m)")
    ax[0].set_title("E-dE")
    pyplot.colorbar(mesh_1, ax=ax[0])
    ax[1].set_xlabel(r"$\theta_{lab}$ (deg)")
    ax[1].set_ylabel(r"B$\rho$ (T*m)")
    ax[1].set_title("Kinematics")
    pyplot.colorbar(mesh_2, ax=ax[1])

    pyplot.tight_layout()

    fig2, ax2 = pyplot.subplots(1, 2)
    fig2.suptitle(f"Runs {config.run.run_min} to {config.run.run_max}")
    mesh_12 = grammer.draw_hist2d("ede", ax2[0], log_z=True)
    mesh_22 = grammer.draw_hist2d("theta_brho", ax2[1], log_z=True)
    ax2[0].set_xlabel("Energy Loss (channels)")
    ax2[0].set_ylabel(r"B$\rho$ (T*m)")
    ax2[0].set_title("E-dE")
    pyplot.colorbar(mesh_12, ax=ax2[0])
    ax2[1].set_xlabel(r"$\theta_{lab}$ (deg)")
    ax2[1].set_ylabel(r"B$\rho$ (T*m)")
    ax2[1].set_title("Kinematics")
    pyplot.colorbar(mesh_22, ax=ax2[1])

    fig3, ax3 = pyplot.subplots(1, 1)
    fig3.suptitle(f"Runs {config.run.run_min} to {config.run.run_max} IC")
    grammer.draw_hist1d("ic_amp", ax3)
    ax3.set_xlabel("IC Amplitude")
    ax3.set_yscale("log")

    pyplot.tight_layout()
    pyplot.show()


# Example of scripted cut generation.
# You have to close the plot window to save the cut
def draw_gate(config: Config):
    ws = Workspace(config.workspace)
    handler = CutHandler()
    grammer = Histogrammer()
    grammer.add_hist2d("pid", (400, 300), ((-100.0, 5000.0), (0.0, 1.5)))
    for run in range(config.run.run_min, config.run.run_max + 1):
        run_path = ws.get_estimate_file_path_parquet(run)
        if not run_path.exists():
            continue
        df = polars.read_parquet(ws.get_estimate_file_path_parquet(run))
        # df = df.filter((polars.col('ic_amplitude') > config.solver.ic_min_val) & (polars.col('ic_amplitude') < config.solver.ic_max_val))
        grammer.fill_hist2d(
            "pid", df.select("dEdx").to_numpy(), df.select("brho").to_numpy()
        )

    _fig, ax = pyplot.subplots(1, 1)
    _selector = widgets.PolygonSelector(ax, handler.onselect)

    mesh = grammer.draw_hist2d(name="pid", axis=ax, log_z=True)

    ax.set_xlabel("Energy loss (channels)")
    ax.set_ylabel(r"B$\rho$ (Tm)")

    pyplot.colorbar(mesh, ax=ax)
    pyplot.tight_layout()
    pyplot.show()

    try:
        handler.cuts["cut_0"].name = "pid_gate"
        write_cut_json(
            handler.cuts["cut_0"], str(ws.get_gate_file_path("pid_gate.json"))
        )
    except Exception:
        pass


def run_gate(config: Config):
    draw_gate(config)


def run_plot(config: Config):
    plot(config)


@click.command()
@click.option(
    "--gate/--plot",
    default=True,
    help="Switch between drawing a gate and plotting basic kinematics",
    show_default=True,
)
@click.argument("config", type=click.Path(exists=True))
def plotter(gate: bool, config: Path):
    """
    Tool for exploring the results of Phase 3 of Spyral analysis. Generate 2D particle ID gates using the --gate option, or display
    estimated kinematics using the --plot option. Configuration parameters are passed using a Spyral configuration file specified by CONFIG.
    """
    configuration = load_config(config)
    if gate:
        run_gate(configuration)
    else:
        run_plot(configuration)


if __name__ == "__main__":
    plotter()
