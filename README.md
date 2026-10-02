# AltGDA-PEP

Research code for studying **alternating gradient descent–ascent (AltGDA)** and
comparing it with **simultaneous gradient descent–ascent (SimGDA)** in constrained
bilinear minimax problems. The repository contains performance estimation
problems (PEPs) formulated as semidefinite programs in Julia, together with Python
notebooks for zero-sum game simulations and visualizations.

## Repository contents

| Path | Purpose |
| --- | --- |
| [`PEP/sdp.jl`](PEP/sdp.jl) | Builds and solves the PEP for prescribed step sizes using JuMP and MOSEK. |
| [`PEP/utils.jl`](PEP/utils.jl) | Constructs algorithm coefficients and matrices used by the PEP. |
| [`PEP/search_stepsize_optimization.jl`](PEP/search_stepsize_optimization.jl) | Searches for constant step sizes for AltGDA and SimGDA. |
| [`PEP/csv.jl`](PEP/csv.jl) | Combines saved JLD results into CSV tables. |
| [`PEP/plot.jl`](PEP/plot.jl) | Plots optimized step sizes and PEP objective values from the CSV tables. |
| [`PEP/data/`](PEP/data/) | Saved PEP results for iteration counts 5–50, CSV summaries, and a notebook for generating LaTeX table rows. |
| [`PEP/figures/`](PEP/figures/) | PEP comparison plots. |
| [`behaviors/AltGDA_behaviors.ipynb`](behaviors/AltGDA_behaviors.ipynb) | Trajectories, averaged-iterate duality gaps, and energy in games with and without an interior Nash equilibrium. |
| [`numerical_performances/numerical_performances.ipynb`](numerical_performances/numerical_performances.ipynb) | Compares AltGDA and SimGDA on random matrix games. |
| [`numerical_performances/numerical_performances_stepsizechoices.ipynb`](numerical_performances/numerical_performances_stepsizechoices.ipynb) | Compares AltGDA with step sizes 0.1, 0.01, and 0.001. |

## Getting started

```sh
git clone https://github.com/CoffeeAndConvexity/AltGDA-PEP.git
cd AltGDA-PEP
```

The Julia and Python experiments can be used independently. Dependency versions
are not pinned in the repository.

### Python notebooks

Install Python 3 and create an environment with the notebook dependencies:

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install jupyterlab numpy matplotlib tqdm mpltern pandas
python -m jupyterlab
```

Open a notebook and run its cells in order. Use the notebook's containing
directory as the kernel's working directory, since input and output paths are
relative to it:

- `behaviors/`: saves trajectory, duality-gap, and energy PNGs in this directory.
- `numerical_performances/`: creates `data/` and `figures/` for simulation results
  and plots.
- `PEP/data/`: run `transform_Latex_table.ipynb` here to read `AltGDA.csv` and
  `SimGDA.csv` and generate LaTeX table rows.

The numerical comparison notebooks default to 30 × 60 games, one million
iterations, six matrix distributions, and ten initializations. Reduce `T` and
the experiment loops for a quick trial. The behavior notebook also includes
long iteration loops for trajectories and equilibrium estimation.

In the step-size comparison notebook, the statistics cell saves summary arrays
to the same `.npz` filename used for raw simulation arrays. Regenerate the raw
data before rerunning that cell, or choose a separate output filename.

### Julia PEP experiments

Install Julia and configure a working MOSEK installation and license to solve
the semidefinite programs. From the repository root, start `julia` and install
the packages imported by the scripts:

```julia
using Pkg
Pkg.activate(".")
Pkg.add([
    "JuMP", "MosekTools", "Mosek", "OffsetArrays", "Gurobi", "Ipopt",
    "JLD2", "Distributions", "OrderedCollections", "BenchmarkTools",
    "PrettyTables", "LDLFactorizations", "HDF5", "JLD",
    "CSV", "DataFrames", "CSVFiles", "Plots"
])
```

This creates a local Julia project environment. The current scripts import
Gurobi and Ipopt as well, although the PEP solver is configured to use MOSEK.

To inspect the saved PEP results without rerunning the optimization, use
[`AltGDA.csv`](PEP/data/AltGDA.csv) and [`SimGDA.csv`](PEP/data/SimGDA.csv).
Each contains the iteration count (`N`), selected step size (`optimal_η`), and
corresponding PEP objective value (`optimal_obj`). To rebuild the CSVs from the
included JLD files, run from the repository root:

```sh
julia --project=. PEP/csv.jl
```

The plotting script expects `PEP/` as its working directory. From the repository
root, run:

```sh
(cd PEP && julia --project=.. plot.jl)
```

To run the step-size search, return to the repository root and execute:

```sh
julia --project=. PEP/search_stepsize_optimization.jl
```

The search defaults to both algorithms, `L = 1`, and iteration counts 5–30.
It repeatedly solves PEPs while refining a grid of candidate step sizes, using
equal primal and dual step sizes `η = 1 / (η_c * L)`. Edit `start_N`, `end_N`,
and the algorithm-specific search intervals in the script to change the
experiment. Results are saved as `PEP/data/<algorithm>_<start_N>_<end_N>.jld`.

Rerunning scripts can overwrite the bundled results and figures. The CSV
conversion script expects the seven saved iteration batches covering 5–50;
adjust its batch list if you use a different partition. Full step-size searches
involve many SDP solves and can take substantial time.
