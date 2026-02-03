import numpy as np
from py_wake.site import UniformWeibullSite
from py_wake.examples.data.hornsrev1 import V80
from py_wake import NOJ
from topfarm import TopFarmProblem
from topfarm.cost_models.py_wake_wrapper import PyWakeAEPCostModelComponent
from topfarm.easy_drivers import EasyScipyOptimizer
from topfarm.constraint_components.boundary import BoundaryConstraint

# --- Synthetic offshore wind climate ---
wd = np.arange(0, 360, 30)
A = np.full_like(wd, 9.5, dtype=float)
k = np.full_like(wd, 2.0, dtype=float)
p_wd = np.array([0.04, 0.05, 0.06, 0.08, 0.10, 0.12,
                 0.14, 0.14, 0.10, 0.08, 0.05, 0.04])
p_wd /= p_wd.sum()

site = UniformWeibullSite(p_wd=p_wd, A=A, k=k, ti=0.08)
wt = V80()
wake_model = NOJ(site, wt)
ws = np.arange(3, 26, 1)

# --- Initial turbine layout ---
x0 = np.array([500, 1500, 2500, 3500])
y0 = np.array([500, 500, 500, 500])

# --- Define polygon boundary (meters) ---
boundary = [
    (0, 0),
    (5000, 0),
    (5000, 4000),
    (2000, 6000),
    (0, 4000)
]

boundary_con = BoundaryConstraint(boundary)

# --- AEP cost model ---
aep_comp = PyWakeAEPCostModelComponent(
    site=site,
    windTurbines=wt,
    wake_model=wake_model,
    wd=wd,
    ws=ws,
    n_wt=len(x0)
)

# --- Optimisation problem ---
problem = TopFarmProblem(
    design_vars={
        "x": (x0, 0, 5000),
        "y": (y0, 0, 6000),
    },
    cost_comp=aep_comp,
    constraints=[boundary_con],
    driver=EasyScipyOptimizer(),
    maximize=True
)

cost, state, recorder = problem.optimize()

print("Optimised AEP:", cost)
print("Optimised x:", state["x"])
print("Optimised y:", state["y"])
