# `underestimating-hyperplanes`: Tangent-Plane Approximation for Nonlinear Curves

A lightweight Python toolkit to build **underestimating hyperplanes** for scalar-valued functions.  
It constructs tangents in 1D/2D/nD and filters them to form a small set of **representative linear underestimators** (basically a simplified lower convex envelope). Useful for curve/surface simplification, robust sizing, and optimisation-friendly surrogates  - e.g., **fan and duct** performance in HVAC.

⚠️ This repository accompanies the paper:  
*Optimal Ventilation Topology Planning via MILP: Integrated Duct Sizing, Fan Placement,
and Control* (Julius H.P. Breuer, Peter F. Pelz, Building & Engineering, 2025, submitted).  

It serves both as **supplementary material** and as a **standalone Python toolkit**.  
If you use this code in academic work, please cite the paper (preferred) or this repository.

---

## Features

- ✏️ **Tangent generation** at user-picked points via function + gradient callbacks.
- ✂️ **Intersection filtering** to keep only planes that lie below the curve/surface on the sampled domain.
- 🧱 **Lower hull utilities** to extract underestimating facets/hyperplanes from sampled point clouds.
- 🧪 **Turbo-machinery helpers**:
  - Filter value ranges safely (`filter_with_upper_limit`)
  - Build **P<sub>el</sub>–n** functions for fans (`get_pel_n_func`)
  - Compute **maximums** for volume flow / pressure rise / losses (`calculate_max_values`)
- 📄 **Export-ready structures**: convert planes to arrays/dicts for easy YAML/CSV dumping.
- 🔧 **Grid sampling** in N-D with optional constraints, great for building training clouds.

---

## Installation

```bash
# from the project root
pip install -e .

# or install runtime deps explicitly
pip install numpy scipy sympy
```

Python ≥ 3.7 required.
	Note: scipy is needed for 3-D lower-hull utilities; sympy is used by turbo-machinery helpers.

## Repository Layout
```
src/underestimating_hyperplanes/
  __init__.py
  tangent_utils.py              # tangents, filtering, hull utilities, grid sampling
  turbo_machinery_tangents.py   # fan/duct helpers (max points, safe filters, Pel(n))

examples/
  duct_curve.ipynb
  fan_curve_under_over_estimation.ipynb
  input_files/
    input_fan_example_data.yaml
  output_files/
    duct_friction.yaml
    output_fan_example_hyperplanes.yaml
    output_fan_example_overestimation_hyperplanes.yaml

LICENSE
README.md
requirements.txt
setup.py
```

See examples/ for end-to-end notebooks and example I/O.

## I/O Examples
- Input fan data (see examples/input_files/input_fan_example_data.yaml): cubic/poly coefficient sets for fan pressure and power curves per diameter.
- Outputs:
  - `examples/output_files/output_fan_example_hyperplanes.yaml`: underestimating planes for fan curve(s)
  ``` 
  intercept: [...]
  grad_volume_flow: [...]
  point_volume_flow: [...]
  ```
  - `examples/output_files/output_fan_example_overestimation_hyperplanes.yaml`: over-estimating planes (for comparison / envelopes).
  - `examples/output_files/duct_friction.yaml`: duct friction curve tangents with per-plane gradients.

## Turbo-machinery helpers
Code that helps with application to underestimate non-convex turbo-machinery.
- `filter_with_upper_limit(arr, upper_limit) -> np.ndarray`
Keep values ≤ limit plus the first value above it (continuity).
- get_pel_n_func(...)
Construct electrical power–speed relation for fans.
- calculate_max_values(...) -> { "q": ..., "dp": ..., "ploss": ... }
Compute maxima for volume flow, pressure rise, power loss from polynomial curve sets.


## Tips
- Pick tangent points where curvature changes—denser around elbows of the curve.
- Use lower_hull_planes as a quick post-processing check to ensure underestimation.
- Keep units consistent (e.g., m³/h, Pa, W) across functions and gradients.
- For 3-D hulls, install scipy. For turbo-machinery characteristics, install sympy.

## Note on AI usage
Parts of this repository (documentation and/or code snippets) were prepared with the assistance of AI-based tools, namely ChatGPT version 4 and 5. All outputs were reviewed, validated, and adapted by the authors.

## Contributing
Contributions are welcome!
- Add small, runnable demos in examples/ (or examples/).
- Include docstrings and minimal examples for new utilities.
- Keep naming and code style consistent with the current modules.


## Funding
The presented results were obtained within the research project ‘‘Algorithmic System Planning of Air Handling Units’’, Project No. 22289 N/1, funded by the program for promoting the Industrial Collective Research (IGF) of the German Ministry of Economic Affairs and Climate Action (BMWK), approved by the Deutsches Zentrum für Luft- und Raumfahrt (DLR). We thank all participants of the working group for the constructive collaboration.


## License
MIT — see LICENSE.


## How to Cite
tbd.
