# WAAM thermal FEA (Abaqus/Standard 2024)

Transient heat-transfer model of the ER5356 wall (12 layers x 3 mm, 230 x 6 mm) deposited on an
Al 5083 substrate (240 x 75 x 25 mm) by pulsed GMAW (102 A, 800 mm/min, 240 s interpass), following
the process of Sundaram et al. (Materials). It supplies the thermal history used in the manuscript.

## Folder layout

```
WAAM_Abaqus/
├── waam_al5356.inp / .odb ...   ORIGINAL model (Goldak a = 3 mm) and its result - kept unchanged
├── scripts/
│   ├── generate_waam_inp.py     parametric generator of the input deck (replaces the missing
│   │                            generate_waam_al5356_inp.py); defaults rebuild waam_al5356.inp exactly
│   ├── verify_generator.py      proves that: nodes, elements, sets, every flux value, every BC list
│   ├── inp_tools.py             small .inp reader used by the checks
│   ├── extract_odb.py           (abaqus python) ODB -> probe histories (.csv) + nodal field (.npz)
│   ├── fea_analysis.py          per-layer metrics, melt-pool size, re-melt depth, interpass T
│   ├── make_fea_figures.py      journal figures (matplotlib, same style as the GNN figures)
│   └── render_abaqus.py         (abaqus cae noGUI) clean 3D contour renders from the ODB
├── verification_runs/           calibration, sensitivity and mesh-refinement jobs
│   └── run_jobs.bat             run_jobs.bat <cpus> <job> [job ...]   (sequential, resumable)
├── fea_results/                 extracted results (.npz/.csv) and metric tables
└── figures/                     fig_fea_*.png (300 dpi) / .pdf (vector); abaqus/ = 3D renders
```

## Final model (used in the paper): 24 layers x 1.5 mm

The paper does not report the layer count. A mass balance at 102 A / 800 mm/min with typical
wire-feed speeds (4-8 m/min) gives 0.9-1.9 mm layers; 3 mm layers would need ~12.7 m/min and, in the
model, never re-melt the previous layer for any Goldak depth (figures/fig_fea_calibration).
The final model therefore uses 24 x 1.5 mm layers, Goldak a = 3 mm (unchanged), one element row per
layer (same 2.5 x 1.5 x 1.5 mm elements and same 18 984-element count as the original).

```
final\final.inp                 generate_waam_inp.py --layers 24 --layer-height 1.5 --rows-per-layer 1 --no-hfl
verification_runs\v_r2_L4       same, first 4 layers, every element edge halved
verification_runs\v_q110_L4     same, first 4 layers, arc power +10 %
verification_runs\v_q090_L4     same, first 4 layers, arc power -10 %
verification_runs\v_hb100_L4    same, first 4 layers, substrate underside h = 100 W/m2K
```

## Reproducing everything

When the jobs above have finished, one command extracts the results, computes all metrics,
draws every figure (matplotlib + Abaqus renders), builds the verification table and writes the
manuscript section `FEA_manuscript_section.md`:

```
powershell -ExecutionPolicy Bypass -File postprocess_all.ps1
```

Individual steps (from scripts\):

```
python verify_generator.py                                   # generator == original deck
abaqus python extract_odb.py ..\final\final.odb ..\fea_results\final
python fea_analysis.py --run final --layers 24 --layer-height 1.5 --rows 1
python make_fea_figures.py --run final
abaqus cae noGUI=render_abaqus.py -- ..\final\final.odb ..\figures\abaqus
python verification_table.py
python write_fea_section.py
```

Verification variants use the same generator: `--refine 2` (element edges halved),
`--layers 3` (truncated build), `--q-scale 1.1 / 0.9` (arc power), `--h-bottom 100`
(substrate underside contact), `--segments 16` (short wall, calibration only).

## Notes on the original model (waam_al5356.odb)

* Completed successfully (565 steps, 9289 increments, 0 errors, 14.3 h on 4 cores).
* With a = 3 mm the liquidus isotherm reaches only 1.9-2.8 mm below the top of a 3 mm layer, so no
  layer and not the substrate is re-melted (re-melt depth -1.1 to -0.2 mm). This contradicts the
  sound, machinable wall and Goldak's own definition (a = molten-pool depth); the heat-source depth
  was therefore calibrated (see verification_runs/cal_a*), and the final model uses the calibrated a.
