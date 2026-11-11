WAAM THERMAL FEA - LAB PC PACKAGE
=================================

Model: ER5356 wall (230 x 6 mm, 24 layers x 1.5 mm = 36 mm) on Al 5083 substrate (240 x 75 x 25 mm),
pulsed GMAW 102 A, 800 mm/min, 240 s interpass, Goldak source Q = 1469 W.
Abaqus/Standard 2024 (any version 2020+ should read these .inp files).

FILES
  final\final.inp                          production model (24 layers, 18 984 elements)
  verification_runs\v_q110_L4\...inp       arc power +10 %   (first 4 layers)
  verification_runs\v_q090_L4\...inp       arc power -10 %   (first 4 layers)
  verification_runs\v_hb100_L4\...inp      substrate underside h = 100 W/m2K (first 4 layers)
  verification_runs\v_r2_L4\...inp         mesh refined x2, all edges halved (first 4 layers)
  scripts\                                 post-processing (Python 3 + numpy/pandas/matplotlib)
  fea_results\cal_*                        calibration results already computed (needed for one figure)

HOW TO RUN
  1. Copy this whole folder to the lab PC (keep the folder structure).
  2. Double-click RUN_ALL.bat  (or in a command window:  RUN_ALL.bat 8   to use 8 cores)
     Jobs run one after another; progress in run_log.txt. If interrupted, run it again -
     finished jobs are skipped.
  3. When it says ALL JOBS FINISHED:
        powershell -ExecutionPolicy Bypass -File postprocess_all.ps1
     -> figures\*.png/.pdf, fea_results\verification_table.csv, FEA_manuscript_section.md

  Single job by hand (inside its folder):
        abaqus job=final input=final.inp cpus=8 interactive

EXPECTED RUN TIMES (measured on a 6-core laptop with 4 cores; a lab workstation is faster)
  final       ~6-10 h        (1 129 steps)
  v_q110/v_q090/v_hb100   ~1 h each
  v_r2_L4     ~4-8 h         (8x more elements, needs ~3-4 GB RAM)
  -> start final first; the verification jobs can also run in parallel on a second
     machine or with spare cores (each job folder is independent).

DISK: final.odb ~2-3 GB (no element heat-flux output requested). Keep >= 10 GB free.

CHECK: each job's .sta file ends with "THE ANALYSIS HAS COMPLETED SUCCESSFULLY".
