# -*- coding: utf-8 -*-
"""Publication-quality 3D temperature contours rendered by Abaqus/Viewer (no GUI).

    abaqus cae noGUI=render_abaqus.py -- <job.odb> <out_dir> [step1 step2 ...]

For each requested step the last frame is rendered with a white background, no title/state
block/logo, a large legend with fixed limits (0-800 degC in 50 degC bands, > 800 degC drawn in grey) and the
same isometric camera, and saved as <out_dir>/abq_<step>.png (and .tif for the journal).
"""
import os
import sys

from abaqus import session
import visualization  # noqa: F401  (registers Odb as a displayable object)
import displayGroupOdbToolset  # noqa: F401
from abaqusConstants import (CONTOURS_ON_UNDEF, FEATURE, FIXED, NODAL, OFF, ON, PNG, SOLID, SPECIFY,
                             TIFF, VERY_THIN)

_i = [k for k, s in enumerate(sys.argv) if s.lower().endswith(".odb")][-1]   # args after the .odb path
args = sys.argv[_i:]
odb_path, out_dir = args[0], args[1]
steps = args[2:] or ["L01_S23", "L06_S23", "L12_S23", "Final_cooling"]
if not os.path.isdir(out_dir):
    os.makedirs(out_dir)

vp = session.Viewport(name="R", origin=(0, 0), width=300, height=150)
vp.makeCurrent()
odb = session.openOdb(name=odb_path, readOnly=True)
vp.setValues(displayedObject=odb)

# ---- clean annotations
vp.viewportAnnotationOptions.setValues(triad=ON, title=OFF, state=OFF, annotations=ON, compass=OFF,
                                       legend=ON, legendFont="-*-arial-bold-r-normal-*-*-150-*-*-p-*-*-*",
                                       legendBox=OFF, legendTitle=OFF, legendDecimalPlaces=0,
                                       legendNumberFormat=FIXED, triadFont="-*-arial-bold-r-normal-*-*-140-*-*-p-*-*-*")
session.graphicsOptions.setValues(backgroundStyle=SOLID, backgroundColor="#FFFFFF")
# same perceptually uniform map as the matplotlib figures (inferno, 12 stops)
session.Spectrum(name="Inferno", colors=(
    "#02020c", "#110a30", "#290b55", "#470b6a", "#61136e", "#7c1d6d", "#952667", "#ae305c",
    "#c73e4c", "#db503b", "#eb6628", "#f78212", "#fc9f07", "#fac026", "#f4df53", "#f5f992"))
session.printOptions.setValues(vpDecorations=OFF, vpBackground=ON, reduceColors=False)
session.pngOptions.setValues(imageSize=(4000, 2000))
session.tiffOptions.setValues(imageSize=(2700, 1350))

od = vp.odbDisplay
od.setPrimaryVariable(variableLabel="NT11", outputPosition=NODAL)
od.display.setValues(plotState=(CONTOURS_ON_UNDEF,))
od.contourOptions.setValues(numIntervals=16, maxAutoCompute=OFF, maxValue=800.0, minAutoCompute=OFF,
                            minValue=0.0, showMaxLocation=OFF, showMinLocation=OFF,
                            outsideLimitsMode=SPECIFY, outsideLimitsAboveColor="#B4B4B4",
                            spectrum="Inferno")
od.commonOptions.setValues(visibleEdges=FEATURE, edgeColorFillShade="#3C3C3C", edgeLineThickness=VERY_THIN)
vp.view.setViewpoint(viewVector=(0.55, -1.0, 0.55), cameraUpVector=(0, 0, 1))
vp.view.fitView()
vp.view.zoom(zoomFactor=0.92)
vp.view.pan(xFraction=0.10, yFraction=0.0)

step_names = list(odb.steps.keys())
for s in steps:
    if s not in step_names:
        print("skip (no step):", s)
        continue
    i = step_names.index(s)
    fr = len(odb.steps[s].frames) - 1
    od.setFrame(step=i, frame=fr)
    base = os.path.join(out_dir, "abq_" + s)
    session.printToFile(fileName=base, format=PNG, canvasObjects=(vp,))
    session.printToFile(fileName=base, format=TIFF, canvasObjects=(vp,))
    print("saved", base)
odb.close()
