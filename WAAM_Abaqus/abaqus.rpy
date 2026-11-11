# -*- coding: mbcs -*-
#
# Abaqus/CAE Release 2024 replay file
# Internal Version: 2023_09_21-18.25.25 RELr426 190762
# Run by Rashaad Rowther on Sun Nov  1 22:27:38 2026
#

# from driverUtils import executeOnCaeGraphicsStartup
# executeOnCaeGraphicsStartup()
#: Executing "onCaeGraphicsStartup()" in the site directory ...
from abaqus import *
from abaqusConstants import *
session.Viewport(name='Viewport: 1', origin=(1.11979, 1.12269), width=164.833, 
    height=111.37)
session.viewports['Viewport: 1'].makeCurrent()
from driverUtils import executeOnCaeStartup
executeOnCaeStartup()
execfile('scripts/render_abaqus.py', __main__.__dict__)
#: Model: C:/Project/Deep Leanring x MP2/deep learning x mp2/WAAM_Abaqus/waam_al5356.odb
#: Number of Assemblies:         1
#: Number of Assembly instances: 0
#: Number of Part instances:     1
#: Number of Meshes:             1
#: Number of Element Sets:       2782
#: Number of Node Sets:          18
#: Number of Steps:              565
#: saved figures\abaqus\abq_L01_S23
#: saved figures\abaqus\abq_L06_S23
#: saved figures\abaqus\abq_L12_S23
#: saved figures\abaqus\abq_Interpass_L11
print('RT script done')
#: RT script done
