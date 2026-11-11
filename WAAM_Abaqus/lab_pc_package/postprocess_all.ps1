# Post-processes the final FEA run and the verification runs, then draws every figure.
# Run from WAAM_Abaqus:   powershell -ExecutionPolicy Bypass -File postprocess_all.ps1
$ErrorActionPreference = "Continue"
$abq = if (Get-Command abaqus -ErrorAction SilentlyContinue) { "abaqus" } else { "C:\SIMULIA\Commands\abaqus.bat" }
Set-Location $PSScriptRoot\scripts

function Done($sta) { (Test-Path $sta) -and (Select-String -Path $sta -Pattern "COMPLETED SUCCESSFULLY" -Quiet) }

# ---- production run: 24 layers x 1.5 mm
if (Done "..\final\final.sta") {
    & $abq python extract_odb.py ..\final\final.odb ..\fea_results\final | Select-Object -Last 1
    python fea_analysis.py --run final --layers 24 --layer-height 1.5 --rows 1 | Out-File ..\fea_results\final_summary.txt -Encoding utf8
    python make_fea_figures.py --run final --layers 24 --layer-height 1.5 --rows 1 --only setup sections fusion cycles metrics
    New-Item -ItemType Directory -Force ..\figures\abaqus | Out-Null
    & $abq cae noGUI=render_abaqus.py -- ..\final\final.odb ..\figures\abaqus L01_S23 L12_S23 L24_S23 Interpass_L23 Final_cooling
} else { Write-Host "final run not complete yet" }

# ---- verification runs (first 4 layers)
$v = @{ "v_r2_L4" = "--refine 2"; "v_q110_L4" = ""; "v_q090_L4" = ""; "v_hb100_L4" = "" }
foreach ($k in $v.Keys) {
    if (Done "..\verification_runs\$k\$k.sta") {
        & $abq python extract_odb.py ..\verification_runs\$k\$k.odb ..\fea_results\$k | Select-Object -Last 1
        $extra = if ($v[$k]) { $v[$k].Split(" ") } else { @() }
        python fea_analysis.py --run $k --layers 4 --layer-height 1.5 --rows 1 @extra | Out-File ..\fea_results\${k}_summary.txt -Encoding utf8
    } else { Write-Host "$k not complete yet" }
}
python make_fea_figures.py --only verification calibration
python verification_table.py
if (Test-Path ..\fea_results\final_layer_metrics.csv) { python write_fea_section.py }
Write-Host "done - figures in WAAM_Abaqus\figures, tables in WAAM_Abaqus\fea_results"
