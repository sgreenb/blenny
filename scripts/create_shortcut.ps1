# Creates a "Blenny GUI" shortcut on the current user's Desktop that
# launches the GUI by double-clicking the icon.
#
# Usage:
#   powershell -NoProfile -ExecutionPolicy Bypass -File scripts\create_shortcut.ps1
#
# (or right-click -> Run with PowerShell). The shortcut points at
# scripts\launch_gui.bat and uses screenshots\blenny_icon.ico as its icon.

$ErrorActionPreference = "Stop"

$repo = Split-Path -Parent $PSScriptRoot
$bat = Join-Path $repo "scripts\launch_gui.bat"
$icon = Join-Path $repo "screenshots\blenny_icon.ico"

if (-not (Test-Path $bat)) { throw "Launcher not found: $bat" }
if (-not (Test-Path $icon)) { throw "Icon not found: $icon" }

$desktop = [Environment]::GetFolderPath("Desktop")
$lnk = Join-Path $desktop "Blenny GUI.lnk"

$ws = New-Object -ComObject WScript.Shell
$sc = $ws.CreateShortcut($lnk)
$sc.TargetPath = $bat
$sc.WorkingDirectory = $repo
$sc.IconLocation = "$icon,0"
$sc.Description = "Launch the Blenny plate reader GUI"
$sc.Save()

Write-Host "Created shortcut: $lnk"
