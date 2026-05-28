param(
    [string]$OuterRoot = "E:\llm-free-memory-fusion-forgetting-master",
    [string]$InnerRoot = "E:\llm-free-memory-fusion-forgetting-master\llm-free-memory-fusion-forgetting-master",
    [string]$TargetRoot = "E:\llm-free-memory-fusion-forgetting-unified",
    [switch]$Force
)

$ErrorActionPreference = "Stop"

function Assert-PathExists {
    param(
        [string]$Path,
        [string]$Label
    )

    if (-not (Test-Path -LiteralPath $Path)) {
        throw "$Label does not exist: $Path"
    }
}

function Copy-Tree {
    param(
        [string]$Source,
        [string]$Destination
    )

    $sourceTrimmed = $Source.TrimEnd('\')
    $destinationTrimmed = $Destination.TrimEnd('\')
    New-Item -ItemType Directory -Force -Path $destinationTrimmed | Out-Null

    $args = @(
        $sourceTrimmed,
        $destinationTrimmed,
        "/E",
        "/COPY:DAT",
        "/DCOPY:DAT",
        "/R:2",
        "/W:1",
        "/NFL",
        "/NDL",
        "/NP"
    )

    & robocopy @args | Out-Host
    $exitCode = $LASTEXITCODE
    if ($exitCode -ge 8) {
        throw "robocopy failed from '$Source' to '$Destination' with exit code $exitCode"
    }
}

Assert-PathExists -Path $OuterRoot -Label "Outer root"
Assert-PathExists -Path $InnerRoot -Label "Inner root"

if ((Test-Path -LiteralPath $TargetRoot) -and -not $Force) {
    throw "Target already exists: $TargetRoot. Re-run with -Force after removing or renaming it."
}

if (Test-Path -LiteralPath $TargetRoot) {
    Remove-Item -LiteralPath $TargetRoot -Recurse -Force
}

$outerExtras = @(
    ".vexp",
    ".vscode",
    "analytical_plots",
    "method_diagrams",
    "reference paper",
    "writing",
    "PAPER_BLUEPRINT.md",
    "upload.py"
)

Write-Host "Creating flattened workspace at $TargetRoot"
Write-Host "Step 1/3: Copying inner repository as the base workspace..."
Copy-Tree -Source $InnerRoot -Destination $TargetRoot

Write-Host "Step 2/3: Copying outer-only workspace materials..."
foreach ($entry in $outerExtras) {
    $sourcePath = Join-Path $OuterRoot $entry
    if (-not (Test-Path -LiteralPath $sourcePath)) {
        continue
    }

    $destinationPath = Join-Path $TargetRoot $entry
    if (Test-Path -LiteralPath $sourcePath -PathType Container) {
        Copy-Tree -Source $sourcePath -Destination $destinationPath
    }
    else {
        $parent = Split-Path -Parent $destinationPath
        New-Item -ItemType Directory -Force -Path $parent | Out-Null
        Copy-Item -LiteralPath $sourcePath -Destination $destinationPath -Force
    }
}

$manifest = [ordered]@{
    created_at = (Get-Date).ToString("yyyy-MM-dd HH:mm:ss")
    outer_root = $OuterRoot
    inner_root = $InnerRoot
    target_root = $TargetRoot
    base_repository = "inner"
    copied_outer_entries = $outerExtras
    intentionally_skipped_outer_entries = @(
        "llm-free-memory-fusion-forgetting-master",
        "tmp_test_write.txt",
        "tmp_test_write2.txt",
        "tmp_test_write3.txt"
    )
}

Write-Host "Step 3/3: Writing migration manifest..."
$manifest | ConvertTo-Json -Depth 4 | Set-Content -LiteralPath (Join-Path $TargetRoot "workspace_flatten_manifest.json")

Write-Host ""
Write-Host "Flattened workspace is ready:"
Write-Host "  $TargetRoot"
Write-Host ""
Write-Host "Original directories were left untouched."
