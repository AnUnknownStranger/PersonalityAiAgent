$ErrorActionPreference = "Stop"

$root = $PSScriptRoot
$sourceDir = Join-Path $root "Harry_transcript"
$outputDir = Join-Path $root "Harry_transcript_clean"
$combinedPath = Join-Path $root "Harry_all_clean.txt"

if (-not (Test-Path $outputDir)) {
    New-Item -ItemType Directory -Path $outputDir | Out-Null
}

function Clean-PureDialogue {
    param([string]$Text)

    if ([string]::IsNullOrWhiteSpace($Text)) {
        return ""
    }

    $clean = $Text
    $clean = $clean -replace '\[[^\]]*\]', ' '
    $clean = $clean -replace '\([^)]*\)', ' '
    $clean = $clean -replace '\s+', ' '
    $clean = $clean.Trim()
    $clean = $clean -replace '^[,;:\- ]+', ''
    $clean = $clean -replace '\s+([,.!?;:])', '$1'
    return $clean.Trim()
}

$allLines = [System.Collections.Generic.List[string]]::new()

Get-ChildItem $sourceDir -File | Sort-Object Name | ForEach-Object {
    $movie = $_.Name
    $outPath = Join-Path $outputDir $movie
    $cleanLines = [System.Collections.Generic.List[string]]::new()

    Get-Content -Encoding UTF8 $_.FullName | ForEach-Object {
        $clean = Clean-PureDialogue $_
        if ($clean) {
            [void]$cleanLines.Add($clean)
            [void]$allLines.Add("[$movie] $clean")
        }
    }

    [System.IO.File]::WriteAllLines($outPath, $cleanLines, [System.Text.UTF8Encoding]::new($false))
    Write-Output "$movie => $($cleanLines.Count) clean lines"
}

[System.IO.File]::WriteAllLines($combinedPath, $allLines, [System.Text.UTF8Encoding]::new($false))
Write-Output "Saved combined clean file => $combinedPath"
