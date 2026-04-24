$ErrorActionPreference = "Stop"

$root = $PSScriptRoot
$transcriptDir = Join-Path $root "Transcript"
$outputDir = Join-Path $root "Harry_transcript"
$combinedPath = Join-Path $root "Harry_all.txt"

if (-not (Test-Path $outputDir)) {
    New-Item -ItemType Directory -Path $outputDir | Out-Null
}

function Clean-DialogueLine {
    param([string]$Text)

    if ([string]::IsNullOrWhiteSpace($Text)) {
        return ""
    }

    $clean = $Text
    $clean = $clean -replace "^\s+", ""
    $clean = $clean -replace "\s+$", ""
    $clean = $clean -replace "\[\[[^\|\]]+\|([^\]]+)\]\]", '$1'
    $clean = $clean -replace "\[\[([^\]]+)\]\]", '$1'
    $clean = $clean -replace "'''''", ""
    $clean = $clean -replace "'''", ""
    $clean = $clean -replace "''", ""
    $clean = $clean -replace "\s+", " "
    return $clean.Trim()
}

function Is-StageLine {
    param([string]$Text)

    $trimmed = $Text.Trim()
    if (-not $trimmed) { return $true }
    if ($trimmed -match '^\[.*\]$') { return $true }
    if ($trimmed -match '^\(\s*.*\s*\)$') { return $true }
    if ($trimmed -match '^==.*==$') { return $true }
    if ($trimmed -match '^\{\{.*\}\}$') { return $true }
    if ($trimmed -match '^\[\[Category:') { return $true }
    if ($trimmed -match "^:''\[.*\]''$") { return $true }
    return $false
}

function Match-SpeakerLine {
    param([string]$Line)

    $trimmed = $Line.Trim()

    $patterns = @(
        "^:'''(?<speaker>[^':]+):'''\s*(?<text>.*)$",
        "^:'''(?<speaker>[^']+)'''\s*(?:\((?<aside>[^)]*)\))?\s*:\s*(?<text>.*)$",
        "^'''(?<speaker>[^':]+):'''\s*(?<text>.*)$",
        "^'''(?<speaker>[^']+)'''\s*(?:\((?<aside>[^)]*)\))?\s*:\s*(?<text>.*)$",
        "^(?<speaker>[A-Za-z][A-Za-z .'\-]+?)\s*(?:\((?<aside>[^)]*)\))?\s*:\s*(?<text>.*)$",
        "^(?<speaker>[A-Z][A-Z0-9 .'\-]+?)\s*:\s*(?<text>.*)$"
    )

    foreach ($pattern in $patterns) {
        if ($trimmed -match $pattern) {
            return [PSCustomObject]@{
                IsMatch = $true
                Speaker = $matches['speaker'].Trim()
                Text    = $matches['text']
            }
        }
    }

    return [PSCustomObject]@{
        IsMatch = $false
        Speaker = $null
        Text    = $null
    }
}

function Is-HarrySpeaker {
    param([string]$Speaker)

    if ([string]::IsNullOrWhiteSpace($Speaker)) {
        return $false
    }

    $normalized = $Speaker.Trim()
    return $normalized -match '^(Harry|HARRY)$'
}

$allLines = [System.Collections.Generic.List[string]]::new()

Get-ChildItem $transcriptDir -File | Sort-Object Name | ForEach-Object {
    $movie = $_.Name
    $outPath = Join-Path $outputDir $movie
    $movieLines = [System.Collections.Generic.List[string]]::new()

    $currentSpeaker = $null
    $currentBuffer = [System.Collections.Generic.List[string]]::new()

    function Flush-CurrentBuffer {
        param(
            [string]$Speaker,
            [System.Collections.Generic.List[string]]$Buffer,
            [System.Collections.Generic.List[string]]$MovieLinesRef,
            [System.Collections.Generic.List[string]]$AllLinesRef,
            [string]$MovieName
        )

        if (-not (Is-HarrySpeaker $Speaker)) {
            $Buffer.Clear()
            return
        }

        $text = ($Buffer | Where-Object { $_ -and -not (Is-StageLine $_) }) -join " "
        $text = Clean-DialogueLine $text

        if ($text) {
            [void]$MovieLinesRef.Add($text)
            [void]$AllLinesRef.Add("[$MovieName] $text")
        }

        $Buffer.Clear()
    }

    Get-Content -Encoding UTF8 $_.FullName | ForEach-Object {
        $line = $_
        $speakerMatch = Match-SpeakerLine $line

        if ($speakerMatch.IsMatch) {
            Flush-CurrentBuffer -Speaker $currentSpeaker -Buffer $currentBuffer -MovieLinesRef $movieLines -AllLinesRef $allLines -MovieName $movie
            $currentSpeaker = $speakerMatch.Speaker

            $speakerText = Clean-DialogueLine $speakerMatch.Text
            if ($speakerText) {
                [void]$currentBuffer.Add($speakerText)
            }
            return
        }

        if ($null -ne $currentSpeaker) {
            $trimmed = $line.Trim()

            if (-not $trimmed) {
                Flush-CurrentBuffer -Speaker $currentSpeaker -Buffer $currentBuffer -MovieLinesRef $movieLines -AllLinesRef $allLines -MovieName $movie
                $currentSpeaker = $null
                return
            }

            if (-not (Is-StageLine $trimmed)) {
                [void]$currentBuffer.Add((Clean-DialogueLine $trimmed))
            }
        }
    }

    Flush-CurrentBuffer -Speaker $currentSpeaker -Buffer $currentBuffer -MovieLinesRef $movieLines -AllLinesRef $allLines -MovieName $movie

    [System.IO.File]::WriteAllLines($outPath, $movieLines, [System.Text.UTF8Encoding]::new($false))
    Write-Output "$movie => $($movieLines.Count) lines"
}

[System.IO.File]::WriteAllLines($combinedPath, $allLines, [System.Text.UTF8Encoding]::new($false))
Write-Output "Saved combined file => $combinedPath"
