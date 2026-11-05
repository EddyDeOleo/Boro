# PowerShell script to render representative still frames for each of the 9 scenes
# and assemble them into a contact sheet.
# Requires ImageMagick's `magick` command to be available in PATH.

$projectRoot = "C:/Users/User/source/repos/Boro/demo-video"
$compositionId = "BoroNewDemo"
$totalFrames = 5400
$stages = 9
$stageDuration = [math]::Floor($totalFrames / $stages)

# Ensure output directory exists
$outDir = Join-Path $projectRoot "out"
if (-not (Test-Path $outDir)) { New-Item -ItemType Directory -Path $outDir | Out-Null }

# Render stills at the middle of each stage
for ($i = 0; $i -lt $stages; $i++) {
    $start = $i * $stageDuration
    $midFrame = $start + [math]::Floor($stageDuration / 2)
    $outputFile = Join-Path $outDir "scene${i}_still.png"
    Write-Host "Rendering frame $midFrame to $outputFile"
    npx remotion still $projectRoot/src/NewDemo/Video.tsx --frame $midFrame --output $outputFile
}

# Assemble contact sheet (3 columns x 3 rows)
$contactSheet = Join-Path $outDir "new_contact_sheet.png"
magick convert (Join-Path $outDir "scene*_still.png") -bordercolor white -border 2 -resize 640x360 -background black -gravity center -append $contactSheet

Write-Host "Contact sheet generated at $contactSheet"
