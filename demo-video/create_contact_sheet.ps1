$outDir = 'C:\Users\User\source\repos\Boro\demo-video\out'
$contactPath = Join-Path $outDir 'brand_review_contact_sheet.png'

# Load System.Drawing assembly
Add-Type -AssemblyName System.Drawing

# Load images
$images = @()
for ($i = 1; $i -le 8; $i++) {
    $path = Join-Path $outDir "scene$($i).png"
    if (-Not (Test-Path $path)) { throw "Missing $path" }
    $img = [System.Drawing.Image]::FromFile($path)
    $images += $img
}
# Assume all same size
$w = $images[0].Width
$h = $images[0].Height
# Create bitmap 4x2 grid
$bitmap = New-Object System.Drawing.Bitmap ($w*4, $h*2)
$g = [System.Drawing.Graphics]::FromImage($bitmap)
$g.Clear([System.Drawing.Color]::Black)
for ($idx=0; $idx -lt $images.Count; $idx++) {
    $row = [Math]::Floor($idx/4)
    $col = $idx % 4
    $destRect = New-Object System.Drawing.Rectangle ($col*$w, $row*$h, $w, $h)
    $g.DrawImage($images[$idx], $destRect)
}
$bitmap.Save($contactPath, [System.Drawing.Imaging.ImageFormat]::Png)
Write-Output "Contact sheet saved to $contactPath"
