$outDir = 'C:\Users\User\source\repos\Boro\demo-video\out'
$contactPath = Join-Path $outDir 'brand_review_contact_sheet.png'
Add-Type -AssemblyName System.Drawing
$images = @()
for ($i = 1; $i -le 8; $i++) {
    $path = Join-Path $outDir "scene$($i).png"
    if (-Not (Test-Path $path)) { throw "Missing $path" }
    $img = [System.Drawing.Image]::FromFile($path)
    $images += $img
}
$w = [int]$images[0].Width
$h = [int]$images[0].Height
Write-Output "Width=$w type=$($w.GetType())"
Write-Output "Height=$h type=$($h.GetType())"
$bitmap = [System.Drawing.Bitmap]::new($w * 4, $h * 2)
$g = [System.Drawing.Graphics]::FromImage($bitmap)
$g.Clear([System.Drawing.Color]::Black)
for ($idx = 0; $idx -lt $images.Count; $idx++) {
    $row = [int][Math]::Floor($idx / 4)
    $col = $idx % 4
    $destRect = New-Object System.Drawing.Rectangle ($col * $w, $row * $h, $w, $h)
    $g.DrawImage($images[$idx], $destRect)
}
$bitmap.Save($contactPath, [System.Drawing.Imaging.ImageFormat]::Png)
Write-Output "Contact sheet saved to $contactPath"
