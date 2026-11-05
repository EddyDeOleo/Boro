$outDir = 'C:\Users\User\source\repos\Boro\demo-video\out'
$files = Get-ChildItem -Path $outDir -Filter 'scene*.png'
$results = @()
foreach ($file in $files) {
    $img = [System.Drawing.Image]::FromFile($file.FullName)
    $dim = "${($img.Width)}x${($img.Height)}"
    $results += [pscustomobject]@{
        FullName = $file.FullName
        Length   = $file.Length
        Dimensions = $dim
    }
    $img.Dispose()
}
$results | ConvertTo-Json -Depth 3
