# Download torch CPU wheel in 8 parallel ranges, verify sha256, then install.
# Single connection to the mirror is ~0.03 MB/s (2h for 203MB); 8 ranges ~0.2 MB/s (~20min).
# NOTE: keep this file ASCII-only. PowerShell 5.1 reads BOM-less files as GBK and breaks on CJK.
$ErrorActionPreference = 'Continue'
$root = 'F:\heima\RAG\RAGpro_智能教育系统'
$u = 'https://mirrors.aliyun.com/pypi/packages/5f/ba/607d013b55b9fd805db2a5c2662ec7551f1910b4eef39653eeaba182c5b2/torch-2.5.1-cp312-cp312-win_amd64.whl'
$sha = '73e58e78f7d220917c5dbfad1a40e09df9929d3b95d25e57d9f8558f84c9a11c'
$total = 212926464
$parts = 8
$size = [math]::Ceiling($total / $parts)
$dir = Join-Path $root 'data\wheels\parts'
$out = Join-Path $root 'data\wheels\torch-2.5.1-cp312-cp312-win_amd64.whl'
New-Item -ItemType Directory -Force -Path $dir | Out-Null
New-Item -ItemType Directory -Force -Path (Split-Path $out) | Out-Null

$want = @()
for ($i = 0; $i -lt $parts; $i++) {
    $s = $i * $size
    $e = [math]::Min($s + $size - 1, $total - 1)
    $want += ($e - $s + 1)
}

$procs = @()
for ($i = 0; $i -lt $parts; $i++) {
    $p = Join-Path $dir "p$i"
    $have = if (Test-Path $p) { (Get-Item $p).Length } else { 0 }
    if ($have -eq $want[$i]) { Write-Host "part $i already complete"; continue }
    if (Test-Path $p) { Remove-Item $p -Force }
    $s = $i * $size
    $e = [math]::Min($s + $size - 1, $total - 1)
    $procs += Start-Process -FilePath 'curl.exe' -ArgumentList @(
        '-s', '-L', '--retry', '10', '--retry-delay', '2', '--connect-timeout', '20',
        '-r', "$s-$e", '-o', $p, $u
    ) -PassThru -WindowStyle Hidden
}
if ($procs.Count -gt 0) {
    Write-Host "downloading $($procs.Count) ranges, waiting up to 90 min..."
    $procs | Wait-Process -Timeout 5400 -ErrorAction SilentlyContinue
}

$fs = [IO.File]::Create($out)
$short = @()
for ($i = 0; $i -lt $parts; $i++) {
    $p = Join-Path $dir "p$i"
    if (-not (Test-Path $p)) { $short += $i; continue }
    $bytes = [IO.File]::ReadAllBytes($p)
    if ($bytes.Length -ne $want[$i]) { $short += $i }
    $fs.Write($bytes, 0, $bytes.Length)
    Write-Host ("part {0}: {1:N1} / {2:N1} MB" -f $i, ($bytes.Length / 1MB), ($want[$i] / 1MB))
}
$fs.Close()
if ($short.Count -gt 0) {
    Write-Host "incomplete ranges: $($short -join ',') - rerun this script"
    exit 1
}
$actual = (Get-FileHash $out -Algorithm SHA256).Hash.ToLower()
Write-Host ("total {0:N1} MB  sha256 {1}" -f ((Get-Item $out).Length / 1MB), $actual)
if ($actual -ne $sha) {
    Write-Host 'sha256 mismatch - abort'
    exit 1
}
Write-Host 'sha256 ok, installing...'
& (Join-Path $root '.venv\Scripts\python.exe') -m pip install --no-deps --force-reinstall $out 2>&1 | Select-Object -Last 3
& (Join-Path $root '.venv\Scripts\python.exe') -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"
