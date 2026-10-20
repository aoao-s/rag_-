# 导出镜像（联网机器上执行一次）

# 把 compose 用到的镜像打包成 tar，拷到离线机器后用 `load.ps1` 导入。
# 只导数据面（mysql + redis）约 1.3 GB；`-Full` 再加 etcd/minio/milvus/attu 约 2 GB。

# 用法：
#   .\save.ps1            # mysql + redis
#   .\save.ps1 -Full      # 全量

param([switch]$Full)

# docker 会把进度写到 stderr，PS 5.1 会当错误记录；改为手动判 exit code
$ErrorActionPreference = 'Continue'
$outDir = Join-Path $PSScriptRoot 'images'
New-Item -ItemType Directory -Force -Path $outDir | Out-Null

$targets = if ($Full) {
    @(
        @{ ref = 'mysql:8.0';                                       file = 'mysql-8.0.tar' },
        @{ ref = 'redis:8.10';                                      file = 'redis-8.10.tar' },
        @{ ref = 'quay.io/coreos/etcd:v3.5.5';                      file = 'etcd-v3.5.5.tar' },
        @{ ref = 'minio/minio:RELEASE.2023-03-20T20-16-18Z';         file = 'minio.tar' },
        @{ ref = 'milvusdb/milvus:v2.4.0';                          file = 'milvus-v2.4.0.tar' },
        @{ ref = 'zilliz/attu:v2.4.12';                             file = 'attu-v2.4.12.tar' }
    )
} else {
    @(
        @{ ref = 'mysql:8.0';  file = 'mysql-8.0.tar' },
        @{ ref = 'redis:8.10'; file = 'redis-8.10.tar' }
    )
}

foreach ($t in $targets) {
    $path = Join-Path $outDir $t.file
    Write-Host "[*] docker save $($t.ref) -> $($t.file)" -ForegroundColor Cyan
    & docker image inspect $t.ref *> $null
    if ($LASTEXITCODE -ne 0) { throw "本机没有镜像 $($t.ref)，先 docker pull $($t.ref)" }
    & docker save -o $path $t.ref
    if ($LASTEXITCODE -ne 0) { throw "docker save 失败: $($t.ref)" }
}

Get-ChildItem $outDir | ForEach-Object {
    '{0,-24} {1,8:N1} MB' -f $_.Name, ($_.Length / 1MB)
}
Write-Host '[ok] 把 images/ 整个目录拷到离线机器，执行 load.ps1' -ForegroundColor Green
