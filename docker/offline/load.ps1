# 导入镜像（离线机器上执行）

# 用法：
#   .\load.ps1            # 导入 images/ 下所有 tar
#   .\load.ps1 -Only mysql-8.0.tar,redis-8.10.tar

# 导入完直接 `scripts\docker_up.ps1`——compose 里每个服务都写了 `pull_policy: never`，
# 镜像不在本地会立刻报错，而不是偷偷去联网拉（离线环境里那种等待会卡很久）。

param([string[]]$Only)

# docker 会把进度写到 stderr，PS 5.1 会当错误记录；改为手动判 exit code
$ErrorActionPreference = 'Continue'
$imgDir = Join-Path $PSScriptRoot 'images'

if (-not (Test-Path $imgDir)) { throw "找不到 $imgDir，先把导出机的 images/ 目录拷过来" }

$files = Get-ChildItem $imgDir -Filter '*.tar' | Where-Object { -not $Only -or $Only -contains $_.Name }
if (-not $files) { throw "没有可导入的 tar（$imgDir）" }

foreach ($f in $files) {
    Write-Host "[*] docker load $($f.Name)" -ForegroundColor Cyan
    & docker load -i $f.FullName
    if ($LASTEXITCODE -ne 0) { throw "docker load 失败: $($f.Name)" }
}

Write-Host ''
Write-Host '[*] 当前镜像：' -ForegroundColor Cyan
# & docker images --format '{{.Repository}}:{{.Tag}} ({{.Size}})'
Write-Host '[ok] 现在可以跑 scripts\docker_up.ps1' -ForegroundColor Green
