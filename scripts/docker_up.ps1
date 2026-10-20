# 启动离线依赖服务（MySQL + Redis）
#
# 用法：
#   scripts\docker_up.ps1              # 只起数据面（mysql + redis），PR1/PR2 够用
#   scripts\docker_up.ps1 -Full        # 连 etcd/minio/milvus/attu 一起起（PR3 需要）
#
# 说明：统一在这里加 --env-file，保证 compose 与 python 应用读同一份 .env，
#      不会出现「compose 里是 123456、.env 里是别的」这种对不上的问题。
#
# 注意：docker 把进度写到 stderr，Windows PowerShell 5.1 会当成 NativeCommandError，
#      所以这里不用 $ErrorActionPreference='Stop'，而是手动判断 $LASTEXITCODE。

param([switch]$Full)

$ErrorActionPreference = 'Continue'
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

if (-not (Test-Path '.env')) {
    Copy-Item '.env.example' '.env'
    Write-Host '[!] 未找到 .env，已从 .env.example 复制，请先按需修改 MYSQL_PASSWORD' -ForegroundColor Yellow
}

$services = @()
if (-not $Full) { $services = @('mysql', 'redis') }
$composeArgs = @('compose', '--env-file', '.env', '-f', 'docker/docker-compose.yml', 'up', '-d') + $services

Write-Host "[*] docker $($composeArgs -join ' ')" -ForegroundColor Cyan
& docker @composeArgs 2>&1 | ForEach-Object { Write-Host "    $_" }
if ($LASTEXITCODE -ne 0) { throw "docker compose up 失败（exit $LASTEXITCODE）" }

Write-Host '[*] 等待健康检查（最多 180 秒）...' -ForegroundColor Cyan
$deadline = (Get-Date).AddSeconds(180)
do {
    Start-Sleep -Seconds 3
    $lines = & docker ps --filter 'name=edurag-' --format '{{.Names}}={{.State}}/{{.Status}}' 2>$null
    $pending = @($lines | Where-Object { $_ -match 'health: starting|Created|Exited' })
} while ($pending.Count -gt 0 -and (Get-Date) -lt $deadline)

$lines | ForEach-Object { Write-Host "    $_" }
if ($pending.Count -gt 0) {
    Write-Host '[!] 仍有服务未就绪，用 docker logs edurag-mysql 排查' -ForegroundColor Yellow
    exit 1
}

Write-Host ''
Write-Host '下一步：' -ForegroundColor Green
Write-Host '  .\.venv\Scripts\python.exe scripts\init_db.py'
Write-Host '  .\.venv\Scripts\python.exe scripts\import_faq.py --publish'
Write-Host '  .\.venv\Scripts\python.exe main.py'
