# 停掉离线依赖服务
#
# 用法：
#   scripts\docker_down.ps1          # 停容器，保数据（docker/volumes 不动）
#   scripts\docker_down.ps1 -Purge   # 连数据卷一起删（⚠ 会清空 MySQL/Redis 数据）

param([switch]$Purge)

# docker 会把进度写到 stderr，PS 5.1 会当错误记录；改为手动判 exit code
$ErrorActionPreference = 'Continue'
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

if ($Purge) {
    $answer = Read-Host '确认删除 docker/volumes 下的所有数据？输入 yes 继续'
    if ($answer -ne 'yes') { Write-Host '已取消'; exit 1 }
    & docker compose --env-file .env -f docker/docker-compose.yml down -v
} else {
    & docker compose --env-file .env -f docker/docker-compose.yml down
}
