# 离线数据面：MySQL + Redis

本项目的数据面（MySQL 真相源 + Redis 缓存）可以在**完全断网**的机器上跑起来。
整条链路的约束是：镜像只能来自 `docker save` 的 tar，compose 不允许联网拉取。

## 1. 端口与账号：只改一处

所有端口、账号、库名都写在仓库根 `.env`，compose 用 `--env-file .env` 读取同一份：

| 变量 | 默认 | 说明 |
|---|---|---|
| `MYSQL_PORT` | 3307 | 宿主机端口，刻意避开 3306（旧项目 rag-pro 的容器占着） |
| `MYSQL_PASSWORD` | 637456 | root 密码，容器与应用共用 |
| `MYSQL_DATABASE` | RAG_abc | 启动时自动建库 |
| `REDIS_PORT` | 6380 | 宿主机端口，避开 6379 |
| `REDIS_PASSWORD` | 1234 | `--requirepass`，应用必须一致 |

改端口只改 `.env` 一处，别去动 `docker/docker-compose.yml`。
如果本机 3307/6380 也被占，改这两个变量即可。

## 2. 联网机器：导出镜像

```powershell
.\docker\offline\save.ps1        # mysql + redis，约 1.3 GB
.\docker\offline\save.ps1 -Full  # 再加 etcd/minio/milvus/attu，约 2 GB（PR3 用）
```

产物在 `docker/offline/images/*.tar`，已加入 `.gitignore`（走移动硬盘 / 内网传输，
不进版本库）。

## 3. 离线机器：导入并启动

```powershell
.\docker\offline\load.ps1        # docker load 所有 tar
.\scripts\docker_up.ps1          # 起 mysql + redis，等健康检查通过
```

`docker-compose.yml` 每个服务都写了 `pull_policy: never`：镜像不在本地会**立刻报错**，
而不是静默联网拉取（离线环境下那种等待会卡很久，报错比等待好）。

## 4. 初始化数据

```powershell
.\.venv\Scripts\python.exe scripts\init_db.py            # 建表（幂等）
.\.venv\Scripts\python.exe scripts\import_faq.py --publish  # CSV → faq，发布才进检索语料
.\.venv\Scripts\python.exe main.py                       # 启动 API
```

验证：

```powershell
curl http://127.0.0.1:8000/readyz
# {"status":"ready","degraded":false,"faq_docs":2000}
```

`faq_docs` 与 `SELECT COUNT(*) FROM faq WHERE status='published'` 必须一致；
不一致说明服务还在读 CSV（检查 `.env` 的 `FAQ_CORPUS_CSV` 是否为空）。

## 5. 常用运维

```powershell
docker ps --filter name=edurag-                       # 看状态
docker logs -f edurag-mysql                            # 看日志
docker exec -it edurag-mysql mysql -uroot -p          # 进库（密码见 .env）
docker exec -it edurag-redis redis-cli -a <REDIS_PASSWORD>  # 进 Redis

.\scripts\docker_down.ps1          # 停容器，保数据
.\scripts\docker_down.ps1 -Purge   # ⚠ 连数据一起删，需输入 yes 确认
```

数据落在 `docker/volumes/{mysql,redis}`，`down` 不会删，只有 `down -v` 或 `-Purge` 才删。

## 6. 已知坑

- **MySQL 首次启动要 30–60 s**：`health: starting` 期间连不上是正常的，
  入口点先初始化数据目录、再重启一次服务。`docker_up.ps1` 会等到 healthy 才返回。
- **Redis 必须带密码连**：没设密码会报 `HELLO must be called with the client already
  authenticated`，看起来像"连不上"，其实是认证失败。
- **`--env-file` 不能省**：compose 默认读 `docker/.env`，而我们唯一配置源是仓库根 `.env`。
  直接 `docker compose -f docker/docker-compose.yml up` 会用默认值（123456/itcast/3307），
  和你 `.env` 里的不一致。这也是为什么脚本里统一封装了 `--env-file`。
- **不要用 `docker compose restart` 改配置**：环境变量在创建容器时固化，
  改完 `.env` 要 `docker_up.ps1`（等价 `up -d --force-recreate`）才生效。
