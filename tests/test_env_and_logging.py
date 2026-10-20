"""回归测试：两处「换个环境就炸」的坑。

1. `base.config` 的 `.env` 必须是绝对路径——相对路径会跟着工作目录走，
   在 PyCharm 里直接 run、或从别的目录调用时读不到配置，表现为
   `Access denied ... (using password: NO)`，与真正的原因（没有 .env）无关。
2. 没有 structlog 的环境里 `get_logger()` 返回的适配器必须能吃 `key=value` 关键字，
   否则异常处理分支里的 `log.warning("...", error=str(exc))` 会二次崩溃，
   把真正的错误盖掉。
"""

from __future__ import annotations

import logging

from base.config import BASE_DIR, ENV_FILE, Settings
from base.logger import _StdlibAdapter, get_logger


def test_env_file_is_absolute_and_in_repo_root() -> None:
    assert ENV_FILE.is_absolute()
    assert ENV_FILE.parent == BASE_DIR
    assert ENV_FILE.name == ".env"


def test_settings_read_env_from_any_cwd(tmp_path, monkeypatch) -> None:
    """切到临时目录后仍应读到仓库根的 .env（而不是 CWD 下的 .env）。"""
    monkeypatch.chdir(tmp_path)
    settings = Settings()
    assert settings.model_config["env_file"] == str(ENV_FILE)


def test_stdlib_adapter_accepts_keyword_fields() -> None:
    adapter = _StdlibAdapter(logging.getLogger("edurag.test.adapter"))
    # structlog 风格：事件名 + 关键字字段，不能抛 TypeError
    adapter.info("faq_loaded", docs=2000, cost_ms=12.5)
    adapter.warning("mysql_ping_failed", error="(1045, 'Access denied')")
    adapter.error("boom", exc_info=False)


def test_get_logger_supports_kwargs_regardless_of_structlog() -> None:
    log = get_logger("edurag.test.normal")
    log.info("hello", answer=42)
    log.warning("degraded", reason="no_db")
