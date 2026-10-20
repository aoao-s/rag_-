"""EduRAG-Pro 记忆模块架构图（07）— Pillow 绘制，2x 超采样"""
import math
import os
from PIL import Image, ImageDraw, ImageFont

S = 2
W, H = 2000, 1140
OUT_DIR = r'F:\heima\RAG\RAGpro_智能教育系统\docs\diagrams'

INK = (31, 41, 55)
SUB = (90, 100, 116)
PANEL = (255, 255, 255)
BG = (247, 249, 252)

PURPLE_F, PURPLE_B = (245, 242, 255), (124, 92, 214)
BLUE_F, BLUE_B = (238, 245, 255), (59, 130, 246)
GREEN_F, GREEN_B = (236, 250, 242), (34, 160, 96)
AMBER_F, AMBER_B = (255, 248, 232), (222, 148, 24)
RED_F, RED_B = (255, 242, 242), (222, 76, 76)
TEAL_F, TEAL_B = (235, 250, 250), (20, 160, 170)
GRAY_F, GRAY_B = (244, 246, 250), (148, 158, 172)

FONT_REG = r'C:\Windows\Fonts\msyh.ttc'
FONT_BOLD = r'C:\Windows\Fonts\msyhbd.ttc'
_cache = {}


def font(size, bold=False):
    key = (size, bold)
    if key not in _cache:
        _cache[key] = ImageFont.truetype(FONT_BOLD if bold else FONT_REG, size * S)
    return _cache[key]


def tw(d, s, f):
    b = d.textbbox((0, 0), s, font=f)
    return (b[2] - b[0]) / S


def text(d, x, y, s, size=18, bold=False, color=INK, anchor='la', maxw=None):
    f = font(size, bold)
    if maxw:
        while tw(d, s, f) > maxw and size > 10:
            size -= 1
            f = font(size, bold)
    d.text((x * S, y * S), s, font=f, fill=color, anchor=anchor)


def rrect(d, x, y, w, h, r, fill, outline, width=2):
    d.rounded_rectangle([x * S, y * S, (x + w) * S, (y + h) * S], radius=r * S,
                        fill=fill, outline=outline, width=int(width * S))


def arrow(d, pts, color, width=2.5, head=9, dashed=False):
    pts = [(p[0] * S, p[1] * S) for p in pts]
    if dashed:
        for i in range(len(pts) - 1):
            (x1, y1), (x2, y2) = pts[i], pts[i + 1]
            dist = ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5
            seg = 10 * S
            t = 0
            while t < dist:
                e = min(t + seg, dist)
                d.line([(x1 + (x2 - x1) * t / dist, y1 + (y2 - y1) * t / dist),
                        (x1 + (x2 - x1) * e / dist, y1 + (y2 - y1) * e / dist)],
                       fill=color, width=int(width * S))
                t = e + 6 * S
    else:
        d.line(pts, fill=color, width=int(width * S), joint='curve')
    (x1, y1), (x2, y2) = pts[-2], pts[-1]
    ang = math.atan2(y2 - y1, x2 - x1)
    hd = head * S
    a1 = (x2 - hd * math.cos(ang - 0.45), y2 - hd * math.sin(ang - 0.45))
    a2 = (x2 - hd * math.cos(ang + 0.45), y2 - hd * math.sin(ang + 0.45))
    d.polygon([(x2, y2), a1, a2], fill=color)


def box(d, x, y, w, h, title, lines, fill, line, ts=19, bs=14, rt=None):
    rrect(d, x, y, w, h, 10, fill, line, 2)
    d.rectangle([x * S, (y + 8) * S, (x + 5) * S, (y + h - 8) * S], fill=line)
    text(d, x + 20, y + 12, title, ts, True, INK, maxw=w - 140)
    if rt:
        text(d, x + w - 18, y + 14, rt, ts - 3, False, line, anchor='ra')
    cy = y + 10 + ts + 6
    for ln in lines:
        text(d, x + 20, cy, ln, bs, False, SUB, maxw=w - 34)
        cy += bs + 6


img = Image.new('RGB', (W * S, H * S), BG)
d = ImageDraw.Draw(img)
d.rectangle([0, 0, W * S, 8 * S], fill=(109, 40, 217))

# ---------- 头部 ----------
text(d, 56, 38, 'EduRAG-Pro 记忆模块：四层记忆 · 写入异步 · 召回并行 · 可解释可删除', 33, True, INK)
text(d, 58, 88, '每个用户一个私有助手（tenant_id + user_id 唯一编号）；单窗口 40 轮 / 28K token，超出先压缩再装配；请求路径只读不写', 17, False, SUB)

# ---------- 接入层 ----------
PH = 84
text(d, 56, 120, '接入层', 15, True, (109, 40, 217))
bw = (1888 - 2 * 16) / 3
for i, (t, l1, l2, f, b) in enumerate([
    ('学员前端', '对话页 + 我的记忆页（查看/编辑/删除/导出）', '删除 2s 内对召回生效', PURPLE_F, PURPLE_B),
    ('API 网关', 'JWT · 租户隔离 · 限流', '所有记忆读写带 request_id 与审计', BLUE_F, BLUE_B),
    ('顾问工作台', '授权后只读学员记忆', 'memory_grant 30 天，撤销即失效', TEAL_F, TEAL_B),
]):
    x = 56 + i * (bw + 16)
    rrect(d, x, 144, bw, PH, 10, f, b, 2)
    text(d, x + 20, 154, t, 21, True, INK)
    text(d, x + 20, 184, l1, 15, False, SUB, maxw=bw - 36)
    text(d, x + 20, 205, l2, 15, False, SUB, maxw=bw - 36)

# ---------- 三列标题 ----------
text(d, 56, 240, '① 写入链路（异步 · p95 ≤1.5s）', 17, True, (34, 160, 96))
text(d, 560, 240, '② 四层记忆（越往上越接近结论、越省 token）', 17, True, (109, 40, 217))
text(d, 1210, 240, '③ 召回链路（与主检索并行 · p95 ≤120ms）', 17, True, (59, 130, 246))

# ---------- 左：写入链路 ----------
LX, LW = 56, 468
write = [
    ('① 门控', ['规则：数字 / 日期 / 课程名 / “我要·改·不去”', '寒暄与重复提问直接丢弃（约 40% 轮次）'], '30 ms', GREEN_F, GREEN_B),
    ('② 抽取', ['deepseek-flash + JSON Schema 强约束', '偏好 / 身份 / 约束 / 决定 / 进度 五类'], '400–800 ms', GREEN_F, GREEN_B),
    ('③ 去重', ['Milvus 向量相似度 > 0.92 合并', '保留信息量更大的表述'], '20 ms', GREEN_F, GREEN_B),
    ('④ 冲突消解', ['科目 / 金额 / 时间冲突规则初筛', '疑似冲突再让 LLM 判定，命中即 supersede'], '200 ms', AMBER_F, AMBER_B),
    ('⑤ 摘要滚动', ['每 10 轮或 3K token 增量摘要', '保留 topics 与 open_items'], '600 ms', GREEN_F, GREEN_B),
    ('⑥ 落库 + 向量化', ['MySQL 事务写明细 → Milvus 异步写（可补偿）', 'Redis 热点失效 · 写审计'], '30 ms', BLUE_F, BLUE_B),
]
yh, gy = 86, 11
for i, (t, ls, rt, f, b) in enumerate(write):
    y = 270 + i * (yh + gy)
    box(d, LX, y, LW, yh, t, ls, f, b, 19, 14, rt)
    if i < len(write) - 1:
        arrow(d, [(LX + LW / 2, y + yh + 1), (LX + LW / 2, y + yh + gy - 2)], (34, 160, 96), 2.5, 7)
text(d, LX, 270 + 6 * (yh + gy) + 2, '队列：Redis Stream（消费组 worker，失败 3 次进死信）', 15, False, SUB)

# ---------- 中：四层记忆 ----------
CX, CW = 560, 616
layers = [
    ('L3 用户画像', ['身份 / 目标 / 水平 / 预算 / 地域', 'MySQL user_profile + Redis 常驻', '注入 300 token · 长期有效'], PURPLE_F, PURPLE_B),
    ('L2 原子记忆', ['跨会话事实与偏好，带 structured 字段与来源轮次', 'Milvus edurag_memory（user_id 分区）+ MySQL memory_atom', '注入 300 · 每用户 ≤500 条活跃 · TTL 90–180d'], TEAL_F, TEAL_B),
    ('L1 会话摘要', ['每个 session 一份滚动摘要：时间 / 话题 / 结论 / 未决事项', 'MySQL session_summary + Redis 热点', '注入 400 token · 保留 90 天'], GREEN_F, GREEN_B),
    ('L0 工作记忆', ['当前会话逐字对话（原话）', 'Redis LIST（热）+ MySQL chat_message（真相源）', '40 轮 / 28K token 双上限 · 超出压缩并归档原文'], BLUE_F, BLUE_B),
]
ch, cg = 132, 16
for i, (t, ls, f, b) in enumerate(layers):
    y = 270 + i * (ch + cg)
    box(d, CX, y, CW, ch, t, ls, f, b, 21, 14)

# ---------- 右：召回链路 ----------
RX, RW = 1210, 734
recall = [
    ('⑦ 生成回答', ['记忆只提供背景，不作为引用来源', '引用仍可溯源到 parent_id → 文件 + 页码'], BLUE_F, BLUE_B),
    ('⑥ 上下文装配 ≤28K', ['系统 600 → 画像 300 → 摘要 400 → 原子 300 → 知识 6000', '逐字历史吃剩余；固定段在前以命中前缀缓存'], PURPLE_F, PURPLE_B),
    ('⑤ RRF 融合（不做阈值硬过滤）', ['B/C 两路按排名融合；中文短 query 余弦分布扁平', '硬阈值会误杀，改用置信度 + 时效排序'], PURPLE_F, PURPLE_B),
    ('④ 逐字历史（L0）', ['当前 session 最近若干轮', 'Redis LIST，miss 回源 MySQL'], BLUE_F, BLUE_B),
    ('③ 会话摘要（L1）', ['当前 session + 最近 3 个 session', 'MySQL（Redis 热点），超时只用当前会话'], GREEN_F, GREEN_B),
    ('② 原子记忆（L2）', ['query 向量 top5 · 只看 valid_to is null', 'confidence ≥0.6 · 已删除 / 已 supersede 永不注入'], TEAL_F, TEAL_B),
    ('① 画像直读（L3）', ['按 user_id 直读 Redis → MySQL', '必然命中，不降级'], PURPLE_F, PURPLE_B),
]
rh, rg = 80, 8
ys = [270 + i * (rh + rg) for i in range(len(recall))]
for i, (t, ls, f, b) in enumerate(recall):
    box(d, RX, ys[i], RW, rh, t, ls, f, b, 19, 14)
for i in range(len(recall) - 1):
    arrow(d, [(RX + 70, ys[i + 1] - 1), (RX + 70, ys[i] + rh + 1)], (59, 130, 246), 2.5, 7)

# ---------- 底部：存储与治理 ----------
BY = 906
rrect(d, 56, BY, 1888, 196, 12, PANEL, GRAY_B, 2)
text(d, 76, BY + 12, '④ 存储与治理', 17, True, (109, 40, 217))
text(d, 210, BY + 13, '请求路径只读；写入、压缩、衰减、清理全部离线或异步', 15, False, SUB)

store = [
    ('MySQL · 真相源', ['chat_message / session_summary', 'memory_atom / user_profile', 'memory_grant / memory_audit', 'qa_pair / tag 体系'], BLUE_F, BLUE_B),
    ('Redis · 热与队列', ['session:{sid} 窗口 LIST', 'mem:hot:{user_id} 画像热点', 'Stream 抽取 / 摘要队列', '限流 · 分布式锁 · 单飞'], RED_F, RED_B),
    ('Milvus · 语义层', ['edurag_memory（user_id 分区）', '只存活跃记忆，supersede 即逻辑删', '与知识库共用实例、独立 collection', '标量过滤前置：tenant / valid_to'], TEAL_F, TEAL_B),
    ('治理规则', ['TTL + 衰减：score <0.2 归档', '软删 30 天后物理清理', '删除链：revive / purge_chain 二选一', '审计：view / edit / delete / export'], AMBER_F, AMBER_B),
]
sw = (1888 - 2 * 20 - 3 * 16) / 4
for i, (t, ls, f, b) in enumerate(store):
    x = 76 + i * (sw + 16)
    rrect(d, x, BY + 46, sw, 134, 9, f, b, 2)
    text(d, x + 14, BY + 56, t, 18, True, INK, maxw=sw - 28)
    for j, ln in enumerate(ls):
        text(d, x + 14, BY + 82 + j * 23, ln, 14, False, SUB, maxw=sw - 28)

os.makedirs(OUT_DIR, exist_ok=True)
out = os.path.join(OUT_DIR, '07_memory_architecture.png')
img = img.resize((W, H), Image.LANCZOS)
img.save(out, optimize=True)
print('saved', out)

