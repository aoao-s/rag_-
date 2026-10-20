# -*- coding: utf-8 -*-
"""生成 EduRAG-Pro 1.41 目标架构图（Pillow 绘制，2x 超采样后缩放，中文使用微软雅黑）"""
import os

from PIL import Image, ImageDraw, ImageFont

S = 2
FONT_REG = 'C:/Windows/Fonts/msyh.ttc'
FONT_BOLD = 'C:/Windows/Fonts/msyhbd.ttc'
OUT_DIR = 'F:/heima/RAG/RAGpro_智能教育系统/docs/diagrams'
NAME = '06_edurag_pro_architecture.png'

TXT = (31, 41, 55)
SUB = (100, 116, 139)
BODY = (71, 85, 105)

BLUE_F, BLUE_B = (232, 241, 254), (59, 130, 246)
GREEN_F, GREEN_B = (234, 247, 239), (16, 185, 129)
AMBER_F, AMBER_B = (255, 247, 237), (245, 158, 11)
PURPLE_F, PURPLE_B = (245, 243, 255), (139, 92, 246)
CYAN_F, CYAN_B = (224, 242, 254), (14, 165, 233)
RED_F, RED_B = (254, 242, 242), (239, 68, 68)
GRAY_F, GRAY_B = (241, 245, 249), (148, 163, 184)
WHITE = (255, 255, 255)


def font(size, bold=False):
    return ImageFont.truetype(FONT_BOLD if bold else FONT_REG, int(size * S))


def new_canvas(w, h, bg=(248, 250, 252)):
    img = Image.new('RGB', (w * S, h * S), bg)
    return img, ImageDraw.Draw(img)


def save(img, w, h, name):
    img = img.resize((w, h), Image.LANCZOS)
    os.makedirs(OUT_DIR, exist_ok=True)
    path = os.path.join(OUT_DIR, name)
    img.save(path)
    print('saved', path)


def rr(d, x, y, w, h, fill, outline=None, r=14, width=3):
    d.rounded_rectangle([x * S, y * S, (x + w) * S, (y + h) * S],
                        radius=r * S, fill=fill, outline=outline,
                        width=int(width * S) if outline else 0)


def wrap(d, s, f, max_w):
    lines = []
    for para in str(s).split('\n'):
        if not para:
            lines.append('')
            continue
        cur = ''
        for ch in para:
            if d.textlength(cur + ch, font=f) <= max_w * S:
                cur += ch
            else:
                lines.append(cur)
                cur = ch
        lines.append(cur)
    return lines


def node(d, x, y, w, h, title, body=None, fill=GREEN_F, border=GREEN_B,
         title_size=20, body_size=15, body_color=BODY):
    rr(d, x, y, w, h, fill, border)
    tf, bf = font(title_size, True), font(body_size)
    tlines = wrap(d, title, tf, w - 24)
    blines = []
    for b in (body or []):
        blines += wrap(d, b, bf, w - 24)
    lh_t = title_size * 1.35 * S
    lh_b = body_size * 1.5 * S
    gap = 8 * S if blines else 0
    total = len(tlines) * lh_t + gap + len(blines) * lh_b
    ty = y * S + (h * S - total) / 2
    for ln in tlines:
        tw = d.textlength(ln, font=tf)
        d.text(((x + w / 2) * S - tw / 2, ty), ln, font=tf, fill=TXT)
        ty += lh_t
    ty += gap
    for ln in blines:
        tw = d.textlength(ln, font=bf)
        d.text(((x + w / 2) * S - tw / 2, ty), ln, font=bf, fill=body_color)
        ty += lh_b


def text(d, x, y, s, size=20, bold=False, color=TXT, anchor=None):
    f = font(size, bold)
    if anchor:
        d.text((x * S, y * S), s, font=f, fill=color, anchor=anchor)
    else:
        d.text((x * S, y * S), s, font=f, fill=color)


def arrow(d, pts, color=(71, 85, 105), width=3, head=11, dashed=False):
    p = [(x * S, y * S) for x, y in pts]
    if dashed:
        for i in range(len(p) - 1):
            x1, y1 = p[i]
            x2, y2 = p[i + 1]
            dist = ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5
            if dist == 0:
                continue
            ux, uy = (x2 - x1) / dist, (y2 - y1) / dist
            t = 0.0
            while t < dist:
                e = min(t + 11 * S, dist)
                d.line([(x1 + ux * t, y1 + uy * t), (x1 + ux * e, y1 + uy * e)],
                       fill=color, width=max(1, int(width * S)))
                t = e + 8 * S
    else:
        d.line(p, fill=color, width=max(1, int(width * S)), joint='curve')
    (x1, y1), (x2, y2) = p[-2], p[-1]
    import math
    ang = math.atan2(y2 - y1, x2 - x1)
    hh = head * S
    a = (x2 - hh * math.cos(ang - 0.42), y2 - hh * math.sin(ang - 0.42))
    b = (x2 - hh * math.cos(ang + 0.42), y2 - hh * math.sin(ang + 0.42))
    d.polygon([(x2, y2), a, b], fill=color)


def tag(d, x, y, s, size=16, fg=(51, 65, 85), bg=(255, 255, 255), pad=6):
    f = font(size, True)
    tw = d.textlength(s, font=f)
    d.rounded_rectangle([x * S - pad * S, (y - 2) * S,
                         x * S + tw + pad * S, (y + size * 1.5) * S],
                        radius=5 * S, fill=bg)
    d.text((x * S, y * S), s, font=f, fill=fg)


W, H = 2000, 1180
img, d = new_canvas(W, H)

text(d, 44, 22, 'EduRAG-Pro 1.41 目标架构：分层解耦 · 编排可控 · 全链路可观测', 36, True)
text(d, 46, 74, '现状是一条写死的 CLI 流水线；目标是“接入 / 编排 / 能力 / 数据”四层解耦，离线层与观测层补齐闭环', 19, False, SUB)

PANEL_X, PANEL_W = 40, 1380
IN_X, IN_W = 64, 1332
CX = PANEL_X + PANEL_W // 2

LAYERS = [
    ('① 接入层  Access', 124, 128, BLUE_F, BLUE_B),
    ('② 编排层  Orchestration（LangGraph 状态机）', 276, 196, PURPLE_F, PURPLE_B),
    ('③ 能力层  Capability（可独立部署、可替换）', 496, 146, GREEN_F, GREEN_B),
    ('④ 数据层  Storage', 666, 128, CYAN_F, CYAN_B),
    ('⑤ 离线层  Ingestion & Evaluation', 818, 128, AMBER_F, AMBER_B),
    ('⑥ 观测层  Observability', 970, 104, GRAY_F, GRAY_B),
]

for label, y, h, fill, border in LAYERS:
    rr(d, PANEL_X, y, PANEL_W, h, fill, border, r=18, width=2)
    text(d, PANEL_X + 16, y + 9, label, 19, True, border)


def row(boxes, y, h, x0=IN_X, total=IN_W, gap=16):
    n = len(boxes)
    w = (total - gap * (n - 1)) / n
    out = []
    for i, (t, b, f, o) in enumerate(boxes):
        x = x0 + i * (w + gap)
        node(d, x, y, w, h, t, b, f, o, 19, 14)
        out.append((x, y, w, h))
    return out


# ---------------- ① 接入层
b1 = row([
    ('Web / App 前端', ['会话式问答', '流式渲染 + 引用卡片'], GRAY_F, GRAY_B),
    ('API 网关', ['JWT 鉴权 · 租户限流', '灰度 · 审计'], BLUE_F, BLUE_B),
    ('FastAPI 服务', ['WS / SSE 流式 · 断线续传', 'request_id 幂等 · 优雅停机'], BLUE_F, BLUE_B),
    ('人工客服工作台', ['转人工工单', '知识纠错回填'], GRAY_F, GRAY_B),
], 162, 84)

# ---------------- ② 编排层
chips_1 = ['① 归一化·敏感词', '② 会话改写', '③ 多级缓存', '④ 意图+槽位', '⑤ 策略规划']
chips_2 = ['⑥ 并行检索', '⑦ 融合重排', '⑧ 证据门', '⑨ 流式生成', '⑩ 引用·校验']
CW, CG = 255, 14
cy1, cy2, ch = 316, 388, 52

for i, t in enumerate(chips_1):
    x = IN_X + i * (CW + CG)
    node(d, x, cy1, CW, ch, t, None, WHITE, PURPLE_B, 19, 14)
    if i:
        arrow(d, [(x - CG + 1, cy1 + ch / 2), (x - 2, cy1 + ch / 2)], PURPLE_B, 2.4, 7)

for i, t in enumerate(chips_2):
    x = IN_X + (4 - i) * (CW + CG)
    node(d, x, cy2, CW, ch, t, None, WHITE, PURPLE_B, 19, 14)
    if i:
        px = IN_X + (4 - i + 1) * (CW + CG)
        arrow(d, [(px - 1, cy2 + ch / 2), (px + CW + CG + 1, cy2 + ch / 2)], PURPLE_B, 2.4, 7)

# 行末下沉到第二行
x_last = IN_X + 4 * (CW + CG)
arrow(d, [(x_last + CW / 2, cy1 + ch), (x_last + CW / 2, cy2 - 2)], PURPLE_B, 2.4, 7)

text(d, IN_X + 2, 446, '每个节点都带：超时预算 · 重试策略 · 降级出边；缓存命中 / 通用知识 / 证据不足 三条短路直接回到接入层', 15, False, (109, 40, 217))

# ---------------- ③ 能力层
b3 = row([
    ('意图 / 槽位服务', ['BERT-small · ONNX · 批量', '5 类意图 + 学科/课程槽位'], GREEN_F, GREEN_B),
    ('检索服务', ['Milvus 混合(稠密+稀疏)', 'tenant / 版本 / 权限标量过滤'], GREEN_F, GREEN_B),
    ('重排 / 证据服务', ['bge-reranker GPU 批推理', 'top-k 阈值 + 可答性判定'], GREEN_F, GREEN_B),
    ('LLM 网关', ['多模型路由 · 重试 · 熔断', '语义缓存 · 提示词版本 · 计量'], GREEN_F, GREEN_B),
], 534, 96)

# ---------------- ④ 数据层
b4 = row([
    ('MySQL', ['FAQ · 会话 · 反馈', '文档元数据 · 审计'], CYAN_F, CYAN_B),
    ('Redis', ['精确 + 语义缓存 · 会话', '限流 · 分布式锁 · 单飞'], CYAN_F, CYAN_B),
    ('Milvus', ['按 tenant / kb_version 分区', 'alias 灰度切换'], CYAN_F, CYAN_B),
    ('MinIO / OSS', ['原始文档', '解析中间件'], CYAN_F, CYAN_B),
], 700, 80)

# ---------------- ⑤ 离线层
b5 = row([
    ('摄入 DAG', ['解析 → 清洗 → 切分 → 质检', '增量 · 幂等 · 可重跑'], AMBER_F, AMBER_B),
    ('索引版本化', ['blue / green collection', 'alias 一键回滚'], AMBER_F, AMBER_B),
    ('评估回归', ['golden set 500 题', 'Recall/MRR + RAGAS 门禁'], AMBER_F, AMBER_B),
    ('反馈闭环', ['未命中挖掘 → 审核', 'FAQ / 知识修订'], AMBER_F, AMBER_B),
], 852, 80)

# ---------------- ⑥ 观测层
b6 = row([
    ('OTel 链路追踪', ['trace_id 贯穿全链路'], GRAY_F, GRAY_B),
    ('Prometheus + Grafana', ['QPS · P95 · 命中率 · token'], GRAY_F, GRAY_B),
    ('结构化日志', ['JSON + trace_id + 轮转'], GRAY_F, GRAY_B),
    ('Langfuse', ['提示词 / 答案质量 · 人工标注'], GRAY_F, GRAY_B),
], 1004, 56)

# ---------------- 层间箭头
for y1, y2, color in [(252, 274, BLUE_B), (472, 494, PURPLE_B), (642, 664, GREEN_B),
                      (794, 816, CYAN_B), (946, 968, AMBER_B)]:
    arrow(d, [(CX, y1), (CX, y2)], color, 4, 10)

# ---------------- 右侧：护栏
PX = 1445
rr(d, PX, 124, 520, 950, (255, 255, 255), (203, 213, 225), r=18, width=2)
text(d, PX + 20, 146, '稳定性 / 性能护栏', 24, True, (30, 41, 59))
text(d, PX + 20, 182, '“更稳”不是靠多写 try，而是靠预算、熔断、幂等和降级阶梯', 15, False, SUB)

cards = [
    ('① 截止时间传递', ['入口 total 8s，逐节点扣减剩余预算', '检索 800ms / LLM 首 token 600ms 硬上限',
                  '超时即走降级边，绝不重试主链路'], RED_F, RED_B),
    ('② 熔断与降级阶梯', ['HyDE / 子查询 → 直接检索', 'rerank → 加权分；LLM → 失败率 >50% 熔断 30s',
                     '末级：FAQ 近义答案 → 兜底话术 → 转人工'], RED_F, RED_B),
    ('③ 并发与合并', ['意图分类 ∥ FAQ 检索；子查询 asyncio.gather', '同一 query 单飞（singleflight）防缓存击穿',
                 '流式提前返回首 token，不等全文'], AMBER_F, AMBER_B),
    ('④ 推理加速（启动预加载）', ['启动即加载 + warmup 三条链路，请求内零加载', '/readyz 未就绪不放流量；GPU 常驻 FP16',
                       '父块 2 → 6，按 token 预算截断'], AMBER_F, AMBER_B),
    ('⑤ 质量门禁', ['引用必须可溯源到 parent_id → 文件名 + 页码', '忠实度校验不过 → 重写一次或走兜底',
                 '低置信意图 / 无证据 → 明确说“不知道”'], GREEN_F, GREEN_B),
]
cy = 214
for t, b, f, o in cards:
    node(d, PX + 18, cy, 484, 150, t, b, f, o, 20, 15)
    cy += 150 + 14

# ---------------- 图例
text(d, 46, 1096, '图例', 17, True, SUB)
legend = [(BLUE_B, '接入'), (PURPLE_B, '编排'), (GREEN_B, '能力'),
          (CYAN_B, '数据'), (AMBER_B, '离线'), (GRAY_B, '观测')]
lx = 100
for color, name in legend:
    d.rounded_rectangle([lx * S, 1099 * S, (lx + 16) * S, 1115 * S], radius=4 * S, fill=color)
    text(d, lx + 24, 1096, name, 17, False, SUB)
    lx += 108

text(d, 900, 1096, '主链路', 17, False, SUB)
arrow(d, [(800, 1107), (880, 1107)], SUB, 2.5, 8)
text(d, 1000, 1096, '旁路 / 降级（虚线）', 17, False, SUB)
arrow(d, [(1180, 1107), (1260, 1107)], RED_B, 2.5, 8, dashed=True)

save(img, W, H, NAME)
