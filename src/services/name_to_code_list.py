# -*- coding: utf-8 -*-
"""
===================================
Name-to-Code List Resolution Engine
===================================

Resolve stock name/code to candidate code list (max 5, sorted A-share → HK → US).

漏斗提取顺序：
  1. 按特殊字符 + 连续数字 + 市场信息切分成 tokens 序列，每个 token 逐一匹配
  2. 若为数字 → 4+ 数字视为代码，2-3 数字转汉字匹配名字
  3. 若存在汉字 → 提取汉字字母组合名字
  4. 连续字母 → 提取匹配名字（1-5 位纯字母未匹配上名字则判定为美股代码）
  5. AkShare 再来一轮

每次匹配成功后立刻矛盾检测。例如 "A股阿里"：阿里匹配成功后发现无 A 股代码 → 矛盾返回空。

性能要点：
  - 每次 resolve 构建一次 _StockIndex（名称去重列表 + 名称→代码反向索引），
    避免每个 token 全表扫描；AkShare 兜底时增量扩展索引
  - 名字的拼音/混合变体按进程级缓存（名字有界），避免重复调用 pypinyin
    及重复生成 2^N 变体
"""

from __future__ import annotations

import difflib
import re
from functools import lru_cache
from typing import Dict, List, Optional, Set, Tuple

from src.data.stock_mapping import STOCK_NAME_MAP
from src.services.name_to_code_resolver import (
    _contains_cjk,
    _get_akshare_name_to_code,
)
from .stock_code_utils import normalize_code

try:
    from pypinyin import lazy_pinyin
except Exception:  # pypinyin 缺失时退化为无拼音匹配
    lazy_pinyin = None

VALUE_CONFLICT = 'value_conflict'


def _normalize_code(raw: str) -> Optional[str]:
    return normalize_code(raw)


def _is_cjk(ch: str) -> bool:
    return '\u3400' <= ch <= '\u9fff'


def _is_ascii_alpha(ch: str) -> bool:
    return ch.isascii() and ch.isalpha()


# 统一市场指示符列表，忽略大小写。
# 预计算小写形式与首/尾字符特征（是否ASCII字母、是否以"股"结尾），按长度降序保证最长匹配优先。
_MARKET_INDICATORS: List[Tuple[str, str, bool, bool, bool]] = sorted(
    [
        (indicator.lower(), mkt, _is_ascii_alpha(indicator[0]),
         _is_ascii_alpha(indicator[-1]), indicator[-1] == "股")
        for indicator, mkt in [
            ("A股", "a"), ("大A", "a"),
            ("港股", "hk"), ("H股", "hk"),
            ("美股", "us"), ("台股", "tw"),
            ("日股", "jp"), ("韩股", "kr"),
            ("hk", "hk"), ("us", "us"),
            ("tw", "tw"), ("jp", "jp"), ("kr", "kr"),
        ]
    ],
    key=lambda x: len(x[0]),
    reverse=True,
)

# 市场排序优先级（用于结果排序：A股→港股→美股）
_MARKET_ORDER: Dict[str, int] = {"a": 0, "hk": 1, "us": 2}

# 非汉字/字母/数字的字符类（拆分用）
_RE_NON_TOKEN = re.compile(r'[^a-zA-Z0-9\u3400-\u9fff]+')

# 已知交易所后缀（独立出现时应跳过，避免被误识别为美股代码）
_EXCHANGE_SUFFIXES = frozenset({"SH", "SZ", "BJ", "SS"})


def _infer_code_market(code: str) -> Optional[str]:
    """根据代码格式推断市场：6位数字→a，5位数字→hk，字母→us。"""
    if not code:
        return None
    c = code.strip().upper()
    if c.isdigit():
        if len(c) == 5:
            return "hk"
        if len(c) == 6:
            return "a"
        return None
    if re.match(r"^[A-Z]{1,5}(\.[A-Z])?$", c):
        return "us"
    return None


def _strip_cjk_noise(token: str) -> str:
    """去除中文股票输入中的干扰词，应对用户的随意输入（如"茅台公司股票"）。
    - 以"股票"、"公司"开头或结尾，直接删去（置信度高），循环处理至无变化
    - 以"股"开头或结尾且相邻为字母，删去"股"（如"BABA股"→"BABA"，"股BABA"→"BABA"）
    """
    # 循环去除"股票"、"公司"前缀/后缀
    changed = True
    while changed:
        changed = False
        for noise in ("股票", "公司"):
            if token.startswith(noise) and len(token) > len(noise):
                token = token[len(noise):]
                changed = True
                break
            if token.endswith(noise) and len(token) > len(noise):
                token = token[:-len(noise)]
                changed = True
                break

    # 一些极端情况
    # 以"股"开头且相邻为字母 → 去掉开头的"股"，保留"股井贡酒"错别字匹配
    if len(token) > 1 and token.startswith("股") and _is_ascii_alpha(token[1]):
        token = token[1:]
    # 以"股"结尾且相邻为字母 → 去掉结尾的"股"
    if len(token) > 1 and token.endswith("股") and _is_ascii_alpha(token[-2]):
        token = token[:-1]

    return token


def _generate_hybrid_variants(name_cn: str) -> List[str]:
    """生成汉字股票的全部 pinyin/CJK 混合边界变体。每个汉字都可能被用户输入成拼音共 2^N 种
    例如 贵州茅台 生成: 贵州茅台, gui州茅台, 贵州maotai, guizhoumaotai, ... 共 16 种情况
    最后一个元素恒为全拼音形式（无汉字时为原名本身）。
    """
    cjk_chars = [ch for ch in name_cn if _is_cjk(ch)]
    if not cjk_chars or lazy_pinyin is None:
        return [name_cn]

    # 仅对 CJK 字符取拼音，避免非汉字干扰, 例如"京东方A"
    pinyins = lazy_pinyin(''.join(cjk_chars))

    n = len(cjk_chars)
    variants: List[str] = []

    # 2^N 种组合：bitmask 中第 i 位为 1 表示第 i 个 CJK 字符替换为拼音
    for mask in range(1 << n):
        result: List[str] = []
        cjk_idx = 0
        for ch in name_cn:
            if _is_cjk(ch):
                if mask & (1 << cjk_idx):
                    result.append(pinyins[cjk_idx])
                else:
                    result.append(ch)
                cjk_idx += 1
            else:
                result.append(ch)
        variants.append(''.join(result))

    return variants


# 名字 → 混合变体列表的进程级缓存。名字集合有界（本地映射 + AkShare 名单），
# 避免每次查询都对全部名字调用 pypinyin 并重新生成 2^N 变体。
_VARIANTS_CACHE: Dict[str, List[str]] = {}


def _get_name_variants(name_cn: str) -> List[str]:
    """带缓存的名字混合变体生成。"""
    variants = _VARIANTS_CACHE.get(name_cn)
    if variants is None:
        variants = _generate_hybrid_variants(name_cn)
        _VARIANTS_CACHE[name_cn] = variants
    return variants


@lru_cache(maxsize=2048)
def _to_full_pinyin(text: str) -> str:
    """将文本中的汉字替换为拼音（非汉字原样保留）。
    例如 毛台→maotai、贵zhou茅tai→guizhoumaotai。用于错别字容错匹配。
    """
    cjk_chars = [ch for ch in text if _is_cjk(ch)]
    if not cjk_chars or lazy_pinyin is None:
        return text
    pinyins = iter(lazy_pinyin(''.join(cjk_chars)))
    return ''.join(next(pinyins) if _is_cjk(ch) else ch for ch in text)


class _StockIndex:
    """stock_database 的查询索引：名称去重列表 + 名称→(代码,市场) 反查表。

    每次 resolve 调用构建一次（懒加载），避免每个 token/名字都全表扫描；
    AkShare 兜底扩充数据时通过 extend 增量更新，已缓存的派生结构保持有效。
    """

    __slots__ = ("db", "_names", "_name_set", "_name_to_codes")

    def __init__(self, db: Dict[str, str]):
        self.db = db
        self._names: Optional[List[str]] = None
        self._name_set: Optional[Set[str]] = None
        self._name_to_codes: Optional[Dict[str, List[Tuple[str, str]]]] = None

    def _ensure_names(self) -> None:
        if self._names is None:
            self._names = list(dict.fromkeys(self.db.values()))
            self._name_set = set(self._names)

    def names(self) -> List[str]:
        """去重保序的名称列表。"""
        self._ensure_names()
        return self._names  # type: ignore[return-value]

    def has_name(self, name: str) -> bool:
        self._ensure_names()
        return name in self._name_set  # type: ignore[operator]

    def codes_for_name(self, name: str) -> List[Tuple[str, str]]:
        """查找指定名称对应的所有 (code, market) 对（按库中插入顺序）。"""
        if self._name_to_codes is None:
            idx: Dict[str, List[Tuple[str, str]]] = {}
            for code, mapped_name in self.db.items():
                m = _infer_code_market(code)
                if m:
                    idx.setdefault(mapped_name, []).append((code, m))
            self._name_to_codes = idx
        return self._name_to_codes.get(name, [])

    def extend(self, code_to_name: Dict[str, str]) -> None:
        """并入新的 code→name 映射（AkShare 兜底），增量更新已构建的索引。"""
        for code, name in code_to_name.items():
            if code in self.db:
                continue
            self.db[code] = name
            if self._names is not None and name not in self._name_set:  # type: ignore[operator]
                self._names.append(name)
                self._name_set.add(name)  # type: ignore[union-attr]
            if self._name_to_codes is not None:
                m = _infer_code_market(code)
                if m:
                    self._name_to_codes.setdefault(name, []).append((code, m))


def _fix_name(s: str, index: _StockIndex) -> List[str]:
    """将模糊名称片段解析为完整股票名称列表（去重保序）。
    输入为 字母+汉字 ，可能存在错别字噪音
    """
    if not s:
        return []
    stock_database_names = index.names()

    # 1. 精确匹配
    if index.has_name(s):
        return [s]

    # 2. 子串匹配，如：茅台
    matches = [name for name in stock_database_names if s in name]
    if matches:
        return matches  # 字串匹配到的列表

    # 3. 拼音的模糊匹配
    # 错别字容错：查询整体转拼音（如 毛台→maotai），纯字母查询无需转换（等价于变体子串检查）
    s_pinyin = _to_full_pinyin(s) if _contains_cjk(s) else None
    for name in stock_database_names:
        variants = _get_name_variants(name)
        # 汉字 或 汉字拼音 组合，例如：贵zhou茅tai
        if any(s in variant for variant in variants):
            matches.append(name)
            continue
        # 查询拼音与名字全拼音（变体列表末位）子串匹配，例如 maotai ⊆ guizhoumaotai
        if s_pinyin and s_pinyin in variants[-1]:
            matches.append(name)
    if matches:
        return matches

    # 4. 错别字模糊匹配，4字股票匹配上3个字，3字股票匹配上两个字
    if len(s) > 2:
        dl_matches = difflib.get_close_matches(s, stock_database_names, n=5, cutoff=0.65)
        if dl_matches:
            return dl_matches

    return []


def _resolve_by_names(names: List[str], market: Optional[str], index: _StockIndex) -> List[Dict[str, str]]:
    """通过名称列表解析 stock_database，按市场过滤 → 排序 → 上限5条。"""
    results: List[Dict[str, str]] = []
    seen: Set[str] = set()
    for full_name in names:
        for c, m in index.codes_for_name(full_name):
            if market and m != market:
                continue
            if c in seen:
                continue
            seen.add(c)
            results.append({"code": c, "name": full_name, "market": m})
    results.sort(key=lambda r: _MARKET_ORDER.get(r["market"], 99))
    return results[:5]


def _digit_to_chinese(s: str) -> str:
    """将纯数字串转为中文数字（如 360→三六零），供名称匹配使用。"""
    dig_cn = "零一二三四五六七八九"
    return "".join(dig_cn[int(ch)] for ch in s)


# 拆分时提取连续数字为独立 token
_RE_DIGITS = re.compile(r'(\d+)')


def _split_by_market(s: str) -> Tuple[List[str], Optional[str]]:
    """提取并移除 token 中的市场指示符，返回 (剩余片段列表, 市场)。
    上游已通过 _split_by_sp_char 和 _split_by_digit 完成切分，
    因此 token 不含特殊字符和连续数字，仅需字符串切片。

    分两轮匹配：
      第一轮 — 汉字市场指示符（A股、港股…），检查后续汉字防止"大港股份"等误识别
      第二轮 — 纯英文市场指示符（hk、us…），检查相邻ASCII字母防止 HKD 等误识别

    例如：A股茅台   -> (['茅台'], 'a')
         阿里港股  -> (['阿里'], 'hk')
         us        -> ([], 'us')
         HKD       -> (['HKD'], None)
         港股      -> ([], 'hk')
         茅台      -> (['茅台'], None)
    """
    markets_found: Set[str] = set()
    cuts: List[Tuple[int, int]] = []
    search_in = s.lower()

    for indicator, mkt, starts_ascii, ends_ascii, ends_gu in _MARKET_INDICATORS:
        start = 0
        while True:
            idx = search_in.find(indicator, start)
            if idx < 0:
                break
            end = idx + len(indicator)

            # 首字符为ASCII字母 → 左侧相邻不能是字母（防 BABA股→A股）
            if starts_ascii and idx > 0 and _is_ascii_alpha(s[idx - 1]):
                start = idx + 1
                continue
            # 尾字符为ASCII字母 → 右侧相邻不能是字母（防 HKD→HK、A股B）
            if ends_ascii and end < len(s) and _is_ascii_alpha(s[end]):
                start = idx + 1
                continue

            right = s[end:]
            # 结尾是"股" → 右侧不能为"分"/"份"/"fen"（防 大港股份→港股）
            if ends_gu and any(right.startswith(p) for p in ("份", "分", "fen")):
                start = idx + 1
                continue
            # 结尾不是"股" → 右侧紧跟"股"时应跳过，交由含"股"的更长/更匹配指示符处理（如 大A股→A股）
            if not ends_gu and right.startswith("股"):
                start = idx + 1
                continue

            markets_found.add(mkt)
            cuts.append((idx, end))
            start = end

    # ---- 确定最终市场 ----
    if len(markets_found) > 1:
        mkt: Optional[str] = VALUE_CONFLICT
    elif len(markets_found) == 1:
        mkt = next(iter(markets_found))
    else:
        mkt = None

    # ---- 开始对市场指示符进行切分提取 ----
    if not cuts:
        return ([s], mkt) if s else ([], mkt)

    cuts.sort(key=lambda x: x[0])
    parts: List[str] = []
    prev = 0
    for st, en in cuts:
        if st > prev:
            parts.append(s[prev:st])
        prev = en
    if prev < len(s):
        parts.append(s[prev:])

    return (parts, mkt)


def _split_by_sp_char(s: str) -> List[str]:
    """按特殊字符（非字母/数字/汉字）切分输入，过滤空串。
    例如："阿里:BABA.us" -> ['阿里', 'BABA', 'us']
    """
    return [t for t in _RE_NON_TOKEN.split(s) if t]


def _split_by_digit(s: str) -> Tuple[List[str], Optional[str]]:
    """按连续数字切分，并提取可能存在的4+位股票代码（置信度较高）。
    同一token内出现多个不同的4+位数字代码时返回 VALUE_CONFLICT。
    2-3位数字保留为token，供后续漏斗阶段转汉字名字匹配。
    例如："A股600519茅台" -> (['A股', '茅台'], '600519')
          "600001茅台600519" -> (['茅台'], 'value_conflict')
          "茅台" -> (['茅台'], None)
    """
    parts = _RE_DIGITS.split(s)
    digit_code: Optional[str] = None
    non_digit_parts: List[str] = []

    for part in parts:
        if not part:
            continue
        if part.isdigit() and len(part) >= 4:
            if digit_code and digit_code != part:
                return (non_digit_parts, VALUE_CONFLICT)
            digit_code = part
        else:
            non_digit_parts.append(part)

    return (non_digit_parts, digit_code)


def _split_input(s: str) -> Tuple[Optional[str], Optional[str], List[str]]:
    """按特殊字符 → 连续数字 → 市场信息 三步切分输入。
    返回 (market, digit_code, tokens)
    """
    market: Optional[str] = None
    digit_code: Optional[str] = None

    # 第一步：特殊字符切分
    raw_tokens = _split_by_sp_char(s)

    # 第二步：连续数字切分，提取4+位数字代码
    tokens_after_digit: List[str] = []
    for token in raw_tokens:
        parts, dc = _split_by_digit(token)
        tokens_after_digit.extend(parts)
        if dc == VALUE_CONFLICT:
            return (market, VALUE_CONFLICT, [])
        if digit_code is not None and dc is not None and digit_code != dc:
            return (market, VALUE_CONFLICT, [])
        digit_code = digit_code or dc

    # 第三步：市场信息切分
    tokens_after_market: List[str] = []
    for token in tokens_after_digit:
        parts, m = _split_by_market(token)
        tokens_after_market.extend(parts)
        if m == VALUE_CONFLICT:
            return (VALUE_CONFLICT, digit_code, [])
        if market is not None and m is not None and market != m:
            return (VALUE_CONFLICT, digit_code, [])
        market = market or m

    return (market, digit_code, tokens_after_market)


# ═══════════════════════════════════════════════════════════════
# 漏斗阶段 2-4 工作流函数（可复用）
#   阶段 2: 数字 → 4+ 代码 / 2-3 转汉字匹配名字
#   阶段 3: 汉字 → 提取汉字字母组合名字
#   阶段 4: 字母 → 匹配名字（1-5 位未命中则美股代码）
# 每次匹配后通过 narrow_fn 收窄结果，矛盾时通过 has_conflict 标记
# ═══════════════════════════════════════════════════════════════
def _funnel_workflow(
    tokens: List[str],
    market: Optional[str],
    index: _StockIndex,
    narrow_fn,
    make_candidate_fn,
) -> Tuple[bool, List[str], List[str], List[str], bool]:
    """执行漏斗阶段2-4：逐token匹配代码/名字。

    Args:
        tokens: 待处理的token列表（纯数字 / 汉字+字母 / 纯字母）
        market: 显式市场（可为None）
        index: stock_database 查询索引
        narrow_fn: 收窄函数，签名为 (candidates: List[Dict]) -> bool (True=矛盾)
        make_candidate_fn: 代码→候选条目构造函数，签名为 (code: str) -> Optional[Dict]

    Returns:
        (has_local_name_hit, unmatched_cjk, unmatched_alpha, pending_digit_names, has_conflict)
        - has_conflict=True 时调用方应立即返回空列表
    """
    has_local_name_hit = False
    unmatched_cjk: List[str] = []
    unmatched_alpha: List[str] = []
    pending_digit_names: List[str] = []

    for token in tokens:
        # 单字符视为无意义
        if len(token) < 2:
            continue
        # 跳过独立出现的交易所后缀
        if token.upper() in _EXCHANGE_SUFFIXES:
            continue

        # ── 漏斗阶段 2：纯数字 → 4位+ 匹配代码 / 2-3位 转汉字匹配名字 ──
        if token.isdigit():
            if len(token) >= 4:
                code = _normalize_code(token)
                if code:
                    cand = make_candidate_fn(code)
                    if cand is None:
                        return False, [], [], [], True
                    if narrow_fn([cand]):
                        return False, [], [], [], True
                continue

            if 2 <= len(token) <= 3:
                chinese = _digit_to_chinese(token)
                names = _fix_name(chinese, index)
                if names:
                    has_local_name_hit = True
                    if narrow_fn(_resolve_by_names(names, market, index)):
                        return False, [], [], [], True
                else:
                    pending_digit_names.append(chinese)
                continue

        token = _strip_cjk_noise(token)
        if not token:
            continue

        # ── 漏斗阶段 3： 汉字+字母 组合匹配带拼音的股票名字如：贵州maotai ──
        if _contains_cjk(token):
            cjk_indices = [i for i, ch in enumerate(token) if _is_cjk(ch)]
            first, last = cjk_indices[0], cjk_indices[-1]
            base = token[first:last + 1]
            left = token[:first]
            right = token[last + 1:]

            names = None

            '''以汉字主体为基底，扩展左右字母组合，即应对正确的汉字拼音组合，也兼顾用户误输入的容错
            例如"阿里i"，"阿里baba"
            '''
            # 优先级 1：最长扩展（双侧字母一起匹配）
            if left and right:
                names = _fix_name(token, index)

            # 优先级 2：单侧扩展
            if not names:
                l_names = _fix_name(left + base, index) if left else None
                r_names = _fix_name(base + right, index) if right else None
                if l_names and r_names:
                    if set(l_names) != set(r_names):
                        return False, [], [], [], True  # 双侧匹配指向不同股票 → 矛盾
                    names = l_names
                elif l_names:
                    names = l_names
                elif r_names:
                    names = r_names

            # 优先级 3：纯基底匹配
            if not names:
                names = _fix_name(base, index)

            if not names:
                unmatched_cjk.append(token)
                continue

            has_local_name_hit = True
            if narrow_fn(_resolve_by_names(names, market, index)):
                return False, [], [], [], True
            continue

        # ── 漏斗阶段 4：纯字母 → 匹配 美股代码 或 拼音名字 ──
        if token.isalpha():
            # 美股代码，例如:BABA
            if token.upper() in index.db:
                code = _normalize_code(token)
                if code:
                    cand = make_candidate_fn(code)
                    if cand is None:
                        return False, [], [], [], True
                    if narrow_fn([cand]):
                        return False, [], [], [], True
                continue

            # 拼音名字，例如:maotai
            names = _fix_name(token, index)
            if names:
                has_local_name_hit = True
                if narrow_fn(_resolve_by_names(names, market, index)):
                    return False, [], [], [], True
                continue
            unmatched_alpha.append(token)
            continue

    return has_local_name_hit, unmatched_cjk, unmatched_alpha, pending_digit_names, False


def resolve_name_to_code_list(name: str) -> List[Dict[str, str]]:
    """将股票名称或代码解析为候选代码列表（最多 5 条，按 A股→港股→美股 排序）。
    允许接受带有 市场+代码+名字 任意顺序的最多3个元素的输入。
    数据源为 STOCK_NAME_MAP + AkShare兜底
    优先匹配置信度较高的市场信息，汉字名字（顺序在市场后），数字代码
    """
    # 输入校验：空值、非字符串类型直接返回空列表
    if not name or not isinstance(name, str):
        return []
    s = name.strip()
    if not s:
        return []

    # 子函数统一数据接口：初始为本地映射，第5步兜底时用 AkShare 更新
    stock_database: Dict[str, str] = dict(STOCK_NAME_MAP)
    index = _StockIndex(stock_database)

    # ═══════════════════════════════════════════════════════════════
    # 漏斗阶段 1：按特殊字符 + 连续数字 + 市场信息切分输入
    # ═══════════════════════════════════════════════════════════════
    market, digit_code, tokens = _split_input(s)
    if not tokens and not digit_code:
        return []

    # ═══════════════════════════════════════════════════════════════
    # 漏斗收窄基底：all_results 为唯一状态，每步匹配后收窄
    # ═══════════════════════════════════════════════════════════════
    all_results: List[Dict[str, str]] = []
    seen: Set[str] = set()

    def _narrow_results(candidates: List[Dict[str, str]]) -> bool:
        """将候选结果收窄到 all_results。返回 True 表示矛盾。"""
        nonlocal all_results, seen
        if not candidates:
            return False
        if not all_results:
            for r in candidates:
                if r["code"] not in seen:
                    seen.add(r["code"])
                    all_results.append(r)
            return False

        narrowed: List[Dict[str, str]] = []
        narrowed_seen: Set[str] = set()
        for exist in all_results:
            for cand in candidates:
                if exist["name"] and cand["name"] and exist["name"] != cand["name"]:
                    continue
                if exist["code"] and cand["code"] and exist["code"] != cand["code"]:
                    continue
                if exist["market"] and cand["market"] and exist["market"] != cand["market"]:
                    continue
                merged = {
                    "code": exist["code"] or cand["code"],
                    "name": exist["name"] or cand["name"],
                    "market": exist["market"] or cand["market"],
                }
                if merged["code"] not in narrowed_seen:
                    narrowed_seen.add(merged["code"])
                    narrowed.append(merged)

        if not narrowed:
            return True
        all_results = narrowed
        seen = narrowed_seen
        return False

    def _make_code_candidate(code: str) -> Optional[Dict[str, str]]:
        """将纯代码构造为候选条目，做市场矛盾检测。"""
        code_market = _infer_code_market(code)
        if market and code_market and market != code_market:
            return None
        return {
            "code": code,
            "name": stock_database.get(code, ""),
            "market": market or code_market or "",
        }

    # 处理 _split_input 提取的 4+ 位数字代码（置信度较高，优先处理）
    if digit_code:
        code = _normalize_code(digit_code)
        if code:
            cand = _make_code_candidate(code)
            if cand is None:
                return []
            if _narrow_results([cand]):
                return []

    # ═══════════════════════════════════════════════════
    # 第一轮：纯本地匹配，调用漏斗阶段 2-4 工作流
    # ═══════════════════════════════════════════════════
    has_local_name_hit, unmatched_cjk, unmatched_alpha, pending_digit_names, has_conflict = \
        _funnel_workflow(tokens, market, index, _narrow_results, _make_code_candidate)
    if has_conflict:
        return []

    # ═══════════════════════════════════════════════════
    # 漏斗阶段 5：AkShare 在线兜底 → 扩充 stock_database → 重新调用工作流
    # ═══════════════════════════════════════════════════
    need_akshare = (
        not has_local_name_hit
        and market in [None, 'a']
        and (not all_results or all(r.get("market") in [None, "", "a"] for r in all_results))
        and (unmatched_cjk or pending_digit_names or unmatched_alpha)
    )
    # 更新扩展 stock_database
    if need_akshare:
        akshare_map = _get_akshare_name_to_code()
        if akshare_map:
            # akshare_map 为 name→code，反转为 code→name 并入索引
            index.extend({ak_code: ak_name for ak_name, ak_code in akshare_map.items()})

        # 将未匹配的所有 token ，用扩充后的 stock_database 重新调用工作流
        retry_tokens = unmatched_cjk + pending_digit_names + unmatched_alpha
        if retry_tokens:
            _, unmatched_cjk, unmatched_alpha, pending_digit_names, has_conflict = \
                _funnel_workflow(retry_tokens, market, index, _narrow_results, _make_code_candidate)
            if has_conflict:
                return []

    # 第二轮工作流后仍未匹配的 CJK token：无结果则非法输入，有结果则跳过
    if unmatched_cjk and not all_results:
        return []

    # 第二轮工作流后仍未匹配的数字转汉字 token → 返回空
    if pending_digit_names:
        return []

    # 处理仍未匹配的纯字母 token：≤5 位视为美股代码
    if unmatched_alpha:
        for token in unmatched_alpha:
            if len(token) <= 5:
                code = token.upper()
                cand = _make_code_candidate(code)
                if cand is None:
                    return []
                if _narrow_results([cand]):
                    return []

    # 最终结果：按市场优先级排序（A股→港股→美股），截断至最多5条
    if all_results:
        all_results.sort(key=lambda r: _MARKET_ORDER.get(r["market"], 99))
        return all_results[:5]
    return []
