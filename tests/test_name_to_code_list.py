# -*- coding: utf-8 -*-
"""Unit tests for src.services.name_to_code_list.

覆盖模块内部的切分/匹配单元：
- 市场推断 (_infer_code_market)
- 噪音词剥离 (_strip_cjk_noise)
- 拼音变体 (_generate_hybrid_variants / _get_name_variants / _to_full_pinyin)
- 输入三步切分 (_split_by_sp_char / _split_by_digit / _split_by_market / _split_input)
- 查询索引 (_StockIndex)
- 名称模糊解析 (_fix_name / _resolve_by_names)
- 端到端 resolve_name_to_code_list（AkShare 一律 mock，禁止真实网络）

与 tests/test_name_to_code_resolver.py 中的端到端用例互补，本文件侧重内部单元行为。
"""

import pytest
from unittest.mock import patch

from src.services.name_to_code_list import (
    VALUE_CONFLICT,
    _StockIndex,
    _digit_to_chinese,
    _fix_name,
    _generate_hybrid_variants,
    _get_name_variants,
    _infer_code_market,
    _resolve_by_names,
    _split_by_digit,
    _split_by_market,
    _split_by_sp_char,
    _split_input,
    _strip_cjk_noise,
    _to_full_pinyin,
    resolve_name_to_code_list,
)


def _codes(results):
    return [r["code"] for r in results]


def _names(results):
    return [r["name"] for r in results]


def _markets(results):
    return [r["market"] for r in results]


# ---------------------------------------------------------------------------
# _infer_code_market
# ---------------------------------------------------------------------------

class TestInferCodeMarket:
    def test_six_digits_is_a_share(self):
        assert _infer_code_market("600519") == "a"

    def test_five_digits_is_hk(self):
        assert _infer_code_market("00700") == "hk"

    def test_letters_is_us(self):
        assert _infer_code_market("BABA") == "us"
        assert _infer_code_market("aapl") == "us"  # 大小写不敏感
        assert _infer_code_market("BRK.B") == "us"

    def test_other_digit_lengths_rejected(self):
        assert _infer_code_market("1234") is None
        assert _infer_code_market("1234567") is None

    def test_empty_or_garbage_rejected(self):
        assert _infer_code_market("") is None
        assert _infer_code_market("60051A") is None


# ---------------------------------------------------------------------------
# _digit_to_chinese
# ---------------------------------------------------------------------------

class TestDigitToChinese:
    def test_basic(self):
        assert _digit_to_chinese("360") == "三六零"
        assert _digit_to_chinese("60") == "六零"


# ---------------------------------------------------------------------------
# _strip_cjk_noise
# ---------------------------------------------------------------------------

class TestStripCjkNoise:
    def test_strip_company_suffix_and_prefix(self):
        assert _strip_cjk_noise("茅台公司") == "茅台"
        assert _strip_cjk_noise("公司茅台") == "茅台"

    def test_strip_stock_suffix_and_prefix(self):
        assert _strip_cjk_noise("茅台股票") == "茅台"
        assert _strip_cjk_noise("股票茅台") == "茅台"

    def test_strip_repeated_noise(self):
        assert _strip_cjk_noise("茅台公司股票") == "茅台"

    def test_gu_stripped_only_next_to_ascii_alpha(self):
        assert _strip_cjk_noise("BABA股") == "BABA"
        assert _strip_cjk_noise("股BABA") == "BABA"
        # 相邻为汉字时保留"股"（兼顾"股井贡酒"类错别字匹配）
        assert _strip_cjk_noise("茅台股") == "茅台股"
        assert _strip_cjk_noise("股茅台") == "股茅台"


# ---------------------------------------------------------------------------
# 拼音变体
# ---------------------------------------------------------------------------

class TestPinyinVariants:
    def test_hybrid_variants_two_chars(self):
        variants = _generate_hybrid_variants("茅台")
        assert variants == ["茅台", "mao台", "茅tai", "maotai"]
        # 约定：末位恒为全拼音形式
        assert variants[-1] == "maotai"

    def test_hybrid_variants_non_cjk_passthrough(self):
        assert _generate_hybrid_variants("AMD") == ["AMD"]

    def test_hybrid_variants_mixed_name(self):
        # 非汉字字符原样保留，不参与拼音组合
        variants = _generate_hybrid_variants("谷歌A")
        assert variants[0] == "谷歌A"
        assert variants[-1] == "gugeA"
        assert all(v.endswith("A") for v in variants)

    def test_variants_cache_returns_same_object(self):
        assert _get_name_variants("贵州茅台") is _get_name_variants("贵州茅台")

    def test_to_full_pinyin(self):
        assert _to_full_pinyin("毛台") == "maotai"
        assert _to_full_pinyin("贵zhou茅tai") == "guizhoumaotai"

    def test_to_full_pinyin_non_cjk_passthrough(self):
        assert _to_full_pinyin("BABA") == "BABA"


# ---------------------------------------------------------------------------
# 输入切分
# ---------------------------------------------------------------------------

class TestSplitBySpChar:
    def test_split_on_special_chars(self):
        assert _split_by_sp_char("阿里:BABA.us") == ["阿里", "BABA", "us"]

    def test_no_special_char(self):
        assert _split_by_sp_char("茅台") == ["茅台"]

    def test_empty_and_blank(self):
        assert _split_by_sp_char("") == []
        assert _split_by_sp_char("  茅台  ") == ["茅台"]


class TestSplitByDigit:
    def test_extracts_4plus_digit_code(self):
        assert _split_by_digit("A股600519茅台") == (["A股", "茅台"], "600519")

    def test_same_code_twice_is_ok(self):
        assert _split_by_digit("600519茅台600519") == (["茅台"], "600519")

    def test_conflicting_codes(self):
        assert _split_by_digit("600001茅台600519") == (["茅台"], VALUE_CONFLICT)

    def test_short_digits_kept_as_token(self):
        assert _split_by_digit("360") == (["360"], None)

    def test_no_digits(self):
        assert _split_by_digit("茅台") == (["茅台"], None)


class TestSplitByMarket:
    def test_prefix_and_suffix(self):
        assert _split_by_market("A股茅台") == (["茅台"], "a")
        assert _split_by_market("阿里港股") == (["阿里"], "hk")

    def test_indicator_only(self):
        assert _split_by_market("us") == ([], "us")
        assert _split_by_market("港股") == ([], "hk")

    def test_no_indicator(self):
        assert _split_by_market("茅台") == (["茅台"], None)

    def test_ascii_indicator_needs_boundary(self):
        # HKD 中的 HK 右侧相邻字母 → 不识别为港股
        assert _split_by_market("HKD") == (["HKD"], None)
        # BABA股 中的 A股 左侧相邻字母 → 不识别为A股
        assert _split_by_market("BABA股") == (["BABA股"], None)

    def test_gu_indicator_rejects_gufen(self):
        # 大港股份 中的 港股 右侧为"份" → 不识别
        assert _split_by_market("大港股份") == (["大港股份"], None)

    def test_longer_gu_indicator_wins(self):
        # 大A股：短指示符"大A"右侧紧跟"股"时跳过，由"A股"处理
        assert _split_by_market("大A股") == (["大"], "a")

    def test_conflicting_markets(self):
        parts, mkt = _split_by_market("A股美股")
        assert mkt == VALUE_CONFLICT

    def test_split_into_multiple_parts(self):
        assert _split_by_market("茅台港股阿里") == (["茅台", "阿里"], "hk")


class TestSplitInput:
    def test_market_code_name_combined(self):
        assert _split_input("A股600519茅台") == ("a", "600519", ["茅台"])

    def test_special_chars_then_market(self):
        assert _split_input("阿里:BABA.us") == ("us", None, ["阿里", "BABA"])

    def test_conflicting_digit_codes(self):
        assert _split_input("600001茅台600519") == (None, VALUE_CONFLICT, [])

    def test_conflicting_markets(self):
        assert _split_input("港股美股") == (VALUE_CONFLICT, None, [])

    def test_plain_name(self):
        assert _split_input("茅台") == (None, None, ["茅台"])


# ---------------------------------------------------------------------------
# _StockIndex
# ---------------------------------------------------------------------------

@pytest.fixture
def index():
    return _StockIndex({
        "600519": "贵州茅台",
        "000001": "平安银行",
        "00700": "腾讯控股",
        "BABA": "阿里巴巴",
        "09988": "阿里巴巴",
    })


class TestStockIndex:
    def test_names_deduped_in_order(self, index):
        assert index.names() == ["贵州茅台", "平安银行", "腾讯控股", "阿里巴巴"]

    def test_has_name(self, index):
        assert index.has_name("贵州茅台")
        assert not index.has_name("茅台")

    def test_codes_for_name_insertion_order(self, index):
        assert index.codes_for_name("阿里巴巴") == [("BABA", "us"), ("09988", "hk")]
        assert index.codes_for_name("贵州茅台") == [("600519", "a")]

    def test_codes_for_name_skips_marketless_codes(self):
        idx = _StockIndex({"1234": "测试"})
        assert idx.codes_for_name("测试") == []

    def test_codes_for_unknown_name(self, index):
        assert index.codes_for_name("不存在") == []

    def test_extend_updates_names_incrementally(self):
        idx = _StockIndex({"600519": "贵州茅台"})
        assert idx.names() == ["贵州茅台"]  # 触发名称列表构建
        idx.extend({"601360": "三六零"})
        assert idx.names() == ["贵州茅台", "三六零"]
        assert idx.has_name("三六零")

    def test_extend_updates_reverse_index_incrementally(self):
        idx = _StockIndex({"600519": "贵州茅台"})
        assert idx.codes_for_name("贵州茅台") == [("600519", "a")]  # 触发反查表构建
        idx.extend({"601360": "三六零"})
        assert idx.codes_for_name("三六零") == [("601360", "a")]

    def test_extend_does_not_overwrite_existing_code(self):
        idx = _StockIndex({"600519": "贵州茅台"})
        idx.extend({"600519": "重复名字"})
        assert idx.db["600519"] == "贵州茅台"


# ---------------------------------------------------------------------------
# _fix_name / _resolve_by_names
# ---------------------------------------------------------------------------

class TestFixName:
    def test_exact_match(self, index):
        assert _fix_name("贵州茅台", index) == ["贵州茅台"]

    def test_substring_match(self, index):
        assert _fix_name("茅台", index) == ["贵州茅台"]
        assert _fix_name("阿里", index) == ["阿里巴巴"]

    def test_pinyin_full_alpha(self, index):
        assert _fix_name("maotai", index) == ["贵州茅台"]

    def test_pinyin_hybrid(self, index):
        assert _fix_name("贵zhou茅tai", index) == ["贵州茅台"]

    def test_typo_via_pinyin(self, index):
        # 错别字容错：毛台 → maotai ⊆ guizhoumaotai
        assert _fix_name("毛台", index) == ["贵州茅台"]

    def test_typo_long_name(self, index):
        assert _fix_name("贵州茅苔", index) == ["贵州茅台"]

    def test_no_match(self, index):
        assert _fix_name("不存在xyz", index) == []

    def test_empty_input(self, index):
        assert _fix_name("", index) == []


class TestResolveByNames:
    def test_sorted_hk_before_us(self, index):
        r = _resolve_by_names(["阿里巴巴"], None, index)
        assert _codes(r) == ["09988", "BABA"]
        assert _markets(r) == ["hk", "us"]

    def test_market_filter(self, index):
        r = _resolve_by_names(["阿里巴巴"], "hk", index)
        assert _codes(r) == ["09988"]
        assert _resolve_by_names(["阿里巴巴"], "a", index) == []

    def test_capped_at_5(self):
        idx = _StockIndex({f"60000{i}": "测试股" for i in range(7)})
        r = _resolve_by_names(["测试股"], None, idx)
        assert len(r) == 5


# ---------------------------------------------------------------------------
# resolve_name_to_code_list 端到端（AkShare 一律 mock）
# ---------------------------------------------------------------------------

class TestResolveNameToCodeListE2E:
    @pytest.fixture(autouse=True)
    def _akshare(self):
        with patch("src.services.name_to_code_list._get_akshare_name_to_code") as m:
            m.return_value = {}
            self.akshare_mock = m
            yield

    def test_empty_input(self):
        assert resolve_name_to_code_list("") == []
        assert resolve_name_to_code_list("   ") == []
        assert resolve_name_to_code_list(None) == []  # type: ignore

    def test_plain_code_resolves_name_from_local_map(self):
        r = resolve_name_to_code_list("600519")
        assert r == [{"code": "600519", "name": "贵州茅台", "market": "a"}]

    def test_market_prefix_with_code(self):
        r = resolve_name_to_code_list("A股600519")
        assert _codes(r) == ["600519"]
        assert _markets(r) == ["a"]

    def test_code_and_consistent_name_narrow_to_single(self):
        r = resolve_name_to_code_list("600519茅台")
        assert r == [{"code": "600519", "name": "贵州茅台", "market": "a"}]

    def test_code_contradicts_name_returns_empty(self):
        # 茅台 → 600519，与显式代码 000001 矛盾
        assert resolve_name_to_code_list("茅台 000001") == []

    def test_conflicting_digit_codes_returns_empty(self):
        assert resolve_name_to_code_list("600001茅台600519") == []

    def test_noise_words_stripped(self):
        r = resolve_name_to_code_list("茅台公司股票")
        assert _codes(r) == ["600519"]
        self.akshare_mock.assert_not_called()

    def test_gu_suffix_after_alpha_stripped(self):
        r = resolve_name_to_code_list("BABA股")
        assert r == [{"code": "BABA", "name": "阿里巴巴", "market": "us"}]

    def test_exchange_suffix_token_skipped(self):
        r = resolve_name_to_code_list("600519 SH")
        assert _codes(r) == ["600519"]

    def test_pinyin_full_alpha(self):
        r = resolve_name_to_code_list("maotai")
        assert _codes(r) == ["600519"]
        assert _markets(r) == ["a"]

    def test_pinyin_hybrid(self):
        r = resolve_name_to_code_list("贵zhou茅tai")
        assert _codes(r) == ["600519"]

    def test_typo_via_pinyin(self):
        r = resolve_name_to_code_list("毛台")
        assert _codes(r) == ["600519"]
        assert _names(r) == ["贵州茅台"]

    def test_multi_token_with_market_indicator(self):
        # 阿里 + BABA + us：名字经市场过滤后与代码收窄为同一候选
        r = resolve_name_to_code_list("阿里:BABA.us")
        assert r == [{"code": "BABA", "name": "阿里巴巴", "market": "us"}]

    def test_digit_to_chinese_via_akshare(self):
        self.akshare_mock.return_value = {"三六零": "601360"}
        r = resolve_name_to_code_list("360")
        assert r == [{"code": "601360", "name": "三六零", "market": "a"}]
        self.akshare_mock.assert_called_once()

    def test_digit_to_chinese_unmatched_returns_empty(self):
        assert resolve_name_to_code_list("360") == []
        self.akshare_mock.assert_called_once()

    def test_garbage_returns_empty_and_consults_akshare(self):
        assert resolve_name_to_code_list("不存在股xyz") == []
        self.akshare_mock.assert_called_once()
