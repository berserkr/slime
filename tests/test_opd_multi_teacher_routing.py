"""Unit tests for MOPD (multi-teacher on-policy distillation) routing.

Covers the two pure helpers added by the multi-teacher patch:
  * slime.utils.arguments._parse_teacher_url_map  -- parses --opd-teacher-urls
  * slime.rollout.on_policy_distillation._resolve_teacher_url -- per-sample routing

No GPUs, servers, or model downloads required.
"""

from types import SimpleNamespace

import pytest

from slime.rollout.on_policy_distillation import _resolve_teacher_url
from slime.utils.arguments import _parse_teacher_url_map


# --------------------------------------------------------------------------- #
# _parse_teacher_url_map
# --------------------------------------------------------------------------- #
def test_parse_map_returns_none_when_unset():
    assert _parse_teacher_url_map(None) is None
    assert _parse_teacher_url_map("") is None


def test_parse_map_valid_pairs_with_surrounding_whitespace():
    got = _parse_teacher_url_map("math=http://h1:8001/generate, code=http://h2:8002/generate")
    assert got == {"math": "http://h1:8001/generate", "code": "http://h2:8002/generate"}


def test_parse_map_url_may_contain_equals():
    # split on the first '=' only, so query strings survive.
    got = _parse_teacher_url_map("a=http://h/gen?k=v")
    assert got == {"a": "http://h/gen?k=v"}


def test_parse_map_rejects_malformed_entry():
    with pytest.raises(ValueError, match="malformed"):
        _parse_teacher_url_map("mathhttp://h1")


def test_parse_map_rejects_empty_name_or_url():
    with pytest.raises(ValueError, match="empty name or url"):
        _parse_teacher_url_map("=http://h1")
    with pytest.raises(ValueError, match="empty name or url"):
        _parse_teacher_url_map("math=")


def test_parse_map_rejects_duplicate_name():
    with pytest.raises(ValueError, match="duplicate teacher name"):
        _parse_teacher_url_map("math=http://a,math=http://b")


def test_parse_map_rejects_all_empty():
    with pytest.raises(ValueError, match="empty map"):
        _parse_teacher_url_map(" , ")


# --------------------------------------------------------------------------- #
# _resolve_teacher_url
# --------------------------------------------------------------------------- #
def _routing_args(url_map, routing_key="teacher", rm_url=None):
    return SimpleNamespace(opd_teacher_url_map=url_map, opd_routing_key=routing_key, rm_url=rm_url)


def test_resolve_routes_by_metadata_tag():
    args = _routing_args({"math": "URL_M", "code": "URL_C"})
    assert _resolve_teacher_url(args, SimpleNamespace(metadata={"teacher": "math"})) == "URL_M"
    assert _resolve_teacher_url(args, SimpleNamespace(metadata={"teacher": "code"})) == "URL_C"


def test_resolve_honors_custom_routing_key():
    args = _routing_args({"m": "URL_M"}, routing_key="expert")
    assert _resolve_teacher_url(args, SimpleNamespace(metadata={"expert": "m"})) == "URL_M"


def test_resolve_raises_on_missing_tag():
    args = _routing_args({"math": "URL_M"})
    with pytest.raises(ValueError, match="missing metadata"):
        _resolve_teacher_url(args, SimpleNamespace(metadata={}))
    with pytest.raises(ValueError, match="missing metadata"):
        _resolve_teacher_url(args, SimpleNamespace(metadata=None))


def test_resolve_raises_on_unknown_teacher():
    args = _routing_args({"math": "URL_M"})
    with pytest.raises(ValueError, match="not in --opd-teacher-urls"):
        _resolve_teacher_url(args, SimpleNamespace(metadata={"teacher": "physics"}))


def test_resolve_falls_back_to_rm_url_without_map():
    # single-teacher back-compat: no map => use rm_url, metadata ignored.
    args = _routing_args(None, rm_url="URL_SINGLE")
    assert _resolve_teacher_url(args, SimpleNamespace(metadata={"teacher": "math"})) == "URL_SINGLE"


def test_resolve_missing_attr_is_single_teacher():
    # a config that never set opd_teacher_url_map (e.g. megatron mode) behaves as single-teacher.
    args = SimpleNamespace(rm_url="URL_SINGLE")
    assert _resolve_teacher_url(args, SimpleNamespace(metadata={"teacher": "math"})) == "URL_SINGLE"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
