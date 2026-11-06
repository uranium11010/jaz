"""`Chunk`, `split_chunks`, and verbatim rendering of chunks as `invoke` inputs."""

import pickle

from jaz._value_rendering import render_value
from jaz.protocol._prompts import render_input_section
from jaz.repl import Chunk, REPLHistoryEntry, chunk_items, split_chunks


def test_split_top_level_blocks_in_order():
    text = 'pre <msg id="1">a <b>x</b></msg> mid <msg id=2>b\nc</msg> post'
    assert split_chunks(text) == ['<msg id="1">a <b>x</b></msg>', "<msg id=2>b\nc</msg>"]
    assert all(type(c) is Chunk for c in split_chunks(text))


def test_nesting_unclosed_and_self_closing():
    assert split_chunks("<a><a>x</a></a>") == ["<a><a>x</a></a>"]
    assert split_chunks("<a>open <b>x</b>") == ["<b>x</b>"]
    assert split_chunks("<br/><br /><a x='1'/><c>y</c>") == ["<c>y</c>"]
    # the `/` ends an unquoted attribute value: an opener, not a self-closing tag
    assert split_chunks("<a href=x/>y</a>") == ["<a href=x/>y</a>"]


def test_namespaced_names_and_comments():
    assert split_chunks("<ns:a>x</ns:a>") == ["<ns:a>x</ns:a>"]
    assert split_chunks("<a><!-- </a> --></a>") == ["<a><!-- </a> --></a>"]
    assert split_chunks("<!-- <a>x</a> --><b>y</b>") == ["<b>y</b>"]


def test_linear_on_unclosed_openers():
    import time

    t = time.time()
    assert split_chunks("<module>\n" * 20000) == []
    assert time.time() - t < 2


def test_chunk_items_flattens_and_rejects_mixed():
    a, b = Chunk("<a>1</a>"), Chunk("<b>2</b>")
    assert chunk_items(a) == [a]
    assert chunk_items([[a], (b,)]) == [a, b]
    assert chunk_items([a, "plain"]) is None
    assert chunk_items([]) is None


def test_render_verbatim():
    a, b = Chunk('<msg q="it\'s">x\\n</msg>'), Chunk("<msg>y</msg>")
    assert render_value([a, b]) == a + "\n" + b
    assert render_value([[a], [b]]) == a + "\n" + b
    assert render_value(a) == a
    assert render_value(["plain", a]) == repr(["plain", a])  # mixed falls back to repr


def _rendered(value, cap):
    return render_input_section({"m": value}, max_invoke_input_length=cap)[0].value


def test_truncation_keeps_whole_chunks():
    cs = [Chunk(f"<m>{i:03d}</m>") for i in range(10)]
    out = _rendered(cs, 40)
    lines = out.split("\n")
    assert all(line in cs for line in lines[:-1])
    assert lines[-1].startswith("[...") and "more chunks omitted" in lines[-1]
    one_left = _rendered(cs[:2], len(cs[0]) + 1)
    assert one_left.endswith("[...1 more chunk omitted...]")


def test_oversized_first_chunk_is_cut_not_dropped():
    big = Chunk("<m>" + "x" * 500 + "</m>")
    out = _rendered([big], 100)
    assert "x" in out and "more chunk" not in out
    out = _rendered(big, 100)  # a bare chunk goes through the same path
    assert "x" in out and len(out) < 300


def test_entry_chunks_cached_and_picklable():
    e = REPLHistoryEntry(llm_response="", repl_output="<m>a</m><m>b</m>", repl_exception=None)
    assert e.chunks is e.chunks
    e2 = pickle.loads(pickle.dumps(e))
    assert e2.chunks == ["<m>a</m>", "<m>b</m>"] and type(e2.chunks[0]) is Chunk
