import pytest

from codeguard.data.quality import code_hash, parses_as_python3, python2_markers


@pytest.mark.parametrize(
    ("code", "marker"),
    [
        ("    print 'hello'", "print_statement"),
        ("x = to_datetime(23L)", "long_literal"),
        ("except ValueError, e:", "except_comma"),
        ("raise ValueError, 'bad'", "raise_comma"),
        ("if d.has_key(k):", "has_key"),
        ("for i in xrange(3):", "xrange"),
        ("for k, v in d.iteritems():", "dict_iter_methods"),
        ("isinstance(s, basestring)", "basestring"),
        ("p = ur'\\d+'", "ur_string"),
    ],
)
def test_python2_markers_detected(code, marker):
    assert marker in python2_markers(code)


@pytest.mark.parametrize(
    "code",
    [
        "print(x)",
        "print (x)",
        "    print = logger.info",
        "except (ValueError, KeyError) as e:",
        "for i in range(3):",
        "total = 23 + l",
    ],
)
def test_python3_code_has_no_markers(code):
    assert python2_markers(code) == []


def test_hash_ignores_whitespace_only_differences():
    assert code_hash("y = f(x)") == code_hash("y  =  f(x)\n")
    assert code_hash("y = f(x)") != code_hash("y = g(x)")
    assert code_hash("a", "b") != code_hash("ab")


def test_parses_as_python3_dedents_snippets():
    assert parses_as_python3("    x = 1\n    y = x + 1")
    assert not parses_as_python3("print 'x'")
