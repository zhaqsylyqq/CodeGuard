from codeguard.data.diff import reconstruct

BASIC = (
    "@@ -45,7 +45,7 @@ else:\n"
    " class T(unittest.TestCase):\n"
    " \n"
    "     def test(self):\n"
    "-        x = f(1970, 1)\n"
    "+        x = g(23)\n"
    "         self.assertEqual(x, 1)\n"
)


def test_basic_replacement():
    pair = reconstruct(BASIC)
    assert pair.buggy == (
        "class T(unittest.TestCase):\n\n    def test(self):\n        x = f(1970, 1)\n"
        "        self.assertEqual(x, 1)"
    )
    assert pair.fixed == (
        "class T(unittest.TestCase):\n\n    def test(self):\n        x = g(23)\n"
        "        self.assertEqual(x, 1)"
    )
    assert pair.removed == ("        x = f(1970, 1)",)
    assert pair.added == ("        x = g(23)",)
    assert pair.n_hunks == 1
    assert pair.is_change


def test_header_is_never_part_of_code():
    pair = reconstruct(BASIC)
    assert "@@" not in pair.buggy and "@@" not in pair.fixed


def test_multiline_removal_and_insertion_keep_order():
    diff = "@@ -1,4 +1,3 @@\n a = 1\n-b = 2\n-c = 3\n+bc = 5\n d = 4\n"
    pair = reconstruct(diff)
    assert pair.buggy == "a = 1\nb = 2\nc = 3\nd = 4"
    assert pair.fixed == "a = 1\nbc = 5\nd = 4"
    assert pair.removed == ("b = 2", "c = 3")


def test_pure_insertion():
    pair = reconstruct("@@ -1,2 +1,3 @@\n x = 1\n+check(x)\n y = 2\n")
    assert pair.buggy == "x = 1\ny = 2"
    assert pair.fixed == "x = 1\ncheck(x)\ny = 2"
    assert pair.removed == ()


def test_no_newline_marker_is_skipped():
    diff = "@@ -1,2 +1,2 @@\n x = 1\n-y = 2\n\\ No newline at end of file\n+y = 3\n"
    pair = reconstruct(diff)
    assert pair.buggy == "x = 1\ny = 2"
    assert pair.fixed == "x = 1\ny = 3"


def test_blank_context_line_without_leading_space():
    pair = reconstruct("@@ -1,3 +1,3 @@\n a = 1\n\n-b = 2\n+b = 3\n")
    assert pair.buggy == "a = 1\n\nb = 2"
    assert pair.fixed == "a = 1\n\nb = 3"


def test_crlf_line_endings():
    pair = reconstruct("@@ -1,2 +1,2 @@\r\n a = 1\r\n-b = 2\r\n+b = 3\r\n")
    assert pair.buggy == "a = 1\nb = 2"
    assert "\r" not in pair.fixed


def test_file_header_before_first_hunk_is_ignored():
    pair = reconstruct("--- a/f.py\n+++ b/f.py\n@@ -1 +1 @@\n-x = 1\n+x = 2\n")
    assert (pair.buggy, pair.fixed) == ("x = 1", "x = 2")


def test_removed_line_that_looks_like_a_marker():
    pair = reconstruct("@@ -1 +1 @@\n--x\n+-x + 1\n")
    assert (pair.buggy, pair.fixed) == ("-x", "-x + 1")


def test_multiple_hunks_are_counted():
    diff = "@@ -1 +1 @@\n-a\n+b\n@@ -9 +9 @@\n-c\n+d\n"
    pair = reconstruct(diff)
    assert pair.n_hunks == 2
    assert pair.buggy == "a\nc"


def test_identical_sides_are_not_a_change():
    assert not reconstruct("@@ -1 +1 @@\n x = 1\n").is_change
