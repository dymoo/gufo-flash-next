"""Coding agent: repair or complete a small Python package through file and test tools.

A repository holds four modules of simple target functions (each with a
realistic bug, visible tests and hidden tests), specification-driven functions
(a docstring that states every edge case, basic visible tests and edge-case
hidden tests) and generated helper functions. Tiers: implement a simple stub
(easy); implement a specified function (medium); fix a bug a failing test
exposes plus one known only from a user report, or implement a specified
function plus fix a reported bug (hard). The checker runs the original visible
tests plus all hidden tests on the final files, so special-casing a visible test
does not pass.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
import tempfile
import textwrap
from pathlib import Path

from . import Call, Step, Task, ToolError, rng_for, tool

MODULES = {"textkit": "String utilities.", "seqkit": "Sequence and list utilities.",
           "numkit": "Number utilities.", "parsekit": "Parsing utilities."}

# name: (module, reference source, bug (old, new) replacement, visible, weak, hidden, report).
# `visible` asserts fail on the bug; `weak` asserts pass on it; `hidden` asserts are checker-only.
TARGETS = {
    "is_palindrome": ("textkit", '''
def is_palindrome(s):
    """Return True when s reads the same backwards, ignoring case and non-alphanumeric characters."""
    cleaned = [c.lower() for c in s if c.isalnum()]
    return cleaned == cleaned[::-1]
''', ("[c.lower() for c in s", "[c for c in s"),
        ["is_palindrome('A man, a plan, a canal: Panama') is True", "is_palindrome('race a car') is False"],
        ["is_palindrome('racecar') is True", "is_palindrome('race a car') is False", "is_palindrome('') is True"],
        ["is_palindrome('No lemon, no melon') is True", "is_palindrome('Aa') is True", "is_palindrome('ab') is False",
         "is_palindrome('Was it a car or a cat I saw?') is True"],
        "`is_palindrome('Step on no pets')` returns False, but the check must ignore case."),
    "word_count": ("textkit", '''
def word_count(text):
    """Return a dict mapping each lowercase word to its count; words are maximal runs of letters."""
    counts = {}
    for word in re.findall(r"[a-z]+", text.lower()):
        counts[word] = counts.get(word, 0) + 1
    return counts
''', ('re.findall(r"[a-z]+", text.lower())', "text.lower().split()"),
        ["word_count('The cat and the hat.') == {'the': 2, 'cat': 1, 'and': 1, 'hat': 1}"],
        ["word_count('') == {}", "word_count('a b a') == {'a': 2, 'b': 1}"],
        ["word_count('Hello, hello... HELLO!') == {'hello': 3}", "word_count('one-two') == {'one': 1, 'two': 1}",
         "word_count(\"It's 9 o'clock\") == {'it': 1, 's': 1, 'o': 1, 'clock': 1}"],
        "`word_count('Hello, hello!')` returns {'hello,': 1, 'hello!': 1}; punctuation must not be part of words."),
    "rle_encode": ("textkit", '''
def rle_encode(s):
    """Run-length encode s as each character followed by its run length, e.g. 'aab' -> 'a2b1'."""
    if not s:
        return ""
    out = []
    prev, count = s[0], 1
    for ch in s[1:]:
        if ch == prev:
            count += 1
        else:
            out.append(f"{prev}{count}")
            prev, count = ch, 1
    out.append(f"{prev}{count}")
    return "".join(out)
''', ('    out.append(f"{prev}{count}")\n    return', "    return"),
        ["rle_encode('aaabccdddd') == 'a3b1c2d4'", "rle_encode('z') == 'z1'"],
        ["rle_encode('') == ''"],
        ["rle_encode('aabbbc') == 'a2b3c1'", "rle_encode('abc') == 'a1b1c1'", "rle_encode('x' * 12) == 'x12'"],
        "`rle_encode('aab')` returns 'a2' instead of 'a2b1'; the final run is lost."),
    "camel_to_snake": ("textkit", '''
def camel_to_snake(name):
    """Convert a lowerCamelCase identifier to snake_case."""
    out = []
    for ch in name:
        if ch.isupper():
            out.append("_" + ch.lower())
        else:
            out.append(ch)
    return "".join(out)
''', ('"_" + ch.lower()', '"_" + ch'),
        ["camel_to_snake('camelCaseString') == 'camel_case_string'"],
        ["camel_to_snake('plain') == 'plain'"],
        ["camel_to_snake('myVar') == 'my_var'", "camel_to_snake('parseJsonData') == 'parse_json_data'",
         "camel_to_snake('aB') == 'a_b'"],
        "`camel_to_snake('userId')` returns 'user_Id' instead of 'user_id'."),
    "count_vowels": ("textkit", '''
def count_vowels(s):
    """Count the vowels a, e, i, o, u in s, case-insensitively."""
    return sum(1 for c in s.lower() if c in "aeiou")
''', ("for c in s.lower()", "for c in s"),
        ["count_vowels('AEIOU') == 5", "count_vowels('Hello World') == 3"],
        ["count_vowels('hello world') == 3", "count_vowels('xyz') == 0"],
        ["count_vowels('Programming In Python') == 5", "count_vowels('') == 0", "count_vowels('UPPER lower') == 4"],
        "`count_vowels('Apple')` returns 1 but should be 2; counting must be case-insensitive."),
    "flatten": ("seqkit", '''
def flatten(xs):
    """Flatten arbitrarily nested lists into one flat list."""
    out = []
    for x in xs:
        if isinstance(x, list):
            out.extend(flatten(x))
        else:
            out.append(x)
    return out
''', ("out.extend(flatten(x))", "out.extend(x)"),
        ["flatten([1, [2, [3, 4]], 5]) == [1, 2, 3, 4, 5]"],
        ["flatten([]) == []", "flatten([1, [2, 3]]) == [1, 2, 3]"],
        ["flatten([[[]], [1]]) == [1]", "flatten([[[[7]]]]) == [7]", "flatten(['a', ['b', ['c']]]) == ['a', 'b', 'c']"],
        "`flatten([1, [2, [3]]])` returns [1, 2, [3]]; nesting deeper than one level is not flattened."),
    "merge_intervals": ("seqkit", '''
def merge_intervals(intervals):
    """Merge overlapping [start, end] lists (touching intervals merge) and return them sorted by start."""
    merged = []
    for start, end in sorted(intervals):
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return merged
''', ("start <= merged[-1][1]", "start < merged[-1][1]"),
        ["merge_intervals([[1, 4], [4, 5]]) == [[1, 5]]",
         "merge_intervals([[1, 3], [2, 6], [8, 10], [15, 18]]) == [[1, 6], [8, 10], [15, 18]]"],
        ["merge_intervals([[1, 3], [2, 6], [8, 10], [15, 18]]) == [[1, 6], [8, 10], [15, 18]]", "merge_intervals([]) == []"],
        ["merge_intervals([[8, 10], [1, 3], [2, 6]]) == [[1, 6], [8, 10]]", "merge_intervals([[1, 10], [2, 3]]) == [[1, 10]]",
         "merge_intervals([[5, 6], [1, 2], [2, 5]]) == [[1, 6]]"],
        "`merge_intervals([[1, 2], [2, 3]])` returns [[1, 2], [2, 3]]; touching intervals must merge into [[1, 3]]."),
    "two_sum": ("seqkit", '''
def two_sum(nums, target):
    """Return a tuple (i, j) with i < j such that nums[i] + nums[j] == target; exactly one solution exists."""
    seen = {}
    for j, x in enumerate(nums):
        if target - x in seen:
            return (seen[target - x], j)
        seen[x] = j
    return None
''', ("        if target - x in seen:\n            return (seen[target - x], j)\n        seen[x] = j\n",
      "        seen[x] = j\n        if target - x in seen:\n            return (seen[target - x], j)\n"),
        ["tuple(two_sum([3, 2, 4], 6)) == (1, 2)", "tuple(two_sum([2, 7, 11, 15], 9)) == (0, 1)"],
        ["tuple(two_sum([2, 7, 11, 15], 9)) == (0, 1)", "tuple(two_sum([1, 5, 9], 14)) == (1, 2)"],
        ["tuple(two_sum([3, 3], 6)) == (0, 1)", "tuple(two_sum([0, 4, 3, 0], 0)) == (0, 3)",
         "tuple(two_sum([-1, -2, -3, -4, -5], -8)) == (2, 4)"],
        "`two_sum([5, 1, 5], 10)` returns (0, 0); an element must not be paired with itself."),
    "rotate_list": ("seqkit", '''
def rotate_list(xs, k):
    """Rotate a list right by k positions (k may exceed the length) and return a new list."""
    if not xs:
        return []
    k %= len(xs)
    return xs[-k:] + xs[:-k] if k else list(xs)
''', ("    k %= len(xs)\n", ""),
        ["rotate_list([1, 2, 3, 4, 5], 2) == [4, 5, 1, 2, 3]", "rotate_list([1, 2], 5) == [2, 1]"],
        ["rotate_list([1, 2, 3, 4, 5], 2) == [4, 5, 1, 2, 3]", "rotate_list([], 3) == []"],
        ["rotate_list([1, 2, 3], 3) == [1, 2, 3]", "rotate_list([1, 2, 3], 4) == [3, 1, 2]", "rotate_list([9], 100) == [9]"],
        "`rotate_list([1, 2, 3], 4)` returns [1, 2, 3] instead of [3, 1, 2]; k larger than the length must wrap around."),
    "transpose": ("seqkit", '''
def transpose(matrix):
    """Return the transpose of a rectangular list-of-lists matrix."""
    return [list(row) for row in zip(*matrix)]
''', ("[list(row) for row in zip(*matrix)]",
      "[[matrix[i][j] for i in range(len(matrix))] for j in range(len(matrix))]"),
        ["transpose([[1, 2, 3], [4, 5, 6]]) == [[1, 4], [2, 5], [3, 6]]"],
        ["transpose([[7]]) == [[7]]", "transpose([[1, 2], [3, 4]]) == [[1, 3], [2, 4]]"],
        ["transpose([[1], [2], [3]]) == [[1, 2, 3]]", "transpose([[1, 2, 3]]) == [[1], [2], [3]]"],
        "`transpose([[1, 2, 3]])` returns [[1]] instead of [[1], [2], [3]]; non-square matrices lose columns."),
    "anagram_groups": ("seqkit", '''
def anagram_groups(words):
    """Group anagrams: return a sorted list of groups, each group a sorted list of words."""
    groups = {}
    for word in words:
        groups.setdefault("".join(sorted(word)), []).append(word)
    return sorted(sorted(group) for group in groups.values())
''', ("sorted(sorted(group) for group in groups.values())", "sorted(groups.values())"),
        ["anagram_groups(['eat', 'tea', 'tan', 'ate', 'nat', 'bat']) == [['ate', 'eat', 'tea'], ['bat'], ['nat', 'tan']]"],
        ["anagram_groups([]) == []", "anagram_groups(['b', 'a']) == [['a'], ['b']]"],
        ["anagram_groups(['listen', 'silent', 'enlist', 'google']) == [['enlist', 'listen', 'silent'], ['google']]",
         "anagram_groups(['ab', 'ba', 'ab']) == [['ab', 'ab', 'ba']]"],
        "`anagram_groups(['tea', 'eat'])` returns [['tea', 'eat']]; words inside each group must be sorted."),
    "binary_search": ("seqkit", '''
def binary_search(xs, target):
    """Return the index of target in the sorted list xs, or -1 when absent."""
    lo, hi = 0, len(xs) - 1
    while lo <= hi:
        mid = (lo + hi) // 2
        if xs[mid] == target:
            return mid
        if xs[mid] < target:
            lo = mid + 1
        else:
            hi = mid - 1
    return -1
''', ("while lo <= hi", "while lo < hi"),
        ["binary_search([1, 3, 5, 7, 9], 9) == 4", "binary_search([1, 3, 5], 4) == -1"],
        ["binary_search([1, 3, 5, 7, 9], 7) == 3", "binary_search([], 1) == -1", "binary_search([1, 3, 5], 4) == -1"],
        ["binary_search([2], 2) == 0", "binary_search([1, 3, 5, 7, 9], 1) == 0", "binary_search(list(range(0, 100, 3)), 99) == 33"],
        "`binary_search([4], 4)` returns -1; the last remaining element is never compared."),
    "nth_prime": ("numkit", '''
def nth_prime(n):
    """Return the n-th prime number, 1-indexed."""
    count, candidate = 0, 1
    while count < n:
        candidate += 1
        if all(candidate % p for p in range(2, int(candidate ** 0.5) + 1)):
            count += 1
    return candidate
''', ("int(candidate ** 0.5) + 1", "int(candidate ** 0.5)"),
        ["nth_prime(6) == 13", "nth_prime(1) == 2"],
        ["nth_prime(1) == 2", "nth_prime(2) == 3"],
        ["nth_prime(100) == 541", "nth_prime(10) == 29", "nth_prime(3) == 5"],
        "`nth_prime(3)` returns 4, which is not prime; some composites slip through."),
    "median": ("numkit", '''
def median(xs):
    """Return the median of a non-empty list of numbers (mean of the two middle values for even length)."""
    ys = sorted(xs)
    mid = len(ys) // 2
    return ys[mid] if len(ys) % 2 else (ys[mid - 1] + ys[mid]) / 2
''', ("ys = sorted(xs)", "ys = list(xs)"),
        ["median([3, 1, 2]) == 2", "median([4, 1, 3, 2]) == 2.5"],
        ["median([1, 2, 3]) == 2", "median([5]) == 5"],
        ["median([9, 1, 5, 3]) == 4.0", "median([7, 7, 1]) == 7", "median([-5, 10, 0]) == 0"],
        "`median([5, 1, 3])` returns 1 instead of 3; unsorted input gives wrong results."),
    "gcd_list": ("numkit", '''
def gcd_list(xs):
    """Return the greatest common divisor of a non-empty list of positive integers."""
    result = xs[0]
    for x in xs[1:]:
        result = math.gcd(result, x)
    return result
''', ("for x in xs[1:]", "for x in xs[2:]"),
        ["gcd_list([9, 28]) == 1", "gcd_list([12, 18, 24]) == 6"],
        ["gcd_list([7]) == 7", "gcd_list([5, 5]) == 5"],
        ["gcd_list([100, 75, 50]) == 25", "gcd_list([17, 34]) == 17", "gcd_list([6, 10, 15]) == 1"],
        "`gcd_list([8, 12])` returns 8 instead of 4; the second element is skipped."),
    "digital_root": ("numkit", '''
def digital_root(n):
    """Repeatedly sum the decimal digits of a non-negative integer until one digit remains."""
    while n >= 10:
        n = sum(int(d) for d in str(n))
    return n
''', ("while n >= 10", "while n > 10"),
        ["digital_root(19) == 1", "digital_root(38) == 2"],
        ["digital_root(38) == 2", "digital_root(0) == 0", "digital_root(9875) == 2"],
        ["digital_root(10) == 1", "digital_root(55) == 1", "digital_root(999999999999) == 9"],
        "`digital_root(10)` returns 10 instead of 1."),
    "fizzbuzz": ("numkit", '''
def fizzbuzz(n):
    """Return strings for 1..n: 'Fizz' for multiples of 3, 'Buzz' for multiples of 5, 'FizzBuzz' for both, else the number."""
    out = []
    for i in range(1, n + 1):
        if i % 15 == 0:
            out.append("FizzBuzz")
        elif i % 3 == 0:
            out.append("Fizz")
        elif i % 5 == 0:
            out.append("Buzz")
        else:
            out.append(str(i))
    return out
''', ('        if i % 15 == 0:\n            out.append("FizzBuzz")\n        elif i % 3 == 0:\n            out.append("Fizz")\n'
      '        elif i % 5 == 0:\n            out.append("Buzz")\n',
      '        if i % 3 == 0:\n            out.append("Fizz")\n        elif i % 5 == 0:\n            out.append("Buzz")\n'
      '        elif i % 15 == 0:\n            out.append("FizzBuzz")\n'),
        ["fizzbuzz(15)[-1] == 'FizzBuzz'", "fizzbuzz(5) == ['1', '2', 'Fizz', '4', 'Buzz']"],
        ["fizzbuzz(5) == ['1', '2', 'Fizz', '4', 'Buzz']", "fizzbuzz(0) == []"],
        ["fizzbuzz(30).count('FizzBuzz') == 2", "fizzbuzz(10)[-1] == 'Buzz'", "len(fizzbuzz(100)) == 100"],
        "`fizzbuzz(15)` ends with 'Fizz' instead of 'FizzBuzz'."),
    "roman_to_int": ("parsekit", '''
def roman_to_int(s):
    """Convert a Roman numeral string to an integer."""
    values = {"I": 1, "V": 5, "X": 10, "L": 50, "C": 100, "D": 500, "M": 1000}
    total = 0
    for i, ch in enumerate(s):
        value = values[ch]
        if i + 1 < len(s) and values[s[i + 1]] > value:
            total -= value
        else:
            total += value
    return total
''', ("values[s[i + 1]] > value", "values[s[i + 1]] >= value"),
        ["roman_to_int('III') == 3", "roman_to_int('MCMXCIV') == 1994"],
        ["roman_to_int('MCMXCIV') == 1994", "roman_to_int('IV') == 4", "roman_to_int('XL') == 40"],
        ["roman_to_int('LVIII') == 58", "roman_to_int('MMXXVI') == 2026", "roman_to_int('CDXLIV') == 444"],
        "`roman_to_int('II')` returns 0 instead of 2; repeated numerals are subtracted."),
    "int_to_roman": ("parsekit", '''
def int_to_roman(n):
    """Convert an integer from 1 to 3999 to a Roman numeral string."""
    table = [(1000, "M"), (900, "CM"), (500, "D"), (400, "CD"), (100, "C"), (90, "XC"), (50, "L"),
             (40, "XL"), (10, "X"), (9, "IX"), (5, "V"), (4, "IV"), (1, "I")]
    out = []
    for value, symbol in table:
        while n >= value:
            out.append(symbol)
            n -= value
    return "".join(out)
''', ('(900, "CM"), ', ""),
        ["int_to_roman(1994) == 'MCMXCIV'", "int_to_roman(4) == 'IV'"],
        ["int_to_roman(4) == 'IV'", "int_to_roman(58) == 'LVIII'", "int_to_roman(2026) == 'MMXXVI'"],
        ["int_to_roman(3999) == 'MMMCMXCIX'", "int_to_roman(900) == 'CM'", "int_to_roman(444) == 'CDXLIV'"],
        "`int_to_roman(900)` returns 'DCCCC' instead of 'CM'."),
    "is_balanced": ("parsekit", '''
def is_balanced(s):
    """Return True when the brackets (), [] and {} in s are balanced and properly nested."""
    pairs = {")": "(", "]": "[", "}": "{"}
    stack = []
    for ch in s:
        if ch in "([{":
            stack.append(ch)
        elif ch in pairs:
            if not stack or stack.pop() != pairs[ch]:
                return False
    return not stack
''', ("return not stack", "return True"),
        ["is_balanced('((') is False", "is_balanced('([]{})') is True"],
        ["is_balanced('([]{})') is True", "is_balanced('([)]') is False", "is_balanced('') is True"],
        ["is_balanced('{[(') is False", "is_balanced('a(b)c') is True", "is_balanced(')(') is False"],
        "`is_balanced('(()')` returns True; unclosed brackets must make the result False."),
}

# name: (module, reference source, explicit stub or None, visible asserts, hidden asserts).
# Medium/hard targets: the docstring is a complete specification; visible tests cover the basic
# cases and the hidden tests its stated edge cases.
SPECS = {
    "parse_duration": ("parsekit", '''
def parse_duration(text):
    """Parse a duration such as '1h30m', '2d 4h', '45s' or '1h 5m 10s' into seconds.

    Units are d (86400 s), h (3600 s), m (60 s) and s (1 s), case-insensitive. Each component is
    a non-negative integer followed by its unit; components may be separated by whitespace, must
    appear in the order d, h, m, s and each unit at most once. Leading and trailing whitespace is
    ignored. An empty string, an unknown unit, a number without a unit, a unit without a number,
    and repeated or out-of-order units raise ValueError.
    """
    s = text.strip().lower()
    if not s:
        raise ValueError("empty duration")
    units = {"d": 86400, "h": 3600, "m": 60, "s": 1}
    total, last, pos = 0, -1, 0
    for match in re.finditer(r"\\s*(\\d+)\\s*([a-z]+)\\s*", s):
        number, unit = match.groups()
        if match.start() != pos or unit not in units or "dhms".index(unit) <= last:
            raise ValueError(f"bad duration {text!r}")
        last = "dhms".index(unit)
        total += int(number) * units[unit]
        pos = match.end()
    if pos != len(s):
        raise ValueError(f"bad duration {text!r}")
    return total
''', None,
        ["parse_duration('1h30m') == 5400", "parse_duration('45s') == 45"],
        ["parse_duration('2d 4h') == 187200", "parse_duration(' 1h 5m 10s ') == 3910", "parse_duration('1H') == 3600",
         "parse_duration('0s') == 0", "_raises(parse_duration, '')", "_raises(parse_duration, '5x')",
         "_raises(parse_duration, '1m1h')", "_raises(parse_duration, '1h1h')", "_raises(parse_duration, '10')",
         "_raises(parse_duration, 'h')"]),
    "compare_versions": ("parsekit", '''
def compare_versions(a, b):
    """Compare two semantic versions 'MAJOR.MINOR.PATCH' with an optional '-PRERELEASE' suffix.

    Returns -1 if a < b, 0 if they are equal and 1 if a > b. MAJOR, MINOR and PATCH compare as
    integers (1.10.0 > 1.9.0). A version with a pre-release suffix is lower than the same version
    without one (1.0.0-rc.1 < 1.0.0). Pre-release identifiers are separated by dots and compared
    left to right: two numeric identifiers compare as integers, a numeric identifier is lower than
    an alphanumeric one, and two alphanumeric identifiers compare as strings. When every shared
    identifier is equal, the version with fewer identifiers is lower.
    """
    def split(version):
        main, _, pre = version.partition("-")
        return tuple(int(x) for x in main.split(".")), (pre.split(".") if pre else [])
    (na, pa), (nb, pb) = split(a), split(b)
    if na != nb:
        return -1 if na < nb else 1
    if not pa or not pb:
        return 0 if pa == pb else (-1 if pa else 1)
    for x, y in zip(pa, pb):
        if x == y:
            continue
        if x.isdigit() and y.isdigit():
            return -1 if int(x) < int(y) else 1
        if x.isdigit() != y.isdigit():
            return -1 if x.isdigit() else 1
        return -1 if x < y else 1
    return 0 if len(pa) == len(pb) else (-1 if len(pa) < len(pb) else 1)
''', None,
        ["compare_versions('1.2.3', '1.2.3') == 0", "compare_versions('1.10.0', '1.9.0') == 1"],
        ["compare_versions('1.0.0-alpha', '1.0.0') == -1", "compare_versions('1.0.0', '1.0.0-rc.1') == 1",
         "compare_versions('1.0.0-alpha.1', '1.0.0-alpha.beta') == -1",
         "compare_versions('1.0.0-beta.2', '1.0.0-beta.11') == -1",
         "compare_versions('1.0.0-alpha', '1.0.0-alpha.1') == -1", "compare_versions('2.0.0', '10.0.0') == -1",
         "compare_versions('1.0.0-rc.1', '1.0.0-rc.1') == 0", "compare_versions('0.0.2', '0.0.1') == 1"]),
    "csv_split": ("parsekit", '''
def csv_split(line, sep=","):
    """Split one CSV line into its fields.

    Fields are separated by `sep`. A field that starts with a double quote is quoted: it ends at
    the next unpaired quote, may contain the separator, and a doubled quote ("") inside it stands
    for one quote character. Whitespace is kept as is. An empty line gives [''] and a trailing
    separator gives a final empty field. An unterminated quoted field raises ValueError.
    """
    fields, field, i, quoted = [], [], 0, False
    while i < len(line):
        ch = line[i]
        if quoted:
            if ch == '"' and i + 1 < len(line) and line[i + 1] == '"':
                field.append('"')
                i += 1
            elif ch == '"':
                quoted = False
            else:
                field.append(ch)
        elif ch == '"' and not field:
            quoted = True
        elif ch == sep:
            fields.append("".join(field))
            field = []
        else:
            field.append(ch)
        i += 1
    if quoted:
        raise ValueError("unterminated quoted field")
    fields.append("".join(field))
    return fields
''', None,
        ["csv_split('a,b,c') == ['a', 'b', 'c']", "csv_split('\"x,y\",z') == ['x,y', 'z']"],
        ["csv_split('') == ['']", "csv_split('a,') == ['a', '']",
         "csv_split('\"he said \"\"hi\"\"\",2') == ['he said \"hi\"', '2']", "csv_split(' a , b ') == [' a ', ' b ']",
         "csv_split('a;b', sep=';') == ['a', 'b']", "_raises(csv_split, '\"open')", "csv_split('\"\",x') == ['', 'x']"]),
    "normalize_path": ("textkit", '''
def normalize_path(path):
    """Normalize a POSIX path: collapse repeated slashes, drop '.' segments and resolve '..'.

    An absolute path stays absolute, and '..' at its root is dropped ('/../a' -> '/a'). A relative
    path keeps the leading '..' segments that cannot be resolved ('../a/../../b' -> '../../b'). A
    trailing slash is removed. An empty result is '/' for an absolute path and '.' otherwise.
    """
    absolute = path.startswith("/")
    parts = []
    for segment in path.split("/"):
        if segment in ("", "."):
            continue
        if segment == "..":
            if parts and parts[-1] != "..":
                parts.pop()
            elif not absolute:
                parts.append("..")
            continue
        parts.append(segment)
    joined = "/".join(parts)
    return "/" + joined if absolute else (joined or ".")
''', None,
        ["normalize_path('/a/./b/../c') == '/a/c'", "normalize_path('a//b/') == 'a/b'"],
        ["normalize_path('/../a') == '/a'", "normalize_path('../a/../../b') == '../../b'", "normalize_path('') == '.'",
         "normalize_path('/') == '/'", "normalize_path('./.') == '.'", "normalize_path('a/b/../../..') == '..'",
         "normalize_path('//x//y//') == '/x/y'"]),
    "split_identifier": ("textkit", '''
def split_identifier(name):
    """Split an identifier into lowercase words.

    Handles snake_case, kebab-case, camelCase and PascalCase, including runs of capitals that form
    an acronym ('HTTPServerError' -> ['http', 'server', 'error']). Digits stay attached to the
    word before them ('utf8Decoder' -> ['utf8', 'decoder'], 'parseJSON5Data' -> ['parse', 'json5',
    'data']). Underscores and hyphens separate words and never produce empty words.
    """
    words = []
    for part in re.split(r"[_\\-]+", name):
        words += re.findall(r"[A-Z]+(?![a-z])\\d*|[A-Z]?[a-z]+\\d*|\\d+", part)
    return [w.lower() for w in words]
''', None,
        ["split_identifier('camelCaseName') == ['camel', 'case', 'name']", "split_identifier('snake_case') == ['snake', 'case']"],
        ["split_identifier('HTTPServerError') == ['http', 'server', 'error']",
         "split_identifier('utf8Decoder') == ['utf8', 'decoder']", "split_identifier('getHTTP') == ['get', 'http']",
         "split_identifier('kebab--case') == ['kebab', 'case']", "split_identifier('_private') == ['private']",
         "split_identifier('parseJSON5Data') == ['parse', 'json5', 'data']", "split_identifier('ID') == ['id']"]),
    "word_wrap": ("textkit", '''
def word_wrap(text, width):
    """Wrap text into lines of at most `width` characters.

    Words are separated by any whitespace and joined with single spaces; a line holds as many
    words as fit. Words are never split, except a word longer than `width`: it starts a new line
    and is cut into chunks of exactly `width` characters, each on its own line; its final, shorter
    chunk then counts as an ordinary word. Empty or blank text gives [].
    """
    lines, line = [], ""
    for word in text.split():
        while len(word) > width:
            if line:
                lines.append(line)
                line = ""
            lines.append(word[:width])
            word = word[width:]
        if not word:
            continue
        if not line:
            line = word
        elif len(line) + 1 + len(word) <= width:
            line += " " + word
        else:
            lines.append(line)
            line = word
    if line:
        lines.append(line)
    return lines
''', None,
        ["word_wrap('the quick brown fox', 10) == ['the quick', 'brown fox']"],
        ["word_wrap('', 5) == []", "word_wrap('   ', 5) == []",
         "word_wrap('abcdefghij xy', 4) == ['abcd', 'efgh', 'ij', 'xy']", "word_wrap('a b c d', 3) == ['a b', 'c d']",
         "word_wrap('hello\\tworld\\nnew', 11) == ['hello world', 'new']", "word_wrap('ab cdefg', 3) == ['ab', 'cde', 'fg']",
         "word_wrap('abcd', 4) == ['abcd']"]),
    "topo_sort": ("seqkit", '''
def topo_sort(deps):
    """Order items so that each comes after the items it depends on.

    `deps` maps an item to a list of the items it depends on; an item that appears only as a
    dependency is an item too. Whenever several items are ready, the alphabetically smallest comes
    first, so the order is unique. A dependency cycle (including an item depending on itself)
    raises ValueError.
    """
    import heapq
    nodes = set(deps) | {d for ds in deps.values() for d in ds}
    waiting = {n: 0 for n in nodes}
    users = {n: [] for n in nodes}
    for node, needs in deps.items():
        for need in set(needs):
            waiting[node] += 1
            users[need].append(node)
    ready = [n for n in nodes if waiting[n] == 0]
    heapq.heapify(ready)
    order = []
    while ready:
        node = heapq.heappop(ready)
        order.append(node)
        for user in users[node]:
            waiting[user] -= 1
            if waiting[user] == 0:
                heapq.heappush(ready, user)
    if len(order) != len(nodes):
        raise ValueError("dependency cycle")
    return order
''', None,
        ["topo_sort({'b': ['a'], 'c': ['b']}) == ['a', 'b', 'c']"],
        ["topo_sort({'x': [], 'a': []}) == ['a', 'x']", "topo_sort({'app': ['lib', 'log'], 'lib': ['log']}) == ['log', 'lib', 'app']",
         "_raises(topo_sort, {'a': ['b'], 'b': ['a']})", "topo_sort({}) == []",
         "topo_sort({'c': ['a', 'b'], 'b': ['a'], 'd': ['a']}) == ['a', 'b', 'c', 'd']", "_raises(topo_sort, {'a': ['a']})",
         "topo_sort({'b': ['a', 'a']}) == ['a', 'b']"]),
    "LRUCache": ("seqkit", '''
class LRUCache:
    """A least-recently-used cache with a fixed capacity.

    get(key) returns the stored value, or None for a missing key, and marks the key as most
    recently used. put(key, value) inserts or updates a key and marks it most recently used; when
    the cache then holds more than `capacity` keys, the least recently used key is evicted.
    len(cache) is the number of keys held. A capacity below 1 raises ValueError.
    """

    def __init__(self, capacity):
        if capacity < 1:
            raise ValueError("capacity must be at least 1")
        self.capacity = capacity
        self._data = {}

    def get(self, key):
        if key not in self._data:
            return None
        value = self._data.pop(key)
        self._data[key] = value
        return value

    def put(self, key, value):
        self._data.pop(key, None)
        self._data[key] = value
        while len(self._data) > self.capacity:
            self._data.pop(next(iter(self._data)))

    def __len__(self):
        return len(self._data)
''', '''
class LRUCache:
    """A least-recently-used cache with a fixed capacity.

    get(key) returns the stored value, or None for a missing key, and marks the key as most
    recently used. put(key, value) inserts or updates a key and marks it most recently used; when
    the cache then holds more than `capacity` keys, the least recently used key is evicted.
    len(cache) is the number of keys held. A capacity below 1 raises ValueError.
    """

    def __init__(self, capacity):
        raise NotImplementedError

    def get(self, key):
        raise NotImplementedError

    def put(self, key, value):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError
''',
        ["(lambda c: (c.put('a', 1), c.get('a'))[-1])(LRUCache(2)) == 1"],
        ["(lambda c: (c.put('a', 1), c.put('b', 2), c.put('c', 3), c.get('a'))[-1])(LRUCache(2)) is None",
         "(lambda c: (c.put('a', 1), c.put('b', 2), c.get('a'), c.put('c', 3), c.get('b'))[-1])(LRUCache(2)) is None",
         "(lambda c: (c.put('a', 1), c.put('b', 2), c.get('a'), c.put('c', 3), c.get('a'))[-1])(LRUCache(2)) == 1",
         "(lambda c: (c.put('a', 1), c.put('b', 2), c.put('a', 10), c.put('c', 3), c.get('a'))[-1])(LRUCache(2)) == 10",
         "(lambda c: (c.put('a', 1), c.put('a', 2), len(c))[-1])(LRUCache(3)) == 1",
         "_raises(LRUCache, 0)", "LRUCache(1).get('x') is None"]),
    "luhn_valid": ("numkit", '''
def luhn_valid(number):
    """Return True when `number`, a string of digits that may contain spaces or dashes, passes the
    Luhn checksum. Any other character, or fewer than two digits after removing spaces and dashes,
    makes it invalid (False)."""
    cleaned = number.replace(" ", "").replace("-", "")
    if len(cleaned) < 2 or not cleaned.isdigit():
        return False
    total = 0
    for i, ch in enumerate(reversed(cleaned)):
        digit = int(ch)
        if i % 2 == 1:
            digit *= 2
            if digit > 9:
                digit -= 9
        total += digit
    return total % 10 == 0
''', None,
        ["luhn_valid('4539 1488 0343 6467') is True", "luhn_valid('8273 1232 7352 0569') is False"],
        ["luhn_valid('79927398713') is True", "luhn_valid('0') is False", "luhn_valid('00') is True",
         "luhn_valid('4111-1111-1111-1111') is True", "luhn_valid('4111 1111 1111 1112') is False",
         "luhn_valid('4539a488') is False", "luhn_valid('') is False"]),
    "to_base": ("numkit", '''
def to_base(n, base):
    """Return the integer n written in `base` (2 to 36) with the digits 0-9 then lowercase a-z.

    Negative numbers get a leading '-' and zero is '0'. A base outside 2..36 raises ValueError.
    """
    if not 2 <= base <= 36:
        raise ValueError("base must be between 2 and 36")
    if n == 0:
        return "0"
    digits = "0123456789abcdefghijklmnopqrstuvwxyz"
    sign, n, out = ("-" if n < 0 else ""), abs(n), []
    while n:
        n, remainder = divmod(n, base)
        out.append(digits[remainder])
    return sign + "".join(reversed(out))
''', None,
        ["to_base(255, 16) == 'ff'", "to_base(5, 2) == '101'"],
        ["to_base(0, 7) == '0'", "to_base(-10, 2) == '-1010'", "to_base(35, 36) == 'z'", "to_base(36, 36) == '10'",
         "_raises(to_base, 5, 1)", "_raises(to_base, 5, 37)", "to_base(-255, 16) == '-ff'"]),
    "next_business_day": ("numkit", '''
def next_business_day(day, holidays=()):
    """Return the first date strictly after `day` (a datetime.date) that is a weekday (Monday to
    Friday) and not in `holidays`, an iterable of dates (it may be a generator)."""
    import datetime
    skip = set(holidays)
    current = day + datetime.timedelta(days=1)
    while current.weekday() >= 5 or current in skip:
        current += datetime.timedelta(days=1)
    return current
''', None,
        ["next_business_day(date(2026, 10, 16)) == date(2026, 10, 19)",
         "next_business_day(date(2026, 10, 20)) == date(2026, 10, 21)"],
        ["next_business_day(date(2026, 10, 17)) == date(2026, 10, 19)",
         "next_business_day(date(2026, 12, 24), [date(2026, 12, 25)]) == date(2026, 12, 28)",
         "next_business_day(date(2026, 10, 19), [date(2026, 10, 20), date(2026, 10, 21)]) == date(2026, 10, 22)",
         "next_business_day(date(2026, 10, 19), (d for d in [date(2026, 10, 20)])) == date(2026, 10, 21)"]),
    "eval_expr": ("numkit", '''
def eval_expr(text):
    """Evaluate an arithmetic expression of non-negative integers with + - * /, parentheses and
    unary minus.

    * and / bind tighter than + and -, operators of equal precedence associate to the left, / is
    true division and whitespace is ignored. The result is an int unless a division occurs, then a
    float. Malformed input (an unknown character, unbalanced parentheses, a missing operand or a
    unary plus) raises ValueError; division by zero raises ZeroDivisionError.
    """
    tokens = re.findall(r"\\d+|\\S", text)
    pos = 0

    def peek():
        return tokens[pos] if pos < len(tokens) else None

    def take():
        nonlocal pos
        if pos >= len(tokens):
            raise ValueError("unexpected end of expression")
        pos += 1
        return tokens[pos - 1]

    def expr():
        value = term()
        while peek() in ("+", "-"):
            op, rhs = take(), term()
            value = value + rhs if op == "+" else value - rhs
        return value

    def term():
        value = factor()
        while peek() in ("*", "/"):
            op, rhs = take(), factor()
            value = value * rhs if op == "*" else value / rhs
        return value

    def factor():
        token = take()
        if token == "-":
            return -factor()
        if token == "(":
            value = expr()
            if take() != ")":
                raise ValueError("expected )")
            return value
        if token.isdigit():
            return int(token)
        raise ValueError(f"unexpected {token!r}")

    value = expr()
    if pos != len(tokens):
        raise ValueError("unexpected trailing input")
    return value
''', None,
        ["eval_expr('1 + 2 * 3') == 7", "eval_expr('(1 + 2) * 3') == 9"],
        ["eval_expr('10 / 4') == 2.5", "eval_expr('-(2 + 3) * 2') == -10", "eval_expr('8 - 3 - 2') == 3",
         "eval_expr('2 * -3') == -6", "eval_expr('16 / 4 / 2') == 2.0", "isinstance(eval_expr('7'), int)",
         "isinstance(eval_expr('6 / 3'), float)", "_raises(eval_expr, '(1 + 2')", "_raises(eval_expr, '1 +')",
         "_raises(eval_expr, '2 $ 3')", "_raises(eval_expr, '+5')", "_raises(eval_expr, '1/0', exc=ZeroDivisionError)"]),
}

NOUNS = ["sensor", "invoice", "shipment", "ticket", "session", "account", "order", "device", "batch", "route",
         "widget", "metric", "cluster", "payload", "channel", "ledger", "tenant", "region", "quota", "token",
         "report", "vendor", "parcel", "carrier", "badge", "camera", "meter", "sample", "station", "cohort",
         "contract", "segment", "budget", "voucher", "pallet", "invoice_line", "webhook", "dataset", "rule", "alert"]


def _filler(kind: str, noun: str, rng) -> tuple[str, list[str], str]:
    """A correct helper function, its visible asserts and its name."""
    if kind == "clamp":
        lo = rng.randrange(0, 50)
        hi = lo + rng.randrange(10, 200)
        name = f"clamp_{noun}"
        src = f'''
def {name}(value, low={lo}, high={hi}):
    """Clamp a {noun} reading into the inclusive range [low, high]."""
    if value < low:
        return low
    if value > high:
        return high
    return value
'''
        return src, [f"{name}({lo - 5}) == {lo}", f"{name}({hi + 1}) == {hi}", f"{name}({lo + 1}) == {lo + 1}"], name
    if kind == "record":
        sep = rng.choice([";", "|", ","])
        name = f"parse_{noun}_record"
        src = f'''
def {name}(line):
    """Parse 'key=value' pairs separated by '{sep}' into a dict with stripped keys and values."""
    record = {{}}
    for part in line.split("{sep}"):
        if "=" in part:
            key, value = part.split("=", 1)
            record[key.strip()] = value.strip()
    return record
'''
        return src, [f"{name}('id = 7{sep} name=x') == {{'id': '7', 'name': 'x'}}", f"{name}('') == {{}}"], name
    if kind == "checksum":
        mod = rng.choice([251, 256, 65521, 4093])
        name = f"{noun}_checksum"
        src = f'''
def {name}(data):
    """Return an additive checksum of a string modulo {mod}."""
    return sum(ord(c) for c in data) % {mod}
'''
        return src, [f"{name}('') == 0", f"{name}('ab') == {(97 + 98) % mod}"], name
    if kind == "format":
        width = rng.randrange(4, 12)
        name = f"format_{noun}"
        src = f'''
def {name}(value, width={width}):
    """Right-align a {noun} value in a field of the given width."""
    return str(value).rjust(width)
'''
        return src, [f"{name}(7) == {' ' * (width - 1) + '7'!r}", f"len({name}(123)) == {width}"], name
    if kind == "key":
        joiner = rng.choice(["_", "-", "."])
        name = f"normalize_{noun}_key"
        src = f'''
def {name}(key):
    """Lowercase a {noun} key and replace runs of non-alphanumeric characters with '{joiner}'."""
    return re.sub(r"[^a-z0-9]+", "{joiner}", key.lower()).strip("{joiner}")
'''
        return src, [f"{name}('  Foo Bar!! ') == 'foo{joiner}bar'", f"{name}('x') == 'x'"], name
    if kind == "chunk":
        size = rng.randrange(2, 9)
        name = f"chunk_{noun}s"
        src = f'''
def {name}(items, size={size}):
    """Split items into consecutive lists of at most `size` elements."""
    return [items[i:i + size] for i in range(0, len(items), size)]
'''
        return src, [f"{name}(list(range(5)), 2) == [[0, 1], [2, 3], [4]]", f"{name}([]) == []"], name
    if kind == "ratio":
        digits = rng.randrange(1, 4)
        name = f"{noun}_ratio"
        src = f'''
def {name}(part, total):
    """Return part / total as a percentage rounded to {digits} decimals; 0.0 when total is 0."""
    if total == 0:
        return 0.0
    return round(100.0 * part / total, {digits})
'''
        return src, [f"{name}(1, 3) == {round(100.0 / 3, digits)!r}", f"{name}(5, 0) == 0.0"], name
    name = f"unique_{noun}s"
    src = f'''
def {name}(items):
    """Return items without duplicates, keeping the first occurrence of each."""
    seen = set()
    out = []
    for item in items:
        if item not in seen:
            seen.add(item)
            out.append(item)
    return out
'''
    return src, [f"{name}([3, 1, 3, 2, 1]) == [3, 1, 2]", f"{name}([]) == []"], name


FILLER_KINDS = ["clamp", "record", "checksum", "format", "key", "chunk", "ratio", "unique"]

TEST_HEADER = "from datetime import date\n\n"
RAISES = '''def _raises(fn, *args, exc=ValueError, **kwargs):
    try:
        fn(*args, **kwargs)
    except exc:
        return True
    except Exception:
        return False
    return False
'''

RUNNER = r'''
import importlib.util, re, resource, signal, sys, traceback
from pathlib import Path
for _limit, _value in ((resource.RLIMIT_CPU, 20), (resource.RLIMIT_AS, 2 << 30), (resource.RLIMIT_FSIZE, 16 << 20)):
    try:
        resource.setrlimit(_limit, (_value, _value))
    except (ValueError, OSError):  # macOS rejects RLIMIT_AS
        pass
root = Path.cwd()
sys.path.insert(0, str(root))
def alarm(*_):
    raise TimeoutError("test exceeded 5 s")
signal.signal(signal.SIGALRM, alarm)
files = sorted((root / "tests").glob("test_*.py"))
tests, errors = [], []
for f in files:
    rel = f.relative_to(root).as_posix()
    spec = importlib.util.spec_from_file_location(f.stem, f)
    mod = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(mod)
    except BaseException as e:
        errors.append((rel, f"{type(e).__name__}: {e}"))
        continue
    tests += [(rel, n, getattr(mod, n)) for n in sorted(vars(mod)) if n.startswith("test_") and callable(getattr(mod, n))]
print("============================= test session starts ==============================")
print("rootdir: utilkit")
print(f"collected {len(tests)} items" + (f" / {len(errors)} errors" if errors else ""))
print()
failed = []
for rel, name, fn in tests:
    signal.setitimer(signal.ITIMER_REAL, 5)
    try:
        fn()
        print(f"{rel}::{name} PASSED")
    except BaseException as e:
        frame = traceback.extract_tb(e.__traceback__)[-1]
        line = (frame.line or "").strip()
        detail = f"{type(e).__name__}: {e}" if str(e) else type(e).__name__
        m = re.match(r"assert (.+?) (==|is) (.+)$", line)
        if isinstance(e, AssertionError) and m:
            try:
                tb = e.__traceback__
                while tb.tb_next:
                    tb = tb.tb_next
                got = eval(m.group(1), tb.tb_frame.f_globals, tb.tb_frame.f_locals)
                detail = f"assert {got!r} {m.group(2)} {m.group(3)}"
            except BaseException as inner:
                detail = f"{type(inner).__name__}: {inner}"
        print(f"{rel}::{name} FAILED")
        failed.append((rel, name, line, detail))
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
if failed or errors:
    print()
    print("=================================== FAILURES ===================================")
    for rel, name, line, detail in failed:
        print(f"____ {name} ____")
        print(f"    {line}")
        print(f"E   {detail}")
        print(f"{rel}: {name}")
    for rel, err in errors:
        print(f"ERROR collecting {rel}")
        print(f"E   {err}")
print()
print("=========================== short test summary info ============================")
for rel, name, _, detail in failed:
    print(f"FAILED {rel}::{name} - {detail}")
for rel, err in errors:
    print(f"ERROR {rel} - {err}")
n_pass = len(tests) - len(failed)
parts = ([f"{len(failed)} failed"] if failed else []) + [f"{n_pass} passed"] + ([f"{len(errors)} errors"] if errors else [])
print(f"======================== {', '.join(parts)} ========================")
sys.exit(1 if failed or errors else 0)
'''


def run_suite(files: dict[str, str], timeout: float = 30.0) -> tuple[bool, str]:
    """Run tests/test_*.py against the files in a fresh directory; returns (passed, output)."""
    # ponytail: rlimits (set by the runner before any model code), an unprivileged uid and a
    # temp cwd; no network sandbox. The boundary the hard-evals harness uses for LiveCodeBench.
    # No preexec_fn: it is unsafe beside the agent threads.
    with tempfile.TemporaryDirectory(prefix="swarm-code-") as d:
        root = Path(d)
        for path, content in files.items():
            target = root / path
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(content)
        (root / ".runner.py").write_text(RUNNER)
        for p in [root, *root.rglob("*")]:
            os.chmod(p, 0o777 if p.is_dir() else 0o666)
        try:
            drop = {"user": 65534, "group": 65534, "extra_groups": []} if os.geteuid() == 0 else {}
            done = subprocess.run([sys.executable, "-I", "-B", ".runner.py"], cwd=root, capture_output=True,
                                  text=True, timeout=timeout, **drop)
        except subprocess.TimeoutExpired:
            return False, f"ERROR: the test run exceeded {timeout:.0f} s and was killed."
    out = done.stdout + (("\n" + done.stderr[-2000:]) if done.stderr.strip() else "")
    return done.returncode == 0, out


class CodeTask(Task):
    family = "code"
    max_turns = 30
    tools = [
        tool("list_files", "List every file in the repository with its size in bytes.", {}, []),
        tool("read_file", "Read a text file. Returns numbered lines (1-based).",
             {"path": {"type": "string", "description": "Repository-relative path, e.g. pkg/textkit.py"},
              "offset": {"type": "integer", "description": "First line to return (default 1)"},
              "limit": {"type": "integer", "description": "Maximum number of lines (default 400)"}}, ["path"]),
        tool("grep", "Search files for a Python regular expression; returns path:line: text for each match.",
             {"pattern": {"type": "string"},
              "path": {"type": "string", "description": "Restrict to files under this path prefix"}}, ["pattern"]),
        tool("edit_file", "Replace exactly one occurrence of old_string with new_string in a file. "
             "old_string must match the file exactly (including indentation) and occur exactly once.",
             {"path": {"type": "string"}, "old_string": {"type": "string"}, "new_string": {"type": "string"}},
             ["path", "old_string", "new_string"]),
        tool("write_file", "Create or overwrite a file with the given content.",
             {"path": {"type": "string"}, "content": {"type": "string"}}, ["path", "content"]),
        tool("run_tests", "Run the repository's test suite (tests/test_*.py) and return the report.", {}, []),
    ]
    latency = {"run_tests": 5.0, "list_files": 0.5, "read_file": 0.5, "grep": 0.5, "edit_file": 0.5, "write_file": 0.5}

    def __init__(self, index: int, seed: int):
        super().__init__(index, seed)
        rng = rng_for(self.family, seed, index)
        mode = ["implement", "spec", "fix_report", "spec_report"][index % 4]
        self.tier = {"implement": "easy", "spec": "medium"}.get(mode, "hard")
        simple = sorted(TARGETS)
        if mode == "implement":
            self.stubs, self.bugs, self.reported = [rng.choice(simple)], [], []
        elif mode == "spec":
            self.stubs, self.bugs, self.reported = [rng.choice(sorted(SPECS))], [], []
        elif mode == "fix_report":
            bug = rng.choice(simple)
            other = rng.choice([n for n in simple if TARGETS[n][0] != TARGETS[bug][0]])
            self.stubs, self.bugs, self.reported = [], [bug], [other]
        else:
            stub = rng.choice(sorted(SPECS))
            other = rng.choice([n for n in simple if TARGETS[n][0] != SPECS[stub][0]])
            self.stubs, self.bugs, self.reported = [stub], [], [other]
        self.mode = mode
        self.targets = self.stubs + self.bugs + self.reported
        self.reference, self.files = self._build(rng)
        self.hidden = self._hidden_tests()
        header = self.header()
        done = "Do not modify anything under tests/. When you are done, reply with a one-line summary."
        where = lambda n: f"`{n}` in `pkg/{(TARGETS.get(n) or SPECS[n])[0]}.py`"  # noqa: E731
        if mode == "implement":
            self.prompt = (f"{header} {where(self.stubs[0])} is not implemented yet (it raises NotImplementedError); "
                           f"its docstring is the specification. Implement it so the whole test suite passes. {done}")
        elif mode == "spec":
            self.prompt = (f"{header} {where(self.stubs[0])} is a stub; its docstring is the complete specification, "
                           f"including the edge cases. Implement it. The existing tests cover only the basic cases, "
                           f"so make sure every rule in the docstring holds. {done}")
        elif mode == "fix_report":
            self.prompt = (f"{header} Two problems. The test suite fails; find the cause in the package code. "
                           f"Separately, a user reported: {TARGETS[self.reported[0]][6]} The tests do not cover the "
                           f"reported case. Fix both without breaking other behaviour. {done}")
        else:
            self.prompt = (f"{header} Two items. Implement {where(self.stubs[0])}, a stub whose docstring is the "
                           f"complete specification, including the edge cases (the existing tests cover only basic "
                           f"cases). Also fix this user-reported bug: {TARGETS[self.reported[0]][6]} {done}")

    def _build(self, rng):
        reference, files = {}, {}
        nouns = rng.sample(NOUNS, len(NOUNS))
        fillers = {m: [] for m in MODULES}
        offset = rng.randrange(len(FILLER_KINDS))
        for i, noun in enumerate(nouns):
            fillers[list(MODULES)[i % 4]].append(_filler(FILLER_KINDS[(i + offset) % len(FILLER_KINDS)], noun, rng))
        for module, doc in MODULES.items():
            units = ([("t", n) for n in TARGETS if TARGETS[n][0] == module]
                     + [("s", n) for n in SPECS if SPECS[n][0] == module] + [("f", f) for f in fillers[module]])
            rng.shuffle(units)
            ref_parts, cur_parts, tests = [], [], []
            for kind, unit in units:
                if kind == "f":
                    src, asserts, name = unit
                    src = textwrap.dedent(src).strip("\n") + "\n"
                    ref_parts.append(src)
                    cur_parts.append(src)
                    tests.append((name, asserts))
                    continue
                spec = TARGETS[unit] if kind == "t" else SPECS[unit]
                src = textwrap.dedent(spec[1]).strip("\n") + "\n"
                ref_parts.append(src)
                if unit in self.stubs:
                    if kind == "s" and spec[2]:
                        cur_parts.append(textwrap.dedent(spec[2]).strip("\n") + "\n")
                    else:
                        sig, rest = src.split('"""', 1)
                        cur_parts.append(sig + '"""' + rest.split('"""', 1)[0] + '"""\n    raise NotImplementedError\n')
                elif unit in self.bugs or unit in self.reported:
                    old, new = spec[2]
                    assert src.count(old) == 1, (unit, old)
                    cur_parts.append(src.replace(old, new))
                else:
                    cur_parts.append(src)
                asserts = spec[4] if (kind == "t" and unit in self.reported) else spec[3]
                tests.append((unit, asserts))
            head = f'"""{doc}"""\n\nimport math\nimport re\n\n\n'
            reference[f"pkg/{module}.py"] = head + "\n\n".join(ref_parts)
            files[f"pkg/{module}.py"] = head + "\n\n".join(cur_parts)
            body = "\n\n".join(f"def test_{n}():\n" + "".join(f"    assert {a}\n" for a in asserts) for n, asserts in tests)
            files[f"tests/test_{module}.py"] = (TEST_HEADER + f"from pkg.{module} import (\n    "
                                                + ",\n    ".join(sorted(n for n, _ in tests)) + ",\n)\n\n\n"
                                                + RAISES + "\n\n" + body)
            reference[f"tests/test_{module}.py"] = files[f"tests/test_{module}.py"]
        files["pkg/__init__.py"] = reference["pkg/__init__.py"] = '"""Utility package."""\n'
        files["README.md"] = reference["README.md"] = (
            "# utilkit\n\nSmall utility functions shared by internal services. Modules live in `pkg/`; "
            "every public function has a test in `tests/`. Run the suite with the `run_tests` tool.\n")
        files["pyproject.toml"] = reference["pyproject.toml"] = (
            '[project]\nname = "utilkit"\nversion = "0.4.2"\nrequires-python = ">=3.10"\n')
        return reference, files

    def _hidden_tests(self) -> str:
        lines = [TEST_HEADER + RAISES]
        for name, spec in TARGETS.items():
            lines.append(f"def test_hidden_{name}():\n    from pkg.{spec[0]} import {name}\n"
                         + "".join(f"    assert {a}\n" for a in spec[3] + spec[4] + spec[5]))
        for name, spec in SPECS.items():
            lines.append(f"def test_hidden_{name}():\n    from pkg.{spec[0]} import {name}\n"
                         + "".join(f"    assert {a}\n" for a in spec[3] + spec[4]))
        return "\n\n".join(lines)

    # ------------------------------------------------------------- tools ----

    def _path(self, path: str) -> str:
        path = path.strip().lstrip("./")
        if path.startswith("/") or ".." in path.split("/"):
            raise ToolError(f"path {path!r} is outside the repository")
        return path

    def tool_list_files(self) -> str:
        return "\n".join(f"{p} ({len(c.encode())} bytes)" for p, c in sorted(self.files.items()))

    def tool_read_file(self, path: str, offset: int = 1, limit: int = 400) -> str:
        path = self._path(path)
        if path not in self.files:
            raise ToolError(f"no such file: {path}")
        lines = self.files[path].splitlines()
        start = max(1, offset)
        chosen = lines[start - 1:start - 1 + max(1, limit)]
        if not chosen:
            return f"(file has {len(lines)} lines)"
        return "\n".join(f"{start + i:6d}\t{line}" for i, line in enumerate(chosen))

    def tool_grep(self, pattern: str, path: str = "") -> str:
        try:
            regex = re.compile(pattern)
        except re.error as error:
            raise ToolError(f"invalid regular expression: {error}") from None
        prefix = path.strip().lstrip("./")
        hits = [f"{p}:{n}: {line}" for p, c in sorted(self.files.items()) if p.startswith(prefix)
                for n, line in enumerate(c.splitlines(), 1) if regex.search(line)]
        return "\n".join(hits[:100]) + (f"\n... {len(hits) - 100} more" if len(hits) > 100 else "") if hits else "no matches"

    def _writable(self, path: str) -> str:
        path = self._path(path)
        if path.startswith("tests/"):
            raise ToolError("tests/ is read-only in this task")
        return path

    def tool_edit_file(self, path: str, old_string: str, new_string: str) -> str:
        path = self._writable(path)
        if path not in self.files:
            raise ToolError(f"no such file: {path}")
        count = self.files[path].count(old_string) if old_string else 0
        if count != 1:
            raise ToolError("old_string not found in the file" if count == 0 else
                            f"old_string occurs {count} times; include more surrounding context")
        self.files[path] = self.files[path].replace(old_string, new_string)
        return f"Edited {path}."

    def tool_write_file(self, path: str, content: str) -> str:
        path = self._writable(path)
        self.files[path] = content
        return f"Wrote {len(content.encode())} bytes to {path}."

    def tool_run_tests(self) -> str:
        _, out = run_suite({p: c for p, c in self.files.items()})
        return out[-12000:]

    # ------------------------------------------------------------ checker ---

    def check(self, final: str) -> tuple[bool, str]:
        files = {p: c for p, c in self.files.items() if not p.startswith("tests/")}
        files.update({p: c for p, c in self.reference.items() if p.startswith("tests/")})  # original visible tests
        files["tests/test_zz_hidden.py"] = self.hidden
        ok, out = run_suite(files)
        failed = [line[7:].split(" - ")[0] for line in out.splitlines() if line.startswith("FAILED ")]
        return ok, "all tests pass" if ok else ("failing: " + ", ".join(failed[:6]) if failed else out[-300:])

    def oracle(self) -> list[Step]:
        modules = sorted({f"pkg/{(TARGETS.get(t) or SPECS[t])[0]}.py" for t in self.targets})
        return [Step(calls=[Call("run_tests", {})]),
                Step(calls=[Call("read_file", {"path": m}) for m in modules]),
                Step(calls=[Call("write_file", {"path": m, "content": self.reference[m]}) for m in modules]),
                Step(calls=[Call("run_tests", {})]),
                Step(content="Fixed " + ", ".join(self.targets) + "; the suite passes.")]
