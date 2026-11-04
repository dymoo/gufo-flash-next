"""Compact task eval for the Flash-Next QA suite: executable code, exact math,
tool calls and JSON output. Every case is graded mechanically."""

from __future__ import annotations

import json
import re
import resource
import subprocess
import sys
import tempfile

CODE = [
    ("is_palindrome", "is_palindrome(s) returns True when s reads the same backwards, ignoring case and non-alphanumeric characters",
     ["assert is_palindrome('A man, a plan, a canal: Panama') is True", "assert is_palindrome('race a car') is False", "assert is_palindrome('') is True"]),
    ("fizzbuzz", "fizzbuzz(n) returns a list of strings for 1..n: 'Fizz' for multiples of 3, 'Buzz' for multiples of 5, 'FizzBuzz' for both, else the number",
     ["assert fizzbuzz(5) == ['1', '2', 'Fizz', '4', 'Buzz']", "assert fizzbuzz(15)[-1] == 'FizzBuzz'", "assert fizzbuzz(0) == []"]),
    ("flatten", "flatten(xs) flattens arbitrarily nested lists into one flat list",
     ["assert flatten([1, [2, [3, 4]], 5]) == [1, 2, 3, 4, 5]", "assert flatten([]) == []", "assert flatten([[[]], [1]]) == [1]"]),
    ("word_count", "word_count(text) returns a dict mapping each lowercase word to its count; words are maximal runs of letters",
     ["assert word_count('The cat and the hat.') == {'the': 2, 'cat': 1, 'and': 1, 'hat': 1}", "assert word_count('') == {}"]),
    ("merge_intervals", "merge_intervals(intervals) merges overlapping [start, end] lists (touching intervals merge) and returns them sorted by start",
     ["assert merge_intervals([[1, 3], [2, 6], [8, 10], [15, 18]]) == [[1, 6], [8, 10], [15, 18]]", "assert merge_intervals([[1, 4], [4, 5]]) == [[1, 5]]", "assert merge_intervals([]) == []"]),
    ("roman_to_int", "roman_to_int(s) converts a Roman numeral string to an integer",
     ["assert roman_to_int('MCMXCIV') == 1994", "assert roman_to_int('III') == 3", "assert roman_to_int('LVIII') == 58"]),
    ("int_to_roman", "int_to_roman(n) converts an integer from 1 to 3999 to a Roman numeral string",
     ["assert int_to_roman(1994) == 'MCMXCIV'", "assert int_to_roman(3999) == 'MMMCMXCIX'", "assert int_to_roman(4) == 'IV'"]),
    ("two_sum", "two_sum(nums, target) returns a tuple (i, j) with i < j such that nums[i] + nums[j] == target; exactly one solution exists",
     ["assert tuple(two_sum([2, 7, 11, 15], 9)) == (0, 1)", "assert tuple(two_sum([3, 2, 4], 6)) == (1, 2)"]),
    ("rle_encode", "rle_encode(s) run-length encodes a string as character followed by count, e.g. 'aab' -> 'a2b1'",
     ["assert rle_encode('aaabccdddd') == 'a3b1c2d4'", "assert rle_encode('') == ''", "assert rle_encode('z') == 'z1'"]),
    ("is_balanced", "is_balanced(s) returns True when the brackets (), [] and {} in s are balanced and properly nested",
     ["assert is_balanced('([]{})') is True", "assert is_balanced('([)]') is False", "assert is_balanced('((') is False", "assert is_balanced('') is True"]),
    ("nth_prime", "nth_prime(n) returns the n-th prime number, 1-indexed",
     ["assert nth_prime(1) == 2", "assert nth_prime(6) == 13", "assert nth_prime(100) == 541"]),
    ("transpose", "transpose(matrix) returns the transpose of a rectangular list-of-lists matrix",
     ["assert transpose([[1, 2, 3], [4, 5, 6]]) == [[1, 4], [2, 5], [3, 6]]", "assert transpose([[7]]) == [[7]]"]),
    ("binary_search", "binary_search(xs, target) returns the index of target in the sorted list xs, or -1 when absent",
     ["assert binary_search([1, 3, 5, 7, 9], 7) == 3", "assert binary_search([1, 3, 5], 4) == -1", "assert binary_search([], 1) == -1"]),
    ("camel_to_snake", "camel_to_snake(name) converts a lowerCamelCase identifier to snake_case",
     ["assert camel_to_snake('camelCaseString') == 'camel_case_string'", "assert camel_to_snake('myVar') == 'my_var'", "assert camel_to_snake('plain') == 'plain'"]),
    ("median", "median(xs) returns the median of a non-empty list of numbers (mean of the two middle values for even length)",
     ["assert median([3, 1, 2]) == 2", "assert median([4, 1, 3, 2]) == 2.5"]),
    ("gcd_list", "gcd_list(xs) returns the greatest common divisor of a non-empty list of positive integers",
     ["assert gcd_list([12, 18, 24]) == 6", "assert gcd_list([7]) == 7", "assert gcd_list([9, 28]) == 1"]),
    ("anagram_groups", "anagram_groups(words) groups anagrams: return a sorted list of groups, each group a sorted list of words",
     ["assert anagram_groups(['eat', 'tea', 'tan', 'ate', 'nat', 'bat']) == [['ate', 'eat', 'tea'], ['bat'], ['nat', 'tan']]"]),
    ("digital_root", "digital_root(n) repeatedly sums the decimal digits of a non-negative integer until one digit remains",
     ["assert digital_root(38) == 2", "assert digital_root(0) == 0", "assert digital_root(9875) == 2"]),
    ("count_vowels", "count_vowels(s) counts the vowels a, e, i, o, u in s, case-insensitively",
     ["assert count_vowels('Hello World') == 3", "assert count_vowels('xyz') == 0", "assert count_vowels('AEIOU') == 5"]),
    ("rotate_list", "rotate_list(xs, k) rotates a list right by k positions (k may exceed the length)",
     ["assert rotate_list([1, 2, 3, 4, 5], 2) == [4, 5, 1, 2, 3]", "assert rotate_list([1, 2], 5) == [2, 1]", "assert rotate_list([], 3) == []"]),
]

MATH = [
    ("A baker makes 12 trays of cookies with 24 cookies on each tray. She sells 230 cookies. How many cookies are left?", 58),
    ("Tom has 3 times as many marbles as Jerry. Together they have 64 marbles. How many marbles does Tom have?", 48),
    ("A train travels at 75 km per hour for 4 hours and then at 60 km per hour for 2.5 hours. How many kilometers does it travel in total?", 450),
    ("A shirt costs $40. It is discounted by 25%, and then a $5 coupon is applied. What is the final price in dollars?", 25),
    ("The sum of three consecutive integers is 141. What is the largest of the three?", 48),
    ("A rectangle has a perimeter of 56 cm and a length of 18 cm. What is its area in square centimeters?", 180),
    ("Maria reads 35 pages per day. Her book has 420 pages. After 8 days, how many pages does she still have to read?", 140),
    ("A farmer has 17 cows and 23 chickens. How many legs do the animals have in total?", 114),
    ("What is the remainder when 2 to the power of 10 is divided by 7?", 2),
    ("A school has 540 students and 45% of them are boys. How many girls are there?", 297),
    ("If 5 workers can build a wall in 12 days, how many days would 6 workers take, working at the same rate?", 10),
    ("A 1200 liter tank is filled at 45 liters per minute while it leaks 15 liters per minute. Starting empty, how many minutes until it is full?", 40),
    ("What is the least common multiple of 12, 18 and 30?", 180),
    ("A store sells pencils only in packs of 12 for $3. How many dollars does it cost to get at least 150 pencils?", 39),
    ("Anna is 4 years older than Ben. In 6 years, the sum of their ages will be 50. How old is Ben now?", 17),
    ("How many positive divisors does 360 have?", 24),
    ("A car uses 7 liters of fuel per 100 km and fuel costs $2 per liter. How many dollars does the fuel cost for a 1500 km trip?", 210),
    ("The average of five numbers is 18. Four of the numbers are 12, 15, 20 and 25. What is the fifth number?", 18),
    ("A cyclist rides 18 km in 45 minutes. At the same speed, how many kilometers will she ride in 2 hours?", 48),
    ("What is the sum of all integers from 1 to 200 that are divisible by 7?", 2842),
]

TOOLS = [
    {"type": "function", "function": {"name": "get_weather", "description": "Current weather for a city.", "parameters": {
        "type": "object", "properties": {"city": {"type": "string"}, "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}},
        "required": ["city", "unit"]}}},
    {"type": "function", "function": {"name": "convert_currency", "description": "Convert an amount between currencies (ISO codes).", "parameters": {
        "type": "object", "properties": {"amount": {"type": "number"}, "from_currency": {"type": "string"}, "to_currency": {"type": "string"}},
        "required": ["amount", "from_currency", "to_currency"]}}},
    {"type": "function", "function": {"name": "create_event", "description": "Create a calendar event.", "parameters": {
        "type": "object", "properties": {"title": {"type": "string"}, "date": {"type": "string", "description": "YYYY-MM-DD"},
                                         "duration_minutes": {"type": "integer"}},
        "required": ["title", "date", "duration_minutes"]}}},
    {"type": "function", "function": {"name": "search_flights", "description": "Search flights for a date.", "parameters": {
        "type": "object", "properties": {"origin": {"type": "string"}, "destination": {"type": "string"}, "date": {"type": "string", "description": "YYYY-MM-DD"}},
        "required": ["origin", "destination", "date"]}}},
]

# (prompt, function, {argument: expected}); a list expects any listed substring (case-insensitive).
TOOL_CASES = [
    ("What's the weather in Paris right now, in celsius?", "get_weather", {"city": ["paris"], "unit": "celsius"}),
    ("Convert 250 US dollars to euros.", "convert_currency", {"amount": 250, "from_currency": ["usd", "dollar"], "to_currency": ["eur"]}),
    ("Schedule a meeting called 'Budget review' on 2026-10-05 for 45 minutes.", "create_event",
     {"title": ["budget review"], "date": "2026-10-05", "duration_minutes": 45}),
    ("Find flights from Tokyo to Seoul on 2026-11-20.", "search_flights", {"origin": ["tokyo", "hnd", "nrt"], "destination": ["seoul", "icn", "gmp"], "date": "2026-11-20"}),
    ("How hot is it in Phoenix? Use fahrenheit.", "get_weather", {"city": ["phoenix"], "unit": "fahrenheit"}),
    ("I need 1000 Japanese yen converted into British pounds.", "convert_currency", {"amount": 1000, "from_currency": ["jpy", "yen"], "to_currency": ["gbp", "pound"]}),
    ("Book a 90 minute event titled Team offsite on 2026-12-01.", "create_event", {"title": ["team offsite"], "date": "2026-12-01", "duration_minutes": 90}),
    ("Are there flights from Berlin to Madrid on 2027-01-15?", "search_flights", {"origin": ["berlin", "ber"], "destination": ["madrid", "mad"], "date": "2027-01-15"}),
]

JSON_CASES = [
    ("Return a JSON object describing the novel Pride and Prejudice with keys \"title\" (string), \"year\" (integer, first publication) and \"authors\" (array of strings). Output only the JSON.",
     lambda v: "pride" in v["title"].lower() and v["year"] == 1813 and "austen" in v["authors"][0].lower()),
    ("Output a JSON array of the first five prime numbers and nothing else.", lambda v: v == [2, 3, 5, 7, 11]),
    ("Give a JSON object with keys \"city\" and \"country\" for the capital of Japan. Output only JSON.",
     lambda v: "tokyo" in v["city"].lower() and "japan" in v["country"].lower()),
    ("Convert to a JSON object with keys name, age and email: John Smith, 34 years old, john@example.com. Output only JSON.",
     lambda v: v == {"name": "John Smith", "age": 34, "email": "john@example.com"}),
    ("Return a JSON object mapping each of the words apple, kiwi and banana to its length. Output only JSON.",
     lambda v: v == {"apple": 5, "kiwi": 4, "banana": 6}),
    ("Produce a JSON object with key \"items\": a list of three objects, each with an integer \"id\" (1, 2, 3 in order) and a string \"label\". Output only JSON.",
     lambda v: [i["id"] for i in v["items"]] == [1, 2, 3] and all(isinstance(i["label"], str) for i in v["items"])),
]


def requests() -> list[dict]:
    """Every task as {id, kind, body, grade}; body is a chat request fragment."""
    out = []
    for name, spec, tests in CODE:
        prompt = (f"Write a Python function {spec}. Use only the standard library. "
                  "Reply with a single ```python code block containing the function and nothing else.")
        out.append({"id": f"code-{name}", "kind": "code", "body": {"messages": [{"role": "user", "content": prompt}], "max_tokens": 768},
                    "grade": lambda r, tests=tests: run_python(extract_code(content(r)), tests)})
    for index, (question, answer) in enumerate(MATH):
        prompt = question + "\nSolve it with brief reasoning, then give the final line as 'Answer: <integer>'."
        out.append({"id": f"math-{index:02d}", "kind": "math", "body": {"messages": [{"role": "user", "content": prompt}], "max_tokens": 512},
                    "grade": lambda r, answer=answer: extract_integer(content(r)) == answer})
    for index, (prompt, function, expected) in enumerate(TOOL_CASES):
        out.append({"id": f"tool-{index:02d}", "kind": "tool", "body": {
            "messages": [{"role": "user", "content": prompt}], "tools": TOOLS, "tool_choice": "auto", "max_tokens": 256},
            "grade": lambda r, function=function, expected=expected: tool_ok(r, function, expected)})
    for index, (prompt, check) in enumerate(JSON_CASES):
        out.append({"id": f"json-{index:02d}", "kind": "json", "body": {"messages": [{"role": "user", "content": prompt}], "max_tokens": 256},
                    "grade": lambda r, check=check: json_ok(content(r), check)})
    return out


def content(response: dict) -> str:
    return response["choices"][0]["message"].get("content") or ""


def extract_code(text: str) -> str:
    match = re.search(r"```(?:python|py)?\s*\n(.*?)```", text, re.S)
    return match.group(1) if match else text


def extract_integer(text: str) -> int | None:
    """First number after the last 'Answer:', else the last number in the text."""
    answered = "Answer:" in text
    numbers = re.findall(r"-?\d[\d,]*(?:\.\d+)?", text.rsplit("Answer:", 1)[1] if answered else text)
    if not numbers:
        return None
    value = float(numbers[0 if answered else -1].replace(",", ""))
    return int(value) if value.is_integer() else None


def _limits() -> None:
    for limit, value in ((resource.RLIMIT_CPU, 10), (resource.RLIMIT_AS, 1 << 30)):
        try:
            resource.setrlimit(limit, (value, value))
        except (ValueError, OSError):  # macOS rejects RLIMIT_AS; Linux applies both
            pass


def run_python(code: str, tests: list[str], timeout: float = 20.0) -> bool:
    # ponytail: rlimits + temp cwd only, no network/fs sandbox; the prompts are fixed and benign.
    source = code + "\n\n" + "\n".join(tests) + "\nprint('QA_OK')\n"
    with tempfile.TemporaryDirectory() as cwd:
        try:
            done = subprocess.run([sys.executable, "-I", "-c", source], cwd=cwd, capture_output=True, text=True,
                                  timeout=timeout, preexec_fn=_limits)
        except subprocess.TimeoutExpired:
            return False
    return done.returncode == 0 and done.stdout.strip().endswith("QA_OK")


def _matches(value, expected) -> bool:
    if isinstance(expected, list):
        return isinstance(value, str) and any(item in value.lower() for item in expected)
    if isinstance(expected, (int, float)) and not isinstance(value, bool) and isinstance(value, (int, float)):
        return float(value) == float(expected)
    return isinstance(value, str) and value.strip().lower() == str(expected).lower()


def tool_ok(response: dict, function: str, expected: dict) -> bool:
    calls = response["choices"][0]["message"].get("tool_calls") or []
    if len(calls) != 1 or calls[0]["function"]["name"] != function:
        return False
    try:
        arguments = json.loads(calls[0]["function"]["arguments"])
    except (TypeError, ValueError):
        return False
    return all(_matches(arguments.get(key), value) for key, value in expected.items())


def json_ok(text: str, check) -> bool:
    text = text.strip()
    fenced = re.fullmatch(r"```(?:json)?\s*\n(.*?)\n?```", text, re.S)
    try:
        return bool(check(json.loads(fenced.group(1) if fenced else text)))
    except (ValueError, KeyError, TypeError, IndexError, AttributeError):
        return False
