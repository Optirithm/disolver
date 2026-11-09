"""Randomised tests: every VersionSet operation is mirrored on a plain Python
set of concrete versions (the "oracle"), and the two must always agree."""
import itertools
import random

from versionset import Version, VersionSet

UNIVERSE = [Version((a, b, c)) for a, b, c in itertools.product(range(4), range(3), range(3))]


def members(vs: VersionSet) -> frozenset:
    return frozenset(v for v in UNIVERSE if v in vs)


def check_invariants(vs: VersionSet):
    ivs = vs.intervals
    for iv in ivs:
        assert not iv.is_empty(), f"empty interval in {vs!r}"
    for a, b in zip(ivs, ivs[1:]):
        # strictly increasing, and a gap between them (not overlapping, not touching)
        assert a.hi_key < b.lo_key, f"intervals overlap or touch in {vs!r}"


def random_primitive(rng):
    kind = rng.choice(["empty", "full", "exact", "at_least", "greater_than",
                       "at_most", "less_than", "between"])
    b, c = rng.choice(UNIVERSE), rng.choice(UNIVERSE)
    lo, hi = sorted([b, c])
    if kind == "empty":
        return VersionSet.empty(), frozenset()
    if kind == "full":
        return VersionSet.full(), frozenset(UNIVERSE)
    if kind == "exact":
        return VersionSet.exact(b), frozenset(v for v in UNIVERSE if v == b)
    if kind == "at_least":
        return VersionSet.at_least(b), frozenset(v for v in UNIVERSE if v >= b)
    if kind == "greater_than":
        return VersionSet.greater_than(b), frozenset(v for v in UNIVERSE if v > b)
    if kind == "at_most":
        return VersionSet.at_most(b), frozenset(v for v in UNIVERSE if v <= b)
    if kind == "less_than":
        return VersionSet.less_than(b), frozenset(v for v in UNIVERSE if v < b)
    return VersionSet.between(lo, hi), frozenset(v for v in UNIVERSE if lo <= v < hi)


def random_pair(rng, depth=4):
    """Build a random expression; apply the same operations to the set and the oracle."""
    if depth == 0 or rng.random() < 0.25:
        return random_primitive(rng)
    op = rng.choice(["and", "or", "not", "sub"])
    a, oa = random_pair(rng, depth - 1)
    if op == "not":
        return ~a, frozenset(UNIVERSE) - oa
    b, ob = random_pair(rng, depth - 1)
    if op == "and":
        return a & b, oa & ob
    if op == "or":
        return a | b, oa | ob
    return a - b, oa - ob


def test_matches_oracle_and_keeps_invariants():
    rng = random.Random(1234)
    for _ in range(3000):
        vs, oracle = random_pair(rng)
        check_invariants(vs)
        assert members(vs) == oracle, f"{vs!r} disagrees with oracle"


def test_algebraic_laws():
    rng = random.Random(99)
    empty, full = VersionSet.empty(), VersionSet.full()
    for _ in range(1500):
        a, _ = random_pair(rng, 3)
        b, _ = random_pair(rng, 3)
        c, _ = random_pair(rng, 3)
        assert ~~a == a                                   # double negation
        assert a & b == b & a and a | b == b | a          # commutative
        assert ~(a & b) == ~a | ~b                        # De Morgan
        assert ~(a | b) == ~a & ~b
        assert a & (b | c) == (a & b) | (a & c)           # distributive
        assert a & ~a == empty and a | ~a == full         # complement laws
        assert a & full == a and a | empty == a           # identities
        assert (a - b).is_disjoint(b)
        assert a.is_subset(a | b) and (a & b).is_subset(a)


def test_constraint_parsing():
    cases = {
        "*":          lambda v: True,
        ">=1.1.0":    lambda v: v >= (1, 1, 0),
        ">1.1.0":     lambda v: v > (1, 1, 0),
        "<=2.0.0":    lambda v: v <= (2, 0, 0),
        "<2":         lambda v: v < (2, 0, 0),
        "==1.1.0":    lambda v: v == (1, 1, 0),
        "1.1.0":      lambda v: v == (1, 1, 0),
        "!=1.1.0":    lambda v: v != (1, 1, 0),
        ">=1.0,<2":   lambda v: (1, 0, 0) <= v < (2, 0, 0),
        "^1.2.0":     lambda v: (1, 2, 0) <= v < (2, 0, 0),
        "^0.2.0":     lambda v: (0, 2, 0) <= v < (0, 3, 0),
        "^0.0.1":     lambda v: (0, 0, 1) <= v < (0, 0, 2),
        "~1.1":       lambda v: (1, 1, 0) <= v < (1, 2, 0),
        ">=1.0,<1.0": lambda v: False,
    }
    for spec, predicate in cases.items():
        vs = VersionSet.from_constraint(spec)
        check_invariants(vs)
        assert members(vs) == frozenset(v for v in UNIVERSE if predicate(v)), spec


def test_diamond_dependency():
    # app -> B (needs D >=1.0,<2)   app -> C (needs D >=1.5)
    from_b = VersionSet.from_constraint(">=1.0,<2")
    from_c = VersionSet.from_constraint(">=1.5")
    both = from_b & from_c
    assert str(both) == ">=1.5.0,<2.0.0"
    assert both.best(["1.0.0", "1.5.0", "1.9.2", "2.0.0"]) == Version((1, 9, 2))

    # the conflicting diamond: B needs D <2, C needs D >=2
    clash = VersionSet.from_constraint("<2") & VersionSet.from_constraint(">=2")
    assert clash.is_empty()


def test_str_and_binary_search():
    vs = VersionSet.from_constraint("<1.0") | VersionSet.from_constraint(">=2.0,<3.0") | VersionSet.exact("5.0.0")
    assert str(vs) == "<1.0.0 | >=2.0.0,<3.0.0 | ==5.0.0"
    assert str(VersionSet.exact("1.2.3").complement()) == "!=1.2.3"
    assert str(VersionSet.empty()) == "∅" and str(VersionSet.full()) == "*"
    for v, expected in [("0.9.9", True), ("1.0.0", False), ("2.0.0", True),
                        ("2.9.9", True), ("3.0.0", False), ("5.0.0", True), ("5.0.1", False)]:
        assert (v in vs) is expected, v


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"ok  {name}")
