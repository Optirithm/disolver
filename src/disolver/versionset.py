"""versionset - an algebra of version ranges.

A VersionSet is a *set of versions*, stored as a sorted list of disjoint
intervals.  It supports the operations a resolver needs:

    a & b      intersection   ("must satisfy both")
    a | b      union          ("either is fine")
    ~a         complement     ("everything NOT in a")
    a - b      difference
    v in a     membership     (binary search)
    a.is_subset(b), a.is_disjoint(b), a.is_empty()

Invariants (this is what makes everything else easy):
    1. intervals are sorted by lower bound
    2. no interval is empty
    3. intervals never overlap AND never touch  ([1,2) and [2,3) are merged)

Because of (3) every set has exactly ONE representation, so `a == b` is a
plain structural comparison.

Versions are treated as a dense line (we never assume "the next version
after 1.0.0 is 1.0.1"), which keeps the algebra valid for PEP 440 style
pre/post/dev releases too.
"""
from __future__ import annotations

import bisect
from dataclasses import dataclass


# --------------------------------------------------------------------------- #
# Version
# --------------------------------------------------------------------------- #
class Version(tuple):
    """(major, minor, patch). Tuples already compare lexicographically."""

    @classmethod
    def parse(cls, text: str) -> "Version":
        try:
            parts = tuple(int(p) for p in text.strip().split("."))
        except ValueError:
            raise ValueError(f"invalid version: {text!r}") from None
        if not 1 <= len(parts) <= 3:
            raise ValueError(f"invalid version: {text!r}")
        return cls(parts + (0,) * (3 - len(parts)))

    def bump(self, idx: int) -> "Version":
        """Increment position idx and zero everything after it."""
        parts = list(self[: idx + 1])
        parts[idx] += 1
        return Version(parts + [0] * (2 - idx))

    def __str__(self) -> str:
        return ".".join(map(str, self))

    def __repr__(self) -> str:
        return f"Version({self})"


def _v(x) -> Version:
    return x if isinstance(x, Version) else Version.parse(x)


# --------------------------------------------------------------------------- #
# Interval:  one contiguous stretch of the version line
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Interval:
    """lo/hi of None mean -infinity / +infinity.  *_inc says if the end is included."""

    lo: Version | None
    lo_inc: bool
    hi: Version | None
    hi_inc: bool

    # Each bound is mapped to a sortable key so that "where does this interval
    # start/end on the line" becomes a plain tuple comparison.
    #   lower bound  [v  starts exactly at v      -> (1, v, 0)
    #                (v  starts just after v      -> (1, v, 1)
    #   upper bound  v]  ends just after v        -> (1, v, 1)
    #                v)  ends just before v       -> (1, v, 0)
    #   -inf = (0,)   +inf = (2,)
    @property
    def lo_key(self) -> tuple:
        return (0,) if self.lo is None else (1, self.lo, 0 if self.lo_inc else 1)

    @property
    def hi_key(self) -> tuple:
        return (2,) if self.hi is None else (1, self.hi, 1 if self.hi_inc else 0)

    def is_empty(self) -> bool:
        return self.hi_key <= self.lo_key

    def intersect(self, other: "Interval") -> "Interval | None":
        a = self if self.lo_key >= other.lo_key else other   # the later start wins
        b = self if self.hi_key <= other.hi_key else other   # the earlier end wins
        result = Interval(a.lo, a.lo_inc, b.hi, b.hi_inc)
        return None if result.is_empty() else result


# --------------------------------------------------------------------------- #
# VersionSet
# --------------------------------------------------------------------------- #
class VersionSet:
    __slots__ = ("intervals",)

    def __init__(self, intervals=()):
        # Callers inside this module guarantee the invariants; use the
        # constructors below (exact, at_least, ...) from outside.
        self.intervals: tuple[Interval, ...] = tuple(intervals)

    # ---- constructors ----------------------------------------------------- #
    @classmethod
    def empty(cls) -> "VersionSet":
        return cls()

    @classmethod
    def full(cls) -> "VersionSet":
        return cls([Interval(None, False, None, False)])

    @classmethod
    def _single(cls, iv: Interval) -> "VersionSet":
        return cls() if iv.is_empty() else cls([iv])

    @classmethod
    def exact(cls, v) -> "VersionSet":
        v = _v(v)
        return cls._single(Interval(v, True, v, True))

    @classmethod
    def at_least(cls, v) -> "VersionSet":        # >= v
        return cls._single(Interval(_v(v), True, None, False))

    @classmethod
    def greater_than(cls, v) -> "VersionSet":    # > v
        return cls._single(Interval(_v(v), False, None, False))

    @classmethod
    def at_most(cls, v) -> "VersionSet":         # <= v
        return cls._single(Interval(None, False, _v(v), True))

    @classmethod
    def less_than(cls, v) -> "VersionSet":       # < v
        return cls._single(Interval(None, False, _v(v), False))

    @classmethod
    def between(cls, lo, hi) -> "VersionSet":    # >= lo, < hi
        return cls._single(Interval(_v(lo), True, _v(hi), False))

    @classmethod
    def from_constraint(cls, spec: str) -> "VersionSet":
        """'>=1.0,<2' | '^1.2.3' | '~1.2' | '!=1.1' | '1.4.0' | '*'"""
        result = cls.full()
        for part in (p.strip() for p in spec.split(",")):
            if part in ("", "*"):
                continue
            result = result & cls._clause(part)
        return result

    @classmethod
    def _clause(cls, part: str) -> "VersionSet":
        if part[0] == "^":   # keep the left-most non-zero component fixed
            v = Version.parse(part[1:])
            idx = next((i for i, n in enumerate(v) if n != 0), 2)
            return cls.between(v, v.bump(idx))
        if part[0] == "~":   # patch-level changes only
            v = Version.parse(part[1:])
            return cls.between(v, v.bump(1))
        ops = [(">=", cls.at_least), ("<=", cls.at_most), ("==", cls.exact),
               ("!=", lambda v: cls.exact(v).complement()),
               (">", cls.greater_than), ("<", cls.less_than)]  # 2-char ops first
        for op, make in ops:
            if part.startswith(op):
                return make(part[len(op):])
        return cls.exact(part)

    # ---- core algebra ----------------------------------------------------- #
    def complement(self) -> "VersionSet":
        """Return the gaps between our intervals (plus the two infinite ends)."""
        out: list[Interval] = []
        cur_lo, cur_inc = None, False            # start of the next gap: -infinity
        for iv in self.intervals:
            if iv.lo is not None:                # gap: [cursor ... iv.lo)
                out.append(Interval(cur_lo, cur_inc, iv.lo, not iv.lo_inc))
            if iv.hi is None:                    # we run to +inf: no gap after us
                return VersionSet(out)
            cur_lo, cur_inc = iv.hi, not iv.hi_inc
        out.append(Interval(cur_lo, cur_inc, None, False))
        return VersionSet(out)

    def intersection(self, other: "VersionSet") -> "VersionSet":
        """Two-pointer sweep: O(len(a) + len(b))."""
        a, b = self.intervals, other.intervals
        out: list[Interval] = []
        i = j = 0
        while i < len(a) and j < len(b):
            both = a[i].intersect(b[j])
            if both is not None:
                out.append(both)
            # whichever interval ends first can't overlap anything further
            if a[i].hi_key < b[j].hi_key:
                i += 1
            else:
                j += 1
        return VersionSet(out)

    def union(self, other: "VersionSet") -> "VersionSet":
        # De Morgan:  a | b  ==  ~(~a & ~b).  Re-using the two operations above
        # means merging of overlapping/touching intervals comes for free.
        return (self.complement() & other.complement()).complement()

    def difference(self, other: "VersionSet") -> "VersionSet":
        return self & other.complement()

    __and__ = intersection
    __or__ = union
    __sub__ = difference
    __invert__ = complement

    # ---- queries ---------------------------------------------------------- #
    def is_empty(self) -> bool:
        return not self.intervals

    def is_any(self) -> bool:
        return self == VersionSet.full()

    def is_subset(self, other: "VersionSet") -> bool:
        return (self & ~other).is_empty()

    def is_disjoint(self, other: "VersionSet") -> bool:
        return (self & other).is_empty()

    def __contains__(self, version) -> bool:
        """Binary search for the last interval starting at or before `version`."""
        probe = (1, _v(version), 0)
        i = bisect.bisect_right(self.intervals, probe, key=lambda iv: iv.lo_key)
        return i > 0 and self.intervals[i - 1].hi_key > probe

    def best(self, candidates):
        """Highest candidate version inside the set (what a resolver picks)."""
        return max((v for v in map(_v, candidates) if v in self), default=None)

    # ---- plumbing --------------------------------------------------------- #
    def __eq__(self, other) -> bool:
        return isinstance(other, VersionSet) and self.intervals == other.intervals

    def __hash__(self) -> int:
        return hash(self.intervals)

    def __str__(self) -> str:
        ivs = self.intervals
        if not ivs:
            return "∅"
        if len(ivs) == 1 and ivs[0].lo is None and ivs[0].hi is None:
            return "*"
        if (len(ivs) == 2 and ivs[0].lo is None and ivs[1].hi is None
                and ivs[0].hi == ivs[1].lo and not ivs[0].hi_inc and not ivs[1].lo_inc):
            return f"!={ivs[0].hi}"
        parts = []
        for iv in ivs:
            if iv.lo is not None and iv.lo == iv.hi:
                parts.append(f"=={iv.lo}")
                continue
            bits = []
            if iv.lo is not None:
                bits.append(f"{'>=' if iv.lo_inc else '>'}{iv.lo}")
            if iv.hi is not None:
                bits.append(f"{'<=' if iv.hi_inc else '<'}{iv.hi}")
            parts.append(",".join(bits))
        return " | ".join(parts)

    def __repr__(self) -> str:
        return f"VersionSet({self})"
