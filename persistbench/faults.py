"""Seeded deterministic fault injection around the world's tools.

The fault is the independent variable. For a given chain and condition the
transformation is fixed (each chain's fault spec names the field and values),
so every model and policy faces identical faults; the run seed is recorded
on the trajectory for provenance.
"""
from __future__ import annotations

import random

NO_FAULT = "no_fault"
EXPLICIT_ERROR = "explicit_error"
EMPTY_PARTIAL = "empty_partial"
SILENT_WRONG_VALUE = "silent_wrong_value"
STALE_VALUE = "stale_value"

FAULT_CONDITIONS = (
    NO_FAULT,
    EXPLICIT_ERROR,
    EMPTY_PARTIAL,
    SILENT_WRONG_VALUE,
    STALE_VALUE,
)


class FaultInjector:
    def __init__(self, condition: str, seed: int = 0):
        if condition not in FAULT_CONDITIONS:
            raise ValueError(f"unknown fault condition {condition!r}")
        self.condition = condition
        self.seed = seed
        self.rng = random.Random(seed)  # reserved for future jitter; transforms are fixed by spec

    def apply(self, fn, kwargs: dict, fault_spec: dict) -> dict:
        """Execute (or suppress/corrupt) one critical tool call."""
        if self.condition == NO_FAULT:
            return fn(**kwargs)
        if self.condition == EXPLICIT_ERROR:
            # Visible failure; the DB effect is NOT applied.
            return {"error": fault_spec["error_message"], "status": "error"}
        if self.condition == EMPTY_PARTIAL:
            # Looks like a response, carries nothing; DB effect NOT applied.
            return {"status": "ok"}
        if self.condition == SILENT_WRONG_VALUE:
            # DB effect applied correctly; the response misreports one field.
            response = fn(**kwargs)
            response[fault_spec["wrong_field"]] = fault_spec["wrong_value"]
            return response
        if self.condition == STALE_VALUE:
            # DB effect applied correctly; the response reports the pre-call value.
            response = fn(**kwargs)
            response[fault_spec["stale_field"]] = fault_spec["stale_value"]
            return response
        raise AssertionError("unreachable")
