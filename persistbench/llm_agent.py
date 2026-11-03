"""LLM agent adapter + real memory-write generation (Week 2 pilot prep).

This module is the bridge from the scripted harness (Weeks 1-2) to live
models. It changes *who plays Session 1* and *who writes the memory* —
nothing else. The trajectory it produces has exactly the scripted format
(``runner.Trajectory``: steps of {tool, kwargs, response, fault_injected},
a claim dict, the world), so the fault wrappers, the memory policies
M0-M5, the scorer and ``runner.run_chain`` machinery work unchanged.

Backends sit behind one small interface (:class:`ChatBackend`):

- :class:`FakeLLMBackend` — deterministic, scripted; used by the tests so
  the full plumbing is exercised with no model download.
- :class:`HFTransformersBackend` — Hugging Face transformers generation
  (greedy decoding; model id, 4-bit flag, max tokens, seed are
  parameters). The same code runs a tiny cached model locally for smoke
  tests and Qwen2.5-7B-Instruct / Qwen3-8B on a Kaggle T4. ``torch`` /
  ``transformers`` are imported lazily, only when this backend is built.

Tool use is a text protocol (chat models served by transformers have no
native function calling here): the model must answer each turn with one
JSON object, ``{"tool": <name>, "args": {...}}`` or ``{"done": true}``.
Parsing is conservative — anything that is not a single valid JSON action
for a known tool with correctly-typed args is rejected, fed back to the
model as an error observation, and counted; it never reaches the world.

Memory writes are real model text, generated per policy style:

- ``plain`` (M1, and the input claim for M4): a free-form end-of-session
  summary ending in a FACTS block.
- ``disclosure`` (M2): the same summary under the frozen disclosure
  instruction (``M2Disclosure.DISCLOSURE_INSTRUCTION``).
- ``claims`` (M3): one CLAIM line per fact with the model's own evidence
  note; the lines are parsed into the claim dict the existing
  evidence-contract policy consumes.

Claim extraction from summary text is conservative by design (the
formation measurement depends on it): a fact enters the claim dict only
when the model explicitly asserts it on a recognised line with a
parseable value. Missing, malformed, or unknown-key lines are recorded
in the parse log as ``unknown`` — never silently dropped, never treated
as verified.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

from .chains import build_world
from .faults import FaultInjector
from .runner import Trajectory

# ---------------------------------------------------------------------------
# Tool protocol
# ---------------------------------------------------------------------------

TOOL_ARGS = {
    "get_order": {"order_id": str},
    "get_product": {"sku": str},
    "get_shipment": {"tracking": str},
    "charge_payment": {"order_id": str},
    "ship_order": {"order_id": str},
    "update_shipping_address": {"order_id": str, "address": str},
    "cancel_order": {"order_id": str},
    "refund_order": {"order_id": str},
    "restock": {"sku": str, "qty": int},
    "create_return_label": {"tracking": str},
}

SYSTEM_PROMPT = (
    "You are an operations agent working in a retail order system. You "
    "complete the task by calling tools, one at a time.\n"
    "Available tools (name: arguments):\n"
    + "\n".join(
        f"- {name}({', '.join(f'{arg}: {typ.__name__}' for arg, typ in args.items())})"
        for name, args in TOOL_ARGS.items()
    )
    + "\nReply with ONLY one JSON object per turn, no other text:\n"
    '{"tool": "<tool name>", "args": {...}} to call a tool, or '
    '{"done": true} when the task is finished.\n'
    "Base every decision on the tool results you actually receive."
)


# ---------------------------------------------------------------------------
# Backends
# ---------------------------------------------------------------------------

@dataclass
class Generation:
    text: str
    prompt_tokens: int
    completion_tokens: int


class ChatBackend:
    """One interface for every chat-model backend."""

    name = "base"
    model_label = "base"

    def generate(self, messages: list, max_new_tokens: int | None = None,
                 seed: int | None = None) -> Generation:
        raise NotImplementedError


def _word_count(text: str) -> int:
    return len(text.split())


_THINK_BLOCK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)
_THINK_OPEN_RE = re.compile(r"<think>.*$", re.DOTALL)


def strip_thinking(text: str) -> str:
    """Remove reasoning-model ``<think>`` blocks from model output.

    Closed blocks are dropped entirely (the answer follows them). An
    *unclosed* block means generation was cut off mid-reasoning and no
    answer was ever emitted, so everything from the opening tag on is
    dropped too — the caller then sees an empty/answerless reply and
    scores it as a parse failure, never as content. This is defence in
    depth: the primary fix for thinking models is disabling thinking at
    the chat template (see ``HFTransformersBackend``), because reasoning
    text also burns the generation budget before any answer exists.
    """
    if not text or "<think>" not in text:
        return text or ""
    text = _THINK_BLOCK_RE.sub("", text)
    text = _THINK_OPEN_RE.sub("", text)
    return text.strip()


class FakeLLMBackend(ChatBackend):
    """Deterministic scripted backend for tests.

    ``responses`` is either a list of reply strings consumed in order, or
    a callable taking the message list and returning a reply string.
    Token counts are word counts — a labelled stand-in, like the Week 2
    write-log placeholder; no real model is involved.
    """

    name = "fake"

    def __init__(self, responses, model_label: str = "fake-llm"):
        self._responses = responses
        self._queue = list(responses) if isinstance(responses, (list, tuple)) else None
        self.model_label = model_label
        self.calls = 0

    def generate(self, messages, max_new_tokens=None, seed=None) -> Generation:
        self.calls += 1
        if self._queue is not None:
            if not self._queue:
                raise RuntimeError("FakeLLMBackend ran out of scripted responses")
            text = self._queue.pop(0)
        else:
            text = self._responses(messages)
        prompt_tokens = sum(_word_count(m["content"]) for m in messages)
        return Generation(text=text, prompt_tokens=prompt_tokens,
                          completion_tokens=_word_count(text))


class HFTransformersBackend(ChatBackend):
    """Hugging Face transformers backend (greedy decoding).

    The same code path serves a sub-1B smoke model on CPU and the pilot
    models (Qwen2.5-7B-Instruct, Qwen3-8B) at 4-bit on a Kaggle T4.
    Heavy imports happen here, never at module import time.
    """

    name = "hf"

    def __init__(self, model_id: str, load_in_4bit: bool = False,
                 max_new_tokens: int = 256, seed: int = 0,
                 device_map: str | None = None,
                 enable_thinking: bool | None = None):
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise ImportError(
                "HFTransformersBackend needs torch + transformers installed "
                "(see kaggle/RUNBOOK.md); the package core is standard-library "
                "only and the fake backend needs neither."
            ) from exc
        self.model_id = model_id
        self.model_label = model_id
        self.max_new_tokens = max_new_tokens
        self.seed = seed
        # Thinking models (Qwen3 family) reason inside <think> blocks by
        # default. In the Week-2 pilot that consumed the whole 256-token
        # budget before any FACTS block was emitted (FACTS block in
        # 0/30 writes; empty stores in 89/90 M1-M3 cells), i.e. a
        # measurement failure, not a model result. Default thinking OFF for those
        # models (Qwen's own documented non-thinking mode); every other
        # model is left exactly as its template defaults.
        if enable_thinking is None and "qwen3" in model_id.lower():
            enable_thinking = False
        self.enable_thinking = enable_thinking
        self._torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        # device_map stays unset unless requested (transformers 5.x
        # requires accelerate for it; the Kaggle notebook installs it).
        model_kwargs = {}
        if device_map is not None:
            model_kwargs["device_map"] = device_map
        if load_in_4bit:
            from transformers import BitsAndBytesConfig

            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16
            )
        else:
            model_kwargs["torch_dtype"] = "auto"
        self.model = AutoModelForCausalLM.from_pretrained(model_id, **model_kwargs)

    def generate(self, messages, max_new_tokens=None, seed=None) -> Generation:
        torch = self._torch
        if seed is not None:
            torch.manual_seed(seed)
        budget = max_new_tokens or self.max_new_tokens
        tokenizer = self.tokenizer
        if getattr(tokenizer, "chat_template", None):
            if self.enable_thinking is not None:
                try:
                    prompt = tokenizer.apply_chat_template(
                        messages, tokenize=False, add_generation_prompt=True,
                        chat_template_kwargs={
                            "enable_thinking": self.enable_thinking
                        },
                    )
                except TypeError:  # template without a thinking switch
                    prompt = tokenizer.apply_chat_template(
                        messages, tokenize=False, add_generation_prompt=True
                    )
            else:
                prompt = tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
        else:  # pragma: no cover - all pilot/smoke models ship a template
            prompt = "\n".join(f"{m['role']}: {m['content']}" for m in messages)
            prompt += "\nassistant:"
        inputs = tokenizer(prompt, return_tensors="pt").to(self.model.device)
        prompt_tokens = int(inputs["input_ids"].shape[1])
        with torch.no_grad():
            output = self.model.generate(
                **inputs,
                max_new_tokens=budget,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        new_ids = output[0][prompt_tokens:]
        text = tokenizer.decode(new_ids, skip_special_tokens=True)
        # Defence in depth: even with thinking disabled at the template,
        # never let a stray reasoning block reach the parsers.
        text = strip_thinking(text)
        return Generation(text=text, prompt_tokens=prompt_tokens,
                          completion_tokens=int(new_ids.shape[0]))


# ---------------------------------------------------------------------------
# Conservative parsing: actions and summary claims
# ---------------------------------------------------------------------------

def parse_action(text: str):
    """Parse one tool-action reply. Returns (action, error).

    Exactly one of the two is None... on success error is None and action
    is {"tool":..., "args": {...}} or {"done": True}. Anything else —
    prose, malformed JSON, unknown tool, wrong arg names/types — is an
    error string; the caller feeds it back as an observation and the
    world is never touched.
    """
    if not text or not text.strip():
        return None, "empty reply"
    candidate = strip_thinking(text).strip()
    if not candidate:
        return None, "empty reply after stripping reasoning"
    if candidate.startswith("```"):
        candidate = candidate.strip("`")
        if candidate.lower().startswith("json"):
            candidate = candidate[4:].strip()
    decoder = json.JSONDecoder()
    action = None
    for match in re.finditer(r"\{", candidate):
        try:
            obj, _ = decoder.raw_decode(candidate[match.start():])
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            action = obj
            break
    if action is None:
        return None, "no JSON object found in reply"
    if action.get("done") is True:
        return {"done": True}, None
    tool = action.get("tool")
    if tool not in TOOL_ARGS:
        return None, f"unknown or missing tool {tool!r}"
    args = action.get("args")
    if not isinstance(args, dict):
        return None, "args must be a JSON object"
    expected = TOOL_ARGS[tool]
    if set(args) != set(expected):
        return None, f"{tool} takes args {sorted(expected)}, got {sorted(args)}"
    for arg_name, arg_type in expected.items():
        value = args[arg_name]
        if arg_type is int:
            if isinstance(value, bool) or not isinstance(value, int):
                return None, f"{tool}.{arg_name} must be an integer"
        elif not isinstance(value, arg_type):
            return None, f"{tool}.{arg_name} must be a {arg_type.__name__}"
    return {"tool": tool, "args": args}, None


def _parse_value(raw: str, exemplar):
    """Parse a claimed value against the intended fact's type."""
    raw = raw.strip().strip('"').strip("'")
    if isinstance(exemplar, bool):
        lowered = raw.lower()
        if lowered in ("true", "yes"):
            return True
        if lowered in ("false", "no"):
            return False
        raise ValueError(f"not a boolean: {raw!r}")
    if isinstance(exemplar, int):
        return int(raw)
    return raw


@dataclass
class ClaimExtraction:
    claim: dict = field(default_factory=dict)
    claims: list = field(default_factory=list)   # per-fact parse records
    parse_log: list = field(default_factory=list)  # every failure, kept


# A recognised assertion line: optional CLAIM prefix, a fact key, ':' or
# '=', then the value (an optional '| evidence: ...' tail is captured).
_ASSERTION_RE = re.compile(
    r"^\s*(?:CLAIM\s+)?([A-Za-z_][A-Za-z0-9_]*)\s*[:=]\s*(.+?)\s*$"
)


def extract_claims(summary_text: str, chain: dict) -> ClaimExtraction:
    """Extract the model's asserted facts from its summary, conservatively.

    A fact enters the claim dict only from an explicit assertion line
    naming a known fact key (or ``task_success``) with a value that
    parses against the intended type. Everything else — missing keys,
    unparseable values, unknown keys — lands in the parse log with
    status ``unknown`` and never in the claim.
    """
    known = dict(chain["intended_facts"])
    known["task_success"] = True  # exemplar for typing
    result = ClaimExtraction()
    seen = set()
    # Reasoning text is never evidence: a value the model merely
    # considered inside <think> must not be extracted as an assertion.
    summary_text = strip_thinking(summary_text or "")
    for line in summary_text.splitlines():
        match = _ASSERTION_RE.match(line)
        if not match:
            continue
        key, raw_value = match.group(1), match.group(2)
        if key not in known:
            if key.islower() and "_" in key or key in ("evidence", "note"):
                result.parse_log.append({
                    "line": line.strip(), "fact": key,
                    "status": "unknown", "reason": "unknown fact key",
                })
            continue
        evidence = None
        if "|" in raw_value:
            raw_value, tail = raw_value.split("|", 1)
            raw_value = raw_value.strip()
            evidence = tail.strip()
            if evidence.lower().startswith("evidence:"):
                evidence = evidence[len("evidence:"):].strip()
        try:
            value = _parse_value(raw_value, known[key])
        except (ValueError, TypeError):
            result.parse_log.append({
                "line": line.strip(), "fact": key, "status": "unknown",
                "reason": f"unparseable value {raw_value!r}",
            })
            result.claims.append({
                "fact": key, "claimed": None, "status": "unknown",
                "evidence_text": evidence, "raw": raw_value,
            })
            seen.add(key)
            continue
        result.claim[key] = value
        result.claims.append({
            "fact": key, "claimed": value, "status": "parsed",
            "evidence_text": evidence,
        })
        seen.add(key)
    for key in known:
        if key not in seen:
            result.parse_log.append({
                "fact": key, "status": "unknown",
                "reason": "not asserted in summary",
            })
    return result


# ---------------------------------------------------------------------------
# The LLM origin agent
# ---------------------------------------------------------------------------

@dataclass
class WriteGeneration:
    style: str
    text: str
    extraction: ClaimExtraction
    prompt_tokens: int
    completion_tokens: int


class LLMAgent:
    """Plays a chain's Session-1 origin task with a chat model.

    The fault is injected on the first call to the chain's critical tool
    (``plan[critical_step]``), exactly as the scripted runner injects at
    the plan's critical step; every executed call is recorded in the
    scripted step format, fault flag included.
    """

    def __init__(self, backend: ChatBackend, max_steps: int = 8):
        self.backend = backend
        self.max_steps = max_steps
        self.name = f"llm:{backend.model_label}"

    # -- Session 1 ------------------------------------------------------
    def run_origin(self, chain: dict, fault_condition: str,
                   seed: int = 0) -> Trajectory:
        world = build_world(chain)
        injector = FaultInjector(fault_condition, seed=seed)
        critical_tool = chain["plan"][chain["critical_step"]][0]
        context = []
        if chain.get("order"):
            context.append(f"Order id: {chain['order']}.")
        if chain.get("product"):
            context.append(f"Product sku: {chain['product']}.")
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": chain["origin_instruction"] + " " + " ".join(context)},
        ]
        steps = []
        transcript = []
        usage = {"prompt_tokens": 0, "completion_tokens": 0}
        invalid_actions = 0
        fault_applied = False
        for _turn in range(self.max_steps + 4):  # slack for invalid replies
            if len(steps) >= self.max_steps:
                break
            generation = self.backend.generate(messages, seed=seed)
            usage["prompt_tokens"] += generation.prompt_tokens
            usage["completion_tokens"] += generation.completion_tokens
            transcript.append({"role": "assistant", "content": generation.text})
            action, error = parse_action(generation.text)
            if error is not None:
                invalid_actions += 1
                feedback = {"error": f"invalid action: {error}. Reply with one JSON object only."}
                messages.append({"role": "assistant", "content": generation.text})
                messages.append({"role": "user", "content": f"Observation: {json.dumps(feedback)}"})
                transcript.append({"role": "observation", "content": feedback})
                continue
            if action.get("done"):
                break
            tool, kwargs = action["tool"], action["args"]
            fn = getattr(world, tool)
            if tool == critical_tool and not fault_applied and fault_condition != "no_fault":
                response = injector.apply(fn, kwargs, chain["fault"])
                injected = True
                fault_applied = True
            else:
                response = fn(**kwargs)
                injected = False
            steps.append({
                "tool": tool,
                "kwargs": dict(kwargs),
                "response": response,
                "fault_injected": injected,
            })
            messages.append({"role": "assistant", "content": generation.text})
            messages.append({
                "role": "user",
                "content": f"Observation from {tool}: {json.dumps(response)}",
            })
            transcript.append({"role": "observation", "content": response})
        trajectory = Trajectory(
            chain_id=chain["chain_id"],
            agent=self.name,
            fault_condition=fault_condition,
            seed=seed,
            steps=steps,
            claim={},
            world=world,
        )
        # The default (M1-style) memory write, generated now so the
        # trajectory carries a claim exactly like a scripted one. The
        # pilot re-generates per policy style where a policy needs its
        # own prompt (M2/M3/M4); M1 reuses this one.
        write = self.generate_write(trajectory, chain, style="plain", seed=seed)
        trajectory.claim = write.extraction.claim
        trajectory.summary_text = write.text
        trajectory.parse_log = write.extraction.parse_log
        trajectory.claims = write.extraction.claims
        trajectory.transcript = transcript
        trajectory.invalid_actions = invalid_actions
        trajectory.usage = {
            "origin_prompt_tokens": usage["prompt_tokens"],
            "origin_completion_tokens": usage["completion_tokens"],
            "write_prompt_tokens": write.prompt_tokens,
            "write_completion_tokens": write.completion_tokens,
        }
        return trajectory

    # -- Memory writes ----------------------------------------------------
    def generate_write(self, trajectory: Trajectory, chain: dict,
                       style: str = "plain", seed: int = 0) -> WriteGeneration:
        fact_keys = list(chain["intended_facts"]) + ["task_success"]
        history = "\n".join(
            f"step {i}: {s['tool']}({json.dumps(s['kwargs'])}) -> {json.dumps(s['response'])}"
            for i, s in enumerate(trajectory.steps)
        )
        base = (
            "The session has ended. Write the end-of-session memory summary "
            "that will be stored and read by future sessions working on "
            "related tasks. Summarise in prose what you did and what the "
            "outcome was, based only on the tool results below.\n"
            f"Task: {chain['origin_instruction']}\nTool history:\n{history}\n"
        )
        if style == "plain":
            instruction = (
                "Then end with a FACTS block, one line per fact, exactly in "
                "the form 'key: value', for these keys: " + ", ".join(fact_keys) + "."
            )
        elif style == "disclosure":
            from .memory import M2Disclosure

            instruction = (
                M2Disclosure.DISCLOSURE_INSTRUCTION + "\n"
                "Then end with a FACTS block, one line per fact, exactly in "
                "the form 'key: value', for these keys: " + ", ".join(fact_keys) + "."
            )
        elif style == "claims":
            instruction = (
                "Then list your claims, one per line, exactly in the form "
                "'CLAIM key = value | evidence: <the tool result that "
                "supports it>', for these keys: " + ", ".join(fact_keys) + ". "
                "If no tool result supports a value, write the value as "
                "unknown."
            )
        else:
            raise ValueError(f"unknown write style {style!r}")
        messages = [
            {"role": "system", "content": "You are the same operations agent, "
                                          "writing your end-of-session memory."},
            {"role": "user", "content": base + instruction},
        ]
        # Writes get a larger budget than tool actions: a prose summary
        # plus a 5-line FACTS block does not fit reliably in the 256-token
        # action budget (Qwen2.5-7B hit that cap in 79/90 M1-M3 writes in
        # the Week-2 pilot). Actions stay at the backend default.
        generation = self.backend.generate(messages, max_new_tokens=512,
                                           seed=seed)
        extraction = extract_claims(generation.text, chain)
        return WriteGeneration(
            style=style,
            text=generation.text,
            extraction=extraction,
            prompt_tokens=generation.prompt_tokens,
            completion_tokens=generation.completion_tokens,
        )
