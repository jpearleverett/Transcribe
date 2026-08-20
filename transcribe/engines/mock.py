"""A fake engine used by the test suite.

Registered only when TRANSCRIBE_TEST is set, so it can never show up in a real
install's engine picker.
"""

from __future__ import annotations

import time

from ..align import Word, Turn
from .base import Context, Engine, EngineError, Result, register


@register
class Mock(Engine):
    name = "mock"
    label = "Mock engine (testing)"
    description = "Returns a canned transcript. Used by the test suite."
    needs_key = False

    def transcribe(self, ctx: Context) -> Result:
        if ctx.options.get("fail"):
            raise EngineError("deliberate test failure")

        steps = int(ctx.options.get("steps") or 3)
        for i in range(steps):
            ctx.check_cancel()
            ctx.progress("transcribing", (i + 1) / (steps + 1))
            time.sleep(float(ctx.options.get("delay") or 0.01))

        words, t = [], 0.0
        script = [
            ("A", "Hello everyone, thanks for joining."),
            ("B", "Happy to be here."),
            ("A", "Let's start with the numbers."),
            ("B", "Revenue is up eleven percent."),
        ]
        turns = []
        for spk, sentence in script:
            turn_start = t
            for token in sentence.split():
                words.append(Word(start=t, end=t + 0.3, text=token, confidence=0.9))
                t += 0.35
            turns.append(Turn(start=turn_start, end=t, speaker=spk))
            t += 0.6

        return Result(words=words, turns=turns, language="en", model="mock-1",
                      text=" ".join(w.text for w in words))
