"""Import every engine so it registers itself.

Order here is the order they appear in the UI's picker, so the best default
comes first.
"""

from . import runpod        # noqa: F401  GPU pod: best accuracy, self-hosted
from . import assemblyai    # noqa: F401  best published diarization accuracy
from . import deepgram      # noqa: F401  cheapest, generous free credit
from . import elevenlabs    # noqa: F401  simplest API, word-level speakers
from . import openai        # noqa: F401  segment-level speakers only
from . import local         # noqa: F401  fully offline, on the phone itself

import os

if os.environ.get("TRANSCRIBE_TEST"):
    from . import mock      # noqa: F401
