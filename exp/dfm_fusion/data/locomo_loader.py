"""LoCoMo dataset loader: parse locomo10.json into memory snippets and QA pairs.
Each conversation is segmented into per-turn memory snippets with timestamps.
Image references are replaced with BLIP captions for text-only evaluation.
"""

import json
import re
from datetime import datetime
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


CATEGORY_NAMES = {
    1: "multi-hop",
    2: "temporal",
    3: "open-domain",
    4: "single-hop",
    5: "adversarial",
}


@dataclass
class MemorySnippet:
    text: str
    speaker: str
    dia_id: str
    session_idx: int
    timestamp: datetime
    turn_index: int


@dataclass
class QAPair:
    question: str
    answer: str
    category: int
    category_name: str
    evidence: list[str]
    adversarial_answer: Optional[str] = None


@dataclass
class Conversation:
    sample_id: str
    speaker_a: str
    speaker_b: str
    snippets: list[MemorySnippet] = field(default_factory=list)
    qa_pairs: list[QAPair] = field(default_factory=list)


def _parse_datetime(dt_str: str) -> datetime:
    dt_str = dt_str.strip()
    dt_str = re.sub(r"(\d+)(st|nd|rd|th)", r"\1", dt_str)
    for fmt in [
        "%I:%M %p on %d %B, %Y",
        "%I:%M %p on %d %B %Y",
        "%I:%M%p on %d %B, %Y",
        "%I:%M%p on %d %B %Y",
    ]:
        try:
            return datetime.strptime(dt_str, fmt)
        except ValueError:
            continue
    raise ValueError(f"Cannot parse datetime: {dt_str!r}")


def _format_turn_text(turn: dict) -> str:
    speaker = turn["speaker"]
    text = turn["text"]
    if "blip_caption" in turn and turn["blip_caption"]:
        text = f"[shares {turn['blip_caption']}] {text}"
    return f"{speaker}: {text}"


def load_locomo(data_path: str | Path) -> list[Conversation]:
    data_path = Path(data_path)
    raw = json.loads(data_path.read_text())

    conversations = []
    for conv_raw in raw:
        conv_data = conv_raw["conversation"]
        conv = Conversation(
            sample_id=conv_raw["sample_id"],
            speaker_a=conv_data["speaker_a"],
            speaker_b=conv_data["speaker_b"],
        )

        session_keys = sorted(
            [k for k in conv_data if re.match(r"session_\d+$", k)],
            key=lambda k: int(k.split("_")[1]),
        )

        global_turn_idx = 0
        for skey in session_keys:
            sess_idx = int(skey.split("_")[1])
            dt_key = f"{skey}_date_time"
            sess_ts = _parse_datetime(conv_data[dt_key])

            for turn in conv_data[skey]:
                snippet = MemorySnippet(
                    text=_format_turn_text(turn),
                    speaker=turn["speaker"],
                    dia_id=turn["dia_id"],
                    session_idx=sess_idx,
                    timestamp=sess_ts,
                    turn_index=global_turn_idx,
                )
                conv.snippets.append(snippet)
                global_turn_idx += 1

        for qa_raw in conv_raw["qa"]:
            if qa_raw["category"] == 5:
                answer = qa_raw.get("adversarial_answer", "")
            else:
                answer = str(qa_raw["answer"])
            qa = QAPair(
                question=qa_raw["question"],
                answer=answer,
                category=qa_raw["category"],
                category_name=CATEGORY_NAMES.get(qa_raw["category"], "unknown"),
                evidence=qa_raw.get("evidence", []),
                adversarial_answer=qa_raw.get("adversarial_answer"),
            )
            conv.qa_pairs.append(qa)

        conversations.append(conv)

    return conversations


if __name__ == "__main__":
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else "external/locomo/data/locomo10.json"
    convs = load_locomo(path)
    for c in convs:
        cats = {}
        for q in c.qa_pairs:
            cats[q.category_name] = cats.get(q.category_name, 0) + 1
        print(
            f"{c.sample_id}: {len(c.snippets)} snippets, "
            f"{len(c.qa_pairs)} QAs, categories={cats}"
        )
