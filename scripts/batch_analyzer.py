"""
Batch .jsonl Chat Record Parser & Hermes Anomaly Auditor (scripts/batch_analyzer.py).

Streams SillyTavern .jsonl export files line-by-line to prevent memory bloat on large chat logs.
Splits mes payloads into pre_thought_garbage, inner_monologue (<thinking>), and public_dialogue.
Flags anomaly tags (MISSING_CLOSING_TAG, PRE_PLANNING_BLEED, OMNISCIENT_BLEED).
Exports pristine Chain-of-Thought datasets via --export-cot CLI flag for fine-tuning.
"""

import json
import re
import argparse
from typing import Dict, Any, List, Generator


class BatchChatAnalyzer:
    """Line-by-line streaming chat record parser and anomaly detector."""

    def __init__(self, hidden_variables: List[str] | None = None):
        self.hidden_variables = [h.lower() for h in (hidden_variables or [])]

    def parse_message_payload(self, text: str) -> Dict[str, Any]:
        """
        Splits raw message text into pre_thought_garbage, inner_monologue, and public_dialogue.
        Identifies structural anomalies.
        """
        anomalies: List[str] = []
        pre_thought_garbage = ""
        inner_monologue = ""
        public_dialogue = text

        has_open_tag = "<thinking>" in text
        has_close_tag = "</thinking>" in text

        if has_open_tag:
            open_tag = "<thinking>"
            close_tag = "</thinking>"
            parts = text.split(open_tag, 1)
            pre_thought_garbage = parts[0].strip()
            if pre_thought_garbage:
                anomalies.append("PRE_PLANNING_BLEED")

            after_open = parts[1]
            if close_tag in after_open:
                mono_parts = after_open.split(close_tag, 1)
                inner_monologue = mono_parts[0].strip()
                public_dialogue = mono_parts[1].strip()
            else:
                anomalies.append("MISSING_CLOSING_TAG")
                inner_monologue = after_open.strip()
                public_dialogue = ""

        # Check omniscient bleed against hidden variables
        if self.hidden_variables:
            lower_public = public_dialogue.lower()
            for hidden in self.hidden_variables:
                if hidden in lower_public:
                    anomalies.append("OMNISCIENT_BLEED")
                    break

        return {
            "pre_thought_garbage": pre_thought_garbage,
            "inner_monologue": inner_monologue,
            "public_dialogue": public_dialogue,
            "anomalies": anomalies,
            "is_clean": len(anomalies) == 0,
        }

    def analyze_file(self, jsonl_path: str) -> Generator[Dict[str, Any], None, None]:
        """Stream a .jsonl export file line by line and yield parsed record entries."""
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line_idx, line in enumerate(f, start=1):
                line_str = line.strip()
                if not line_str:
                    continue
                try:
                    data = json.loads(line_str)
                    raw_text = data.get("mes", "") or data.get("text", "") or data.get("content", "")
                    parsed = self.parse_message_payload(raw_text)
                    yield {
                        "line_number": line_idx,
                        "raw": data,
                        **parsed
                    }
                except Exception as exc:
                    yield {
                        "line_number": line_idx,
                        "raw": line_str,
                        "anomalies": [f"JSON_PARSE_ERROR: {exc}"],
                        "is_clean": False,
                    }

    def export_cot_dataset(self, input_jsonl: str, output_jsonl: str) -> int:
        """Filter out anomalous records and export clean CoT fine-tuning dataset."""
        exported_count = 0
        with open(output_jsonl, "w", encoding="utf-8") as out:
            for record in self.analyze_file(input_jsonl):
                if record.get("is_clean") and record.get("inner_monologue") and record.get("public_dialogue"):
                    cot_entry = {
                        "system": "You are a Character Subagent.",
                        "input": record["raw"].get("name", "User"),
                        "thought": record["inner_monologue"],
                        "output": record["public_dialogue"],
                    }
                    out.write(json.dumps(cot_entry) + "\n")
                    exported_count += 1
        return exported_count


def main():
    parser = argparse.ArgumentParser(description="SPM Batch Chat Analyzer & CoT Exporter")
    parser.add_argument("input_file", help="Path to input SillyTavern .jsonl export file")
    parser.add_argument("--export-cot", help="Output path for clean Chain-of-Thought dataset")
    parser.add_argument("--hidden-var", action="append", help="Hidden variable strings to audit for OMNISCIENT_BLEED")

    args = parser.parse_args()
    analyzer = BatchChatAnalyzer(hidden_variables=args.hidden_var)

    if args.export_cot:
        count = analyzer.export_cot_dataset(args.input_file, args.export_cot)
        print(f"Exported {count} clean Chain-of-Thought records to {args.export_cot}")
    else:
        clean_count = 0
        anomaly_count = 0
        for rec in analyzer.analyze_file(args.input_file):
            if rec.get("is_clean"):
                clean_count += 1
            else:
                anomaly_count += 1
                print(f"Line {rec['line_number']} Anomalies: {rec.get('anomalies')}")
        print(f"Analysis Complete. Clean: {clean_count}, Anomalies: {anomaly_count}")


if __name__ == "__main__":
    main()
