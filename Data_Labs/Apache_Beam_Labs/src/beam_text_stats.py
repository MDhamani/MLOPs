import argparse
import json
import re
from typing import Iterable

import apache_beam as beam
from apache_beam.options.pipeline_options import PipelineOptions


TOKEN_RE = re.compile(r"[A-Za-z']+")


class LineMetrics(beam.DoFn):
    def process(self, line: str):
        token_count = len(TOKEN_RE.findall(line))
        yield {
            "line_count": 1,
            "char_count": len(line),
            "token_count": token_count,
            "non_empty_line_count": int(bool(line.strip())),
        }


def combine_metrics(rows: Iterable[dict]) -> dict:
    total = {
        "line_count": 0,
        "char_count": 0,
        "token_count": 0,
        "non_empty_line_count": 0,
    }

    for row in rows:
        total["line_count"] += row["line_count"]
        total["char_count"] += row["char_count"]
        total["token_count"] += row["token_count"]
        total["non_empty_line_count"] += row["non_empty_line_count"]

    return total


def finalize_metrics(metrics: dict) -> dict:
    line_count = metrics["line_count"]
    non_empty_line_count = metrics["non_empty_line_count"]

    metrics["avg_chars_per_line"] = round(metrics["char_count"] / line_count, 3) if line_count else 0.0
    metrics["avg_tokens_per_non_empty_line"] = (
        round(metrics["token_count"] / non_empty_line_count, 3) if non_empty_line_count else 0.0
    )
    return metrics


def to_json_line(data: dict) -> str:
    return json.dumps(data, sort_keys=True)


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(description="Compute text-level quality/statistics with Apache Beam.")
    parser.add_argument("--input", required=True, help="Input text file path.")
    parser.add_argument("--output", required=True, help="Output file prefix (without extension).")
    return parser.parse_known_args()


def main() -> None:
    args, pipeline_args = parse_args()
    pipeline_options = PipelineOptions(pipeline_args)

    with beam.Pipeline(options=pipeline_options) as p:
        (
            p
            | "ReadInput" >> beam.io.ReadFromText(args.input)
            | "ComputeLineMetrics" >> beam.ParDo(LineMetrics())
            | "CombineMetrics" >> beam.CombineGlobally(combine_metrics)
            | "FinalizeMetrics" >> beam.Map(finalize_metrics)
            | "ToJSON" >> beam.Map(to_json_line)
            | "WriteOutput" >> beam.io.WriteToText(args.output, file_name_suffix=".json", shard_name_template="")
        )


if __name__ == "__main__":
    main()
