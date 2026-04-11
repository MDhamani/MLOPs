import argparse
import re
from typing import Iterable

import apache_beam as beam
from apache_beam.options.pipeline_options import PipelineOptions


WORD_RE = re.compile(r"[A-Za-z']+")


class NormalizeAndExtractWords(beam.DoFn):
    def __init__(self, lowercase: bool, min_word_len: int):
        self.lowercase = lowercase
        self.min_word_len = min_word_len

    def process(self, line: str) -> Iterable[str]:
        for word in WORD_RE.findall(line):
            if self.lowercase:
                word = word.lower()
            if len(word) >= self.min_word_len:
                yield word


def format_kv(kv):
    word, count = kv
    return f"{word},{count}"


def build_pipeline(
    p: beam.Pipeline,
    input_path: str,
    output_path: str,
    lowercase: bool,
    min_word_len: int,
    top_n: int | None,
):
    counts = (
        p
        | "ReadInput" >> beam.io.ReadFromText(input_path)
        | "ExtractWords" >> beam.ParDo(NormalizeAndExtractWords(lowercase, min_word_len))
        | "PairWithOne" >> beam.Map(lambda word: (word, 1))
        | "CountWords" >> beam.CombinePerKey(sum)
    )

    (
        counts
        | "SortForReadability" >> beam.transforms.combiners.ToList()
        | "SortList" >> beam.Map(lambda rows: sorted(rows, key=lambda x: (-x[1], x[0])))
        | "TakeTopN"
        >> beam.Map(
            lambda rows: rows[:top_n]
            if top_n is not None and top_n > 0
            else rows
        )
        | "FlattenRows" >> beam.FlatMap(lambda rows: rows)
        | "FormatCSV" >> beam.Map(format_kv)
        | "WriteOutput" >> beam.io.WriteToText(output_path, file_name_suffix=".csv", shard_name_template="")
    )


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(description="Apache Beam word count with normalization options.")
    parser.add_argument("--input", required=True, help="Input text file path.")
    parser.add_argument("--output", required=True, help="Output file prefix (without extension).")
    parser.add_argument("--lowercase", action="store_true", help="Lowercase words before counting.")
    parser.add_argument("--min-word-len", type=int, default=1, help="Minimum word length to keep.")
    parser.add_argument("--top-n", type=int, default=None, help="Optional top N most frequent words.")
    return parser.parse_known_args()


def main() -> None:
    args, pipeline_args = parse_args()
    pipeline_options = PipelineOptions(pipeline_args)

    with beam.Pipeline(options=pipeline_options) as p:
        build_pipeline(
            p=p,
            input_path=args.input,
            output_path=args.output,
            lowercase=args.lowercase,
            min_word_len=args.min_word_len,
            top_n=args.top_n,
        )


if __name__ == "__main__":
    main()
