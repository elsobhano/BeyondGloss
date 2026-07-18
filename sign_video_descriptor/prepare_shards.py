"""Split the list of videos still needing descriptions into work shards.

Compares the videos available (one frame-subdirectory each under --frames_dir)
against those already described (JSONs under --descriptions_dir) and writes the
remaining ones as chunk_*.json shards, so Stage 1 can be run in parallel jobs
(e.g. one SLURM task per shard).
"""

import argparse
import json
import os


def chunk_list(data, chunk_size):
    for i in range(0, len(data), chunk_size):
        yield data[i:i + chunk_size]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames_dir", type=str, required=True,
                        help="Root directory with one frame-subdirectory per video.")
    parser.add_argument("--descriptions_dir", type=str, required=True,
                        help="Directory of already-produced <video>.json descriptions.")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Where to write the chunk_*.json shard files.")
    parser.add_argument("--chunk_size", type=int, default=5000)
    args = parser.parse_args()

    directories = [d for d in os.listdir(args.frames_dir)
                   if os.path.isdir(os.path.join(args.frames_dir, d))]
    done = {f[:-5] for f in os.listdir(args.descriptions_dir) if f.endswith(".json")}
    remaining = [d for d in directories if d not in done]

    print(f"{len(directories)} videos total, {len(done)} done, {len(remaining)} remaining.")
    os.makedirs(args.output_dir, exist_ok=True)

    for idx, chunk in enumerate(chunk_list(remaining, args.chunk_size)):
        path = os.path.join(args.output_dir, f"chunk_{idx + 1}.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(chunk, f, indent=4)
        print(f"Saved {path} ({len(chunk)} videos)")


if __name__ == "__main__":
    main()
