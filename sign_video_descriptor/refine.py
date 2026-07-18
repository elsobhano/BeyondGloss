"""Sign Video Descriptor — Stage 2: merge and refine with an LLM.

The per-segment descriptions produced by ``describe_segments.py`` are
concatenated in temporal order and passed to an LLM (GPT-4o-mini) that removes
irrelevant content (people, background, colours, facial expression, generic
"sign language" phrasing) and returns a single coherent sentence describing the
hand motion across the whole video. The result is stored under the ``refined``
key of each JSON file.

Set the OpenAI API key via the OPENAI_API_KEY environment variable:

    export OPENAI_API_KEY="sk-..."
"""

import argparse
import json
import os
import time

from openai import OpenAI

client = OpenAI()  # reads OPENAI_API_KEY from the environment

REFINE_PROMPT = """The following paragraph is the description of consecutive parts of a video.
Relevant information are hands movements, handshapes, hands states, body movements, body parts and gestures.
Anything else is irrelevant like people, backgrounds, colours and facial expression.
Please remove any irrelevant information from the paragraph and give us the refined output in just one sentence.
Do not put such phrases like "suggesting a different sign language gesture","sign language" or anything else without a direct relation with state of hands.
Do not add something based on your conclusion, just stick to the paragraph.
Do not change the temporal order of our favourite information.
Do not start your sentence with any information about person in the video.
Avoid paraphrasing as much as it is possible.
Do not add anything to your response like: "OUTPUT: ".

Paragraph: "{paragraph}"
"""


def refining(paragraph, model="gpt-4o-mini"):
    completion = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": REFINE_PROMPT.format(paragraph=paragraph)},
        ],
        max_tokens=1024,
    )
    return completion.choices[0].message.content


def generate_refined(paragraph, filename, retries=3):
    print(f"Processing: {filename}")
    for attempt in range(retries):
        try:
            return refining(paragraph)
        except Exception as exc:  # noqa: BLE001
            if attempt < retries - 1:
                print(f"Request failed ({exc}). Retrying in 10 seconds...")
                time.sleep(10)
            else:
                print(f"Failed after {retries} attempts.")
                return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, required=True,
                        help="Directory of per-video JSONs from describe_segments.py.")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Directory to write JSONs with an added 'refined' field.")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    done = {f for f in os.listdir(args.output_dir) if f.endswith(".json")}

    for dirpath, _, filenames in os.walk(args.input_dir):
        for filename in sorted(filenames):
            if not filename.endswith(".json"):
                continue
            if filename in done:
                print(f"{filename} already refined, skipping.")
                continue

            with open(os.path.join(dirpath, filename), "r", encoding="utf-8") as fh:
                data = json.load(fh)

            paragraph = "".join(v for k, v in data.items() if k.startswith("chunk"))
            data["refined"] = generate_refined(paragraph, filename)

            with open(os.path.join(args.output_dir, filename), "w", encoding="utf-8") as fh:
                json.dump(data, fh, indent=4, ensure_ascii=False)


if __name__ == "__main__":
    main()
