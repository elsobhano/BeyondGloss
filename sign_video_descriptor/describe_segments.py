"""Sign Video Descriptor — Stage 1: segment-level description with a VideoLLM.

For each sign-language video (given as a directory of extracted frames), the
frames are split into non-overlapping 16-frame segments. Each segment is laid
out as a single grid image and passed to ShareGPT4Video-8B together with a
hand-centric prompt, producing a textual description of the hand movements,
shapes and trajectories in that segment.

The output is one JSON file per video containing the per-segment descriptions
(``chunk_0``, ``chunk_1``, ...), which are later merged and refined by
``refine.py`` (Stage 2).

Requires the ShareGPT4Video / LLaVA package to be installed and importable
(``llava.*``). See: https://github.com/ShareGPT4Omni/ShareGPT4Video
"""

import argparse
import json
import os

import numpy as np
import torch
from PIL import Image

from llava.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX
from llava.conversation import conv_templates
from llava.mm_utils import (get_model_name_from_path, process_images,
                            tokenizer_image_token)
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from tqdm import tqdm

PRE_QUERY_PROMPT = (
    "This is a sign language video, and it's important to capture "
    "details such as hand gestures and hand trajectories accurately. Please "
    "identify and describe the specific motion in this video, including hand "
    "position, movement, trajectory and any relevant facial expressions or "
    "emotions shown."
)

DEFAULT_QUERY = (
    "Describe only the motion and gestures of the person in the image focus "
    "on hands and face."
)


def create_frame_grid(img_array, interval_width=50):
    """Tile a stack of frames into a single square-ish grid image."""
    n, h, w, c = img_array.shape
    grid_size = int(np.ceil(np.sqrt(n)))

    horizontal_band = np.ones((h, interval_width, c),
                              dtype=img_array.dtype) * 255
    vertical_band = np.ones((interval_width, w + (grid_size - 1)
                            * (w + interval_width), c), dtype=img_array.dtype) * 255

    rows = []
    for i in range(grid_size):
        row_frames = []
        for j in range(grid_size):
            idx = i * grid_size + j
            if idx < n:
                frame = img_array[idx]
            else:
                frame = np.ones_like(img_array[0]) * 255
            if j > 0:
                row_frames.append(horizontal_band)
            row_frames.append(frame)
        combined_row = np.concatenate(row_frames, axis=1)
        if i > 0:
            rows.append(vertical_band)
        rows.append(combined_row)

    return np.concatenate(rows, axis=0)


def resize_image_grid(image, max_length=1920):
    """Downscale the grid image so its longest side is at most ``max_length``."""
    width, height = image.size
    if max(width, height) > max_length:
        if width > height:
            scale = max_length / width
        else:
            scale = max_length / height
        image = image.resize((int(width * scale), int(height * scale)),
                             Image.BILINEAR)
    return image


def video_answer(prompt, model, processor, tokenizer, img_grid, do_sample=True,
                 max_new_tokens=200, num_beams=1, top_p=0.9,
                 temperature=1.0, **kwargs):
    if not isinstance(img_grid, (list, tuple)):
        img_grid = [img_grid]
    image_size = img_grid[0].size
    image_tensor = process_images(img_grid, processor, model.config)[0]
    input_ids = tokenizer_image_token(
        prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt')
    input_ids = input_ids.unsqueeze(0).to(device=model.device, non_blocking=True)
    pad_token_id = (tokenizer.pad_token_id if tokenizer.pad_token is not None
                    else tokenizer.eos_token_id)

    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            images=image_tensor.to(
                dtype=torch.float16, device=model.device, non_blocking=True),
            image_sizes=[image_size],
            do_sample=do_sample,
            temperature=temperature,
            top_p=top_p,
            num_beams=num_beams,
            max_new_tokens=max_new_tokens,
            pad_token_id=pad_token_id,
            use_cache=True,
            **kwargs)
    return tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()


def divide_range(paths_list, start, end, chunk_size=16):
    """Split a frame list into consecutive chunks; fold a short tail back in."""
    if end > chunk_size:
        chunks = [paths_list[i: min(i + chunk_size, end)]
                  for i in range(start, end, chunk_size)]
        if len(chunks[-1]) < chunk_size:
            chunks[-2].extend(chunks[-1])
            chunks.pop()
    else:
        chunks = [paths_list[start:end]]
    return chunks


def load_frames_from_directory(directory_path, num_segments=16):
    paths_list = sorted(
        os.path.join(directory_path, f)
        for f in os.listdir(directory_path)
        if f.endswith(('.png', '.jpg', '.jpeg'))
    )
    chunks = divide_range(paths_list, 0, len(paths_list), chunk_size=num_segments)

    img_grids = []
    for image_files in chunks:
        img_array = np.stack(
            [np.array(Image.open(f).convert("RGB")) for f in image_files])
        img_grid = create_frame_grid(img_array, 50)
        img_grid = Image.fromarray(img_grid).convert("RGB")
        img_grids.append(resize_image_grid(img_grid))

    return img_grids, f"The directory contains {len(paths_list)} frames."


def describe_video(model, processor, tokenizer, frames_dir, qs,
                   pre_query_prompt=None, num_frames=16, conv_mode="plain"):
    img_grids, _ = load_frames_from_directory(frames_dir, num_segments=num_frames)

    conv = conv_templates[conv_mode].copy()
    qs = DEFAULT_IMAGE_TOKEN + '\n' + ((pre_query_prompt or "") + qs)
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    data = {'directory': frames_dir}
    for idx, img_grid in enumerate(img_grids):
        data[f'chunk_{idx}'] = video_answer(
            prompt, model=model, processor=processor, tokenizer=tokenizer,
            do_sample=False, img_grid=img_grid, max_new_tokens=512)
    return data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="Lin-Chen/sharegpt4video-8b",
                        help="HF id or local path to the ShareGPT4Video-8B checkpoint.")
    parser.add_argument("--conv-mode", type=str, default="llava_llama_3")
    parser.add_argument("--query", type=str, default=DEFAULT_QUERY)
    parser.add_argument("--json_path", type=str, required=True,
                        help="JSON file: list of video (frame-directory) names to process.")
    parser.add_argument("--main_dir", type=str, required=True,
                        help="Root directory holding one frame-subdirectory per video.")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Directory to write one <video>.json of segment descriptions.")
    parser.add_argument("--num_frames", type=int, default=16,
                        help="Frames per segment.")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, processor, _ = load_pretrained_model(
        model_path, None, model_name, device_map='cuda')
    model = model.cuda().eval()

    with open(args.json_path, "r", encoding="utf-8") as f:
        directories = json.load(f)

    # Skip videos already processed (resumable).
    done = {f[:-5] for f in os.listdir(args.output_dir) if f.endswith(".json")}
    directories = [d for d in directories if d not in done]

    for directory in tqdm(directories):
        video_dir = os.path.join(args.main_dir, directory)
        print(f"Processing {video_dir}")
        outputs = describe_video(
            model, processor, tokenizer, video_dir,
            qs=args.query, pre_query_prompt=PRE_QUERY_PROMPT,
            num_frames=args.num_frames, conv_mode=args.conv_mode)

        with open(os.path.join(args.output_dir, directory + ".json"),
                  "w", encoding="utf-8") as f:
            json.dump(outputs, f, ensure_ascii=False, indent=4)


if __name__ == "__main__":
    main()
