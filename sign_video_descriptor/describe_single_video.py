"""Sign Video Descriptor — single-video demo.

Runs the Stage-1 VideoLLM description on one video file (e.g. an .mp4) instead
of a directory of frames. Useful for quickly inspecting the descriptor output.

Requires the ShareGPT4Video / LLaVA package (``llava.*``) and ``decord``.
See: https://github.com/ShareGPT4Omni/ShareGPT4Video
"""

import argparse
import os

import numpy as np
import torch
from decord import VideoReader, cpu
from PIL import Image

from llava.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX
from llava.conversation import conv_templates
from llava.mm_utils import (get_model_name_from_path, process_images,
                            tokenizer_image_token)
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init

from describe_segments import (DEFAULT_QUERY, create_frame_grid,
                               resize_image_grid, video_answer)

PRE_QUERY_PROMPT = (
    "This is a sign language video, and it's important to capture "
    "details such as hand gestures and facial expressions accurately. Please "
    "identify and describe the specific motion in this video, including hand "
    "position, movement, shape and any relevant facial expressions or emotions "
    "shown."
)


def load_video(video_path, num_segments=32):
    def get_index(num_frames, num_segments):
        seg_size = float(num_frames - 1) / num_segments
        start = int(seg_size / 2)
        return np.array([start + int(np.round(seg_size * idx))
                         for idx in range(num_segments)])

    vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
    frame_indices = get_index(len(vr), num_segments)
    img_array = vr.get_batch(frame_indices).asnumpy()
    img_grid = create_frame_grid(img_array, 50)
    img_grid = Image.fromarray(img_grid).convert("RGB")
    return resize_image_grid(img_grid)


def describe(model, processor, tokenizer, vid_path, qs,
             pre_query_prompt=None, num_frames=32, conv_mode="plain"):
    img_grid = load_video(vid_path, num_segments=num_frames)

    conv = conv_templates[conv_mode].copy()
    qs = DEFAULT_IMAGE_TOKEN + '\n' + ((pre_query_prompt or "") + qs)
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    return video_answer(prompt, model=model, processor=processor,
                        tokenizer=tokenizer, do_sample=False,
                        img_grid=img_grid, max_new_tokens=512)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="Lin-Chen/sharegpt4video-8b")
    parser.add_argument("--video", type=str, required=True, help="Path to a video file.")
    parser.add_argument("--conv-mode", type=str, default="llava_llama_3")
    parser.add_argument("--query", type=str, default=DEFAULT_QUERY)
    parser.add_argument("--num_frames", type=int, default=32)
    args = parser.parse_args()

    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, processor, _ = load_pretrained_model(
        model_path, None, model_name, device_map='cuda')
    model = model.cuda().eval()

    output = describe(model, processor, tokenizer, args.video, qs=args.query,
                      pre_query_prompt=PRE_QUERY_PROMPT,
                      num_frames=args.num_frames, conv_mode=args.conv_mode)
    print(output)


if __name__ == "__main__":
    main()
