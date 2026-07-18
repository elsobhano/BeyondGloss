# BeyondGloss — SLT Framework

Training code for *Beyond Gloss: A Hand-Centric Framework for Gloss-Free Sign
Language Translation* (Sections 3.1 and 3.3). This is the main framework:
a two-stage pipeline (contrastive **pre-training**, then encoder–decoder
**fine-tuning**) built around a DINOv2 visual encoder, HaMeR hand-feature
distillation, and an mBART translation backbone.

The sign-description generation used as supervision here lives in a separate
module (`../sign_video_descriptor`); this framework consumes its output as
**precomputed description features** (see below).

## 🧠 Method overview

**Pre-training** (`pretrain.py`, `models/Pretrain_Model.py`) optimises a single
combined loss (Eq. 8 in the paper):

```
total = L_align + landa_desc * L_desc + landa_hamer * L_distill
```

with the best configuration `landa_desc = 0.5`, `landa_hamer = 0.3`. Components:

1. **Visual encoder** — DINOv2 (ViT-S/14), LoRA adapters on the top 3 blocks
   (rank 4). See `models/spatial_models/frame_models/dino_adaptor_model.py` and
   `class dino` in `models/clip_models.py`.
2. **HaMeR distillation** (`L_distill`) — a small head (`hamer_mapper_*`) regresses
   HaMeR hand features from the encoder output; masked L2 loss against precomputed
   HaMeR targets (`masked_l2_loss` in `models/Pretrain_Model.py`).
3. **Temporal encoder** — 4-layer transformer (hidden 512, 8 heads), local
   self-attention (window 7), average-pool downsampling, RoPE positional encoding
   (`Transformer` / `MultiHeadAttention` in `models/clip_models.py`).
4. **Video–description alignment** (`L_desc`) — contrastive loss between video
   features and precomputed sign-description features (`Desc_Clip` / the `_desc`
   logits in `SLRCLIP`).
5. **Video–target alignment** (`L_align`) — CLIP-style symmetric contrastive loss
   between video features and the target spoken sentence encoded by mBART
   (`SLRCLIP`, `TextCLIP`, `KLLoss`).

**Fine-tuning** (`finetune.py`, `models/Finetune_Model.py`) loads the pre-trained
encoder and attaches an mBART encoder–decoder (`gloss_free_model`) with LoRA on
`q_proj`/`v_proj`, trained with cross-entropy for translation.

### A note on the two mBARTs

The paper describes two text encoders: mBART-large-50 for descriptions and
mBART-large-cc25 for the target sentence. In this code they play different roles:

- **Description features are precomputed offline** (with mBART-large-50) and stored
  under `data.desc_path`; the framework just loads them (`load_desc` in
  `dataset/slt_dataset.py`) and feeds them to the video–description alignment head.
  No description mBART runs at train time.
- The **in-framework** text encoder / SLT backbone is a trimmed mBART
  (`model.transformer` / `model.tokenizer` in the config), used for video–target
  alignment and for the translation decoder.

## 📁 Layout

```
pretrain.py              # pre-training entry point (full combined loss)
finetune.py              # fine-tuning entry point (SLT)
pretrain-desc.py         # ablation: description-alignment-only pre-training
configs/
    config.yaml          # Phoenix14T
    csldaily.yaml        # CSL-Daily
models/
    clip_models.py       # SLRCLIP, encoders, temporal transformer, gloss_free_model
    Pretrain_Model.py    # pre-training LightningModule + losses
    Finetune_Model.py    # fine-tuning LightningModule
    dinov2/, spatial_models/   # DINOv2 visual-encoder backbone
dataset/
    slt_dataset.py       # dataset + DataModule (frames, HaMeR feats, desc feats)
    utils.py             # LMDB / feature readers
scripts/                 # example SLURM submission files
launch.sh                # sets PYTHONPATH then execs the given command
trim.py                  # preprocessing: build the trimmed mBART (pretrained/MBart_trimmed)
environment.yml          # conda/pip environment
```

## ⚙️ Setup

```bash
conda env create -f environment.yml
conda activate <env>   # name it as you like
```

Provide the following locally and point the config at them (all paths in
`configs/*.yaml` are placeholders):

- `data.lmdb_path` — preprocessed RGB frames as an LMDB store, one sub-folder per
  split (`train`/`dev`/`test`).
- `data.labels` — label files `<labels>.train`, `<labels>.dev`, `<labels>.test`
  (from the Phoenix14T / CSL-Daily datasets; obtain them from the dataset authors).
- `data.hamer_path` — precomputed HaMeR hand features (distillation targets).
- `data.desc_path` — precomputed sign-description features (from
  `../sign_video_descriptor`, encoded with mBART-large-50).
- `model.dino` — DINOv2 (ViT-S/14) weights; `model.transformer` / `model.tokenizer`
  — the trimmed mBART.

To log with Weights & Biases, run `wandb login` (or set `WANDB_API_KEY` in your
environment); pass `--logger tensorboard` to log locally instead.

## ▶️ Usage

```bash
# 1) Pre-training
./launch.sh python pretrain.py --data_config configs/config.yaml \
    --batch_size 16 --lr 3e-4 --landa_desc 0.5 --landa_hamer 0.3 \
    --warmup 0.05 --scheduler cosine

# 2) Fine-tuning (set training.ckpt_path in the config to the pre-trained checkpoint,
#    or pass --model_ckpt)
./launch.sh python finetune.py --data_config configs/config.yaml \
    --batch_size 16 --lr 4e-4 --num_beams 5 --label_smoothing 0.2 --warmup 0.2
```

Swap `configs/config.yaml` for `configs/csldaily.yaml` to run on CSL-Daily. Example
cluster submission files are in `scripts/`.

## 🙏 Acknowledgements

Built on [DINOv2](https://github.com/facebookresearch/dinov2), mBART
([Hugging Face Transformers](https://github.com/huggingface/transformers)),
[HaMeR](https://github.com/geopavlakos/hamer), [PEFT](https://github.com/huggingface/peft),
and [PyTorch Lightning](https://github.com/Lightning-AI/pytorch-lightning).
