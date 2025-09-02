# MoodMuse 🎶✨

**MoodMuse** is a lightweight GPT-style text generation project fine-tuned to create **Instagram-style photo captions with aesthetic vibes**.  
Given a short description (e.g. *“golden hour by the sea”*) and a target **vibe** (*moody, minimal, playful, wanderlust*), the model generates captions styled with hashtags and short text.

---

## 🚀 Project Overview

- **Goal**: Learn how to build, train, and run a GPT-style character-level language model from scratch.  
- **Training**: Used Instagram caption dataset from Kaggle → cleaned, tokenized at character level → trained with GPT blocks.  
- **Checkpoint**: Final model checkpoint saved as `moodmuse_best.pt`.  
- **Inference**: Runs standalone with `inference.py`, loading the trained checkpoint and generating captions with top-k sampling.

---

## ⚙️ Installation

Clone the repository:

```bash
git clone https://github.com/pranami4501/MoodMuse.git
cd MoodMuse
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## 🧩 Components

- **src/gpt.py**: Implements a mini GPT model from scratch:

1. Multi-head causal self-attention

2. Feedforward layers

3. Positional embeddings

4. Used for training (not for inference directly).

- **src/inference.py**: 
1. A compatibility wrapper to load checkpoints saved during training.

2. Defines CompatGPT, which matches the saved weights exactly.

3. Provides generate_caption() function:

   Adds style prompt (#moody, #playful, etc.)

   Generates captions using top-k sampling.
- **ckpts/moodmuse_best.pt**:
1. Trained checkpoint (~15 MB).

2. Contains model weights, vocabulary, and block size.

- **data/tokenizer_char.json**:

1. Vocabulary mappings (stoi, itos).

2. Required for encoding/decoding characters during inference.

---

## ▶️ Running Inference

To generate captions:

```bash
python src/inference.py \
  --ckpt ckpts/moodmuse_best.pt \
  --desc "golden hour by the sea" \
  --vibe moody
```

Parameters

--ckpt: path to trained checkpoint (ckpts/moodmuse_best.pt).

--desc: short description of image.

--vibe: one of [moody, minimal, playful, wanderlust].

--max_new_tokens: how long to generate (default 120).

--temperature: creativity level (lower = safer, higher = wilder).

--top_k: restricts vocabulary to top-k likely tokens.
