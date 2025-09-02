# MoodMuse — Style-Guided Caption Generator

Generate short, aesthetic Instagram captions from a brief photo description and a style (e.g., `moody`, `minimal`, `playful`, `wanderlust`).

## Quickstart
```bash
pip install -r requirements.txt
python -m src.inference \
  --ckpt ckpts/moodmuse_best.pt \
  --desc "golden hour on the beach" \
  --vibe "moody"
