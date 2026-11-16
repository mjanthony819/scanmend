# Improvement plans (post-v1)

- Noise head: apply noise AFTER downscale to 160px in synthesize.py so the signal survives; regenerate data, retrain, re-eval.
- Real-photo validation set (a few dozen phone photos of printed pages) to confirm synthetic-trained estimates transfer.
- Accessible (tagged) PDF export with reading order, if a tagging library can be verified.
