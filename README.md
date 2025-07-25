

This repo provides an extension of Hugging Face's trl.SFTTrainer that adds a KL divergence loss between a LoRA-adapted model and its base counterpart. It enables more stable and conservative fine-tuning by regularizing the adapted model's predictions against its original distribution.
