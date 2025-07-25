# Custom SFT Trainer with KL divergence loss

This repo provides an extension of Hugging Face's trl.SFTTrainer that adds a KL divergence loss between a LoRA-adapted model and its base counterpart. It enables more stable and conservative fine-tuning by regularizing the adapted model's predictions against its original distribution.


## Custom Loss 

The custom loss is implemented in the `custom_trainer.py` file. It extends the `SFTTrainer` class and overrides the `compute_loss` method to add a KL divergence loss term. 

## Training

The training script is implemented in the `train.py` file. You can compare the custom loss to the standard SFT loss.
