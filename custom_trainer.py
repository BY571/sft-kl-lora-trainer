
import torch
from trl import SFTTrainer
from transformers.trainer import Accelerator
import warnings

class CustomSFTTrainer_w_kl(SFTTrainer):
    def __init__(self, model, args, train_dataset, data_collator=None, processing_class=None, kl_alpha=0.1):
        
        super().__init__(model=model,
                         args=args,
                         train_dataset=train_dataset,
                         data_collator=data_collator,
                         processing_class=processing_class)

        if not hasattr(model, 'disable_adapter'):
            warnings.warn(
                "The model provided does not appear to be a PEFT model with a LoRA adapter. "
                "The KL divergence loss term will not be calculated and the trainer will fall back to standard SFT."
            )

        self.accelerator = Accelerator()
        self._total_train_tokens = 0
        # After super().__init__, self._metrics is initialized by the parent Trainer class.
        # We need to add our custom metric to it.
        if "kl_loss" not in self._metrics["train"]:
            self._metrics["train"]["kl_loss"] = []
        if "eval" in self._metrics and "kl_loss" not in self._metrics["eval"]:
            self._metrics["eval"]["kl_loss"] = []
        self.kl_alpha = kl_alpha


    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """
        Computes SFT loss and adds a KL divergence penalty term.
        The reference log probabilities are computed on the fly by disabling the LoRA adapter.
        """
        if self.model_accepts_loss_kwargs:
            loss_kwargs = {}
            if num_items_in_batch is not None:
                loss_kwargs["num_items_in_batch"] = num_items_in_batch
        # Compute standard SFT loss with the adapter enabled
        sft_loss, outputs = super().compute_loss(model, inputs, return_outputs=True, **loss_kwargs)

        # If the model is not a PEFT model, we can't compute KL loss, so we return SFT loss
        if not hasattr(model, 'disable_adapter'):
            warnings.warn(
                "The model provided does not appear to be a PEFT model with a LoRA adapter. "
                "The KL divergence loss term will not be calculated and the trainer will fall back to standard SFT."
            )
            return (sft_loss, outputs) if return_outputs else sft_loss

        # Get the logits from the adapter-enabled model
        logits = outputs.logits
        labels = inputs["labels"]

        # Compute reference logits by disabling the adapter
        with torch.no_grad(), model.disable_adapter():
            ref_outputs = model(
                input_ids=inputs["input_ids"],
                attention_mask=inputs.get("attention_mask", None),
            )
        ref_logits = ref_outputs.logits

        # Shift all tensors for next-token prediction
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        shift_ref_logits = ref_logits[..., :-1, :].contiguous()

        # Create a mask for the answer tokens (non -100 labels)
        answer_mask = (shift_labels != -100).float()

        # Calculate log probabilities from the adapter model and reference model logits
        log_probs = torch.nn.functional.log_softmax(shift_logits, dim=-1)
        ref_log_probs = torch.nn.functional.log_softmax(shift_ref_logits, dim=-1)

        # Calculate KL divergence: KL(P_ref || P_model) = sum(P_ref * (log P_ref - log P_model))
        kl_divergence = torch.exp(ref_log_probs) * (ref_log_probs - log_probs)

        # Sum over the vocabulary dimension and apply the mask and average over the sequence and batch
        kl_loss_per_token = (kl_divergence.sum(dim=-1) * answer_mask)
        kl_loss = kl_loss_per_token.sum() / answer_mask.sum()

        # Combine losses
        loss = sft_loss + self.kl_alpha * kl_loss

        # Logging
        self._metrics["train"]["kl_loss"].append(kl_loss.item())

        return (loss, outputs) if return_outputs else loss