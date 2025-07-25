from make_model import get_adapter_model
from datasets import load_dataset
from trl import SFTTrainer
from transformers import TrainingArguments
from custom_trainer import CustomSFTTrainer_w_kl

model, tokenizer = get_adapter_model("unsloth/Qwen2.5-0.5B-Instruct-bnb-4bit", 8, 16, 0.1)

dataset = load_dataset("trl-lib/Capybara", split="train[:10%]")

# trainer = SFTTrainer(
#     model=model,
#     train_dataset=dataset,
#     args=TrainingArguments(
#         per_device_train_batch_size=1,
#         gradient_accumulation_steps=4,
#     ),
# )

trainer = CustomSFTTrainer_w_kl(
    model=model,
    args=TrainingArguments(
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        logging_steps=10,
    ),
    train_dataset=dataset,
    kl_alpha=0.1,
)
trainer.train()