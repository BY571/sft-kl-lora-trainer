import transformers
import torch
from peft import LoraConfig, get_peft_model, TaskType


def get_model(model_name):
    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    return model


def get_tokenizer(tokenizer_name):
    tokenizer = transformers.AutoTokenizer.from_pretrained(tokenizer_name)
    return tokenizer
    

def get_adapter_model(model_name, lora_r, lora_alpha, lora_dropout):
    # get model and tokenizer
    model = get_model(model_name)
    tokenizer = get_tokenizer(model_name)
    
    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],  # adjust depending on model
        lora_dropout=lora_dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM  # or TaskType.SEQ_CLS, etc.
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    return model, tokenizer


    
    