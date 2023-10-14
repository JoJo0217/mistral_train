from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig,HfArgumentParser,TrainingArguments,pipeline, logging, TextStreamer
from peft import LoraConfig, PeftModel, prepare_model_for_kbit_training, get_peft_model
import os,torch, wandb, platform, gradio, warnings
from datasets import load_dataset
from trl import SFTTrainer
from huggingface_hub import notebook_login
base_model, new_model = "mistralai/Mistral-7B-v0.1" ,  "mistral_7b"
dataset_name='./data/train.jsonl'
dataset = load_dataset('json',data_files=dataset_name,split='train')

#data = dataset.map(
#     lambda x: 
#     {'text': f"아래는 작업을 설명하는 명령어와 추가 컨텍스트를 제공하는 입력이 짝을 이루는 예제입니다. 요청을 적절히 완료하는 응답을 작성하세요.\n\n### 명령어:\n{x['instruction']}\n\n### 입력:\n{x['input']}\n\n### 응답:\n{x['output']}" } # <|endoftext|>
#     if x['input'] else 
#     {'text':f"아래는 작업을 설명하는 명령어입니다. 요청을 적절히 완료하는 응답을 작성하세요.\n\n### 명령어:\n{x['instruction']}\n\n### 응답:\n{x['output']}"},
#)
dataset = dataset.map(
     lambda x: 
     {'text': f"[INST]{x['instruction']}{x['input']}[\INST]{x['output']}"} # <|endoftext|>
     if x['input'] else 
     {'text':f"[INST]{x['instruction']}[\INST]{x['output']}"},
)
dataset = dataset.remove_columns("instruction")
dataset = dataset.remove_columns("input")
dataset = dataset.remove_columns("output")
# Load base model(Mistral 7B)
bnb_config = BitsAndBytesConfig(
    load_in_4bit= True,
    bnb_4bit_quant_type= "nf4",
    bnb_4bit_compute_dtype= torch.bfloat16,
    bnb_4bit_use_double_quant= False,
)

model = AutoModelForCausalLM.from_pretrained(
   base_model,
    quantization_config=bnb_config,
    device_map={"": 0}
)
model.config.use_cache = False # silence the warnings. Please re-enable for inference!
model.config.pretraining_tp = 1
model.gradient_checkpointing_enable()
# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
tokenizer.padding_side = 'right'
tokenizer.pad_token = tokenizer.eos_token
tokenizer.add_eos_token = True
tokenizer.add_bos_token, tokenizer.add_eos_token

#Adding the adapters in the layers
model = prepare_model_for_kbit_training(model)
peft_config = LoraConfig(
        r=16,
        lora_alpha=16,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj","gate_proj"]
    )
model = get_peft_model(model, peft_config)
# Monitering the LLM
#Hyperparamter
model.print_trainable_parameters()
training_arguments = TrainingArguments(
    output_dir= "./results",
    num_train_epochs= 3,
    per_device_train_batch_size= 1,
    gradient_accumulation_steps= 2,
    optim = "paged_adamw_8bit",
    save_strategy="no",
    save_steps= 1000,
    logging_steps= 1,
    learning_rate= 3e-4,
    weight_decay= 0.1,
    fp16= False,
    bf16= False,
    max_grad_norm= 0.3,
    max_steps= -1,
    warmup_ratio= 0.3,
    group_by_length= False,
    lr_scheduler_type= "constant",
    report_to="tensorboard"
)
# Setting sft parameters
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    peft_config=peft_config,
    max_seq_length= 1024,
    dataset_text_field="text",
    tokenizer=tokenizer,
    args=training_arguments,
    packing= False,
)
trainer.train()
# Save the fine-tuned model
trainer.model.save_pretrained(new_model)
model.config.use_cache = True
model.eval()
