import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from accelerate import Accelerator, FullyShardedDataParallelPlugin
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from torch.distributed.fsdp.fully_sharded_data_parallel import FullOptimStateDictConfig, FullStateDictConfig
from transformers import (AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig,
                          TrainingArguments, Trainer,
                          DataCollatorForLanguageModeling)
import torch
import wandb
from datetime import datetime
from configparser import ConfigParser
from model.data_processing import DataProcessor

class ModelTrainer:
    def __init__(self, config_path) -> None:
        # configuration parameters
        c = ConfigParser()
        c.read(config_path)
        self.hf_token = c.get("DEFAULT", "token") # HuggingFace model token
        self.base_model_id = c.get("DEFAULT", "model") # HuggingFace Model id
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # paths
        base_path = c.get("DEFAULT", "base_path") # base path where data is located and where model will be saved
        self.data_train_path = base_path + c.get("DEFAULT", "train_data_path") # train data file name
        self.data_val_path = base_path + c.get("DEFAULT", "validation_data_path") # validation data file name
        self.data_processor = DataProcessor()
        # accelerator
        fsdp_plugin = FullyShardedDataParallelPlugin(
            state_dict_config=FullStateDictConfig(offload_to_cpu=True, rank0_only=False),
            optim_state_dict_config=FullOptimStateDictConfig(offload_to_cpu=True, rank0_only=False),
        )
        self.accelerator = Accelerator(fsdp_plugin=fsdp_plugin)

    # Print trainable parameters of the model
    def print_trainable_parameters(self, model):
        """
        Print the number of trainable parameters in the model.
        
        Arguments:
        model -- model
        """
        trainable_params = 0
        all_param = 0
        for _, param in model.named_parameters():
            all_param += param.numel()
            if param.requires_grad:
                trainable_params += param.numel()
        print(
            f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
        )

    def __load_data(self):
        # load datasets
        train_dataset = load_dataset('json', data_files=self.data_train_path, split='train')
        eval_dataset = load_dataset('json', data_files=self.data_val_path, split='train')

        return train_dataset, eval_dataset
    
    def __tokenize_data(self, data, tokenizer, max_length=1700):
        # tokenize prompts
        tokenized_dataset = data.map(lambda x : 
                                           self.data_processor.generate_and_tokenize_prompt2(tokenizer, x, max_length=max_length))
        return tokenized_dataset
    
    def __load_model(self):
        print("Loading model...")
        # base model
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16
        )
        model = AutoModelForCausalLM.from_pretrained(
            self.base_model_id,
            device_map={"": 0},
            quantization_config=bnb_config,
            token=self.hf_token)

        return model
    
    def __load_tokenizer(self):
        # tokenization
        tokenizer = AutoTokenizer.from_pretrained(self.base_model_id, 
                                                padding_side="left", 
                                                add_eos_token=True, 
                                                add_bos_token=True,
                                                token=self.hf_token)
        tokenizer.pad_token = tokenizer.eos_token

        return tokenizer
    
    def __setup_lora(self, m):
        m.gradient_checkpointing_enable()
        model = prepare_model_for_kbit_training(m)

        # LoRA configuration and model preparation
        config = LoraConfig(
            r=32,
            lora_alpha=64,
            target_modules=[
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
                "lm_head",
            ],
            bias="none",
            lora_dropout=0.05,  # Conventional
            task_type="CAUSAL_LM",
        )

        model = get_peft_model(model, config)

        return model

    def train(self, output_dir, run_name="llama2-13b -- traice", wandb_project="traice"):
        """
        Train model.
        
        Keyword arguments:
        output_dir -- directory in which the trained model will be saved
        run_name -- wandb run name
        wandb_project -- wandb project name
        """
        # create tokenizer
        print("Creating tokenizer...")
        tokenizer = self.__load_tokenizer()
        
        # load data
        print("Loading data...")
        train_dataset, val_dataset = self.__load_data()
        print("Tokenize data...")
        tokenized_train_dataset = self.__tokenize_data(train_dataset, tokenizer)
        tokenized_val_dataset = self.__tokenize_data(val_dataset, tokenizer)

        # set up lora
        print("Setting up lora...")
        model = self.__setup_lora(self.__load_model())

        # apply the accelerator
        print("Applying accelerator...")
        # you can comment this out to remove the accelerator.
        model = self.accelerator.prepare_model(model)

        # weights and biases setup
        wandb.login()
        os.environ["WANDB_PROJECT"] = wandb_project

        # training configuration
        print("Configuring training...")
        if torch.cuda.device_count() > 1: # If more than 1 GPU
            print("More than one GPU...")
            model.is_parallelizable = True
            model.model_parallel = True

        # training execution
        trainer = Trainer(
            model=model,
            train_dataset=tokenized_train_dataset,
            eval_dataset=tokenized_val_dataset,
            args=TrainingArguments(
                output_dir=output_dir,
                warmup_steps=1,
                per_device_train_batch_size=2,
                gradient_accumulation_steps=1,
                max_steps=500,
                learning_rate=2.5e-5, # Want a small lr for finetuning
                # bf16=True,
                optim="paged_adamw_8bit",
                logging_dir="./logs",        # Directory for storing logs
                save_strategy="steps",       # Save the model checkpoint every logging step
                save_steps=50,                # Save checkpoints every 50 steps
                evaluation_strategy="steps", # Evaluate the model every logging step
                eval_steps=50,               # Evaluate and save checkpoints every 50 steps
                do_eval=True,                # Perform evaluation at the end of training
                report_to="wandb",           # Comment this out if you don't want to use weights & baises
                run_name=f"{run_name}-{datetime.now().strftime('%Y-%m-%d-%H-%M')}"          # Name of the W&B run (optional)
            ),
            data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
        )

        model.config.use_cache = False  # silence the warnings. Please re-enable for inference!
        trainer.train()

    def run_model(self, prompt, model, tokenizer, max_length=1700):
        """
        Get inference from a prompt using model.
        
        Keyword arguments:
        prompt -- input prompt
        model -- model
        tokenizer -- tokenizer
        """
        
        model_input = tokenizer(prompt, return_tensors="pt").to(self.device)
        model.eval()
        with torch.no_grad():
            print(tokenizer.decode(model.generate(**model_input, max_new_tokens=max_length, pad_token_id=2).to(self.device)[0], skip_special_tokens=True))