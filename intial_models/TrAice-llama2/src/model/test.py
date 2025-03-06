import torch
from configparser import ConfigParser
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
from pathlib import Path
from model.data_processing import DataProcessor

class ModelTester:
    def __init__(self, config_path) -> None:
        # configuration parameters
        c = ConfigParser()
        c.read(config_path)
        self.hf_token = c.get("DEFAULT", "token")
        self.base_model_id = c.get("DEFAULT", "model")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.data_processor = DataProcessor()

    def __load_model(self):
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
        
    def __load_tokenizer(self, max_length=1700):
        # create tokenizer with padding and truncation
        tokenizer = AutoTokenizer.from_pretrained(self.base_model_id,
                                        add_eos_token=True, 
                                        add_bos_token=True,
                                        trust_remote_code=True,
                                        truncation=True, 
                                        max_length=max_length,
                                        token=self.hf_token)
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding="max_length"
        tokenizer.padding_side="left"

        return tokenizer

    def test(self, model_path, pcap_path, max_length=1700):
        """
        Get explanation of a pcap file from the trained model.
        
        Keyword arguments:
        model_path -- trained model path
        pcap_path -- pcap file path
        Return: pcap file explanation.
        """
        
        # check parameters
        if model_path == None or pcap_path == None:
            print("Invalid arguments: <model_path> and <pcap_file_path> cannot be None.")
            return
        
        # load model
        print("Loading model...")
        model = self.__load_model()

        # load tokenizer
        print("Loading tokenizer...")
        tokenizer = self.__load_tokenizer(max_length=max_length)

        # Load the QLoRA adapter from the appropriate checkpoint directory
        ft_model = PeftModel.from_pretrained(model, model_path)

        # run inference
        eval_prompt_input = self.data_processor.prepare_prompt(Path(pcap_path)) # build prompt with pcap file content
        model_input = tokenizer(eval_prompt_input,
                                return_tensors="pt",
                                truncation=True, 
                                max_length=max_length,
                                padding="max_length").to(self.device) # load model

        
        # generate response
        ft_model.eval()
        with torch.no_grad():
            exp = tokenizer.decode(ft_model.generate(**model_input, max_new_tokens=max_length, pad_token_id=2)[0], skip_special_tokens=True)
            print(exp)
            return exp