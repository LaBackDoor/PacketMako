import matplotlib.pyplot as plt
from pathlib import Path
import os
import json


class DataProcessor:
    def __init__(self) -> None:
        pass

    def get_prompt(self):
        """
        Get prompt.
        
        Return: prompt
        """
        
        prompt = """Question: Explain the table, covering login attempts, brute force attacks, certificate issues,
        and other network activities. Include causes, detection methods, and mitigation strategies for each scenario. 
        Be clear and specific."""

        return prompt

    def get_dataset(self, path:Path, explanations_path:Path, pcaps_path:Path, name:str):
        """
        Save data directory files in a dataset json file with {"input": "", "output": ""} format
        
        Keyword arguments:
        path -- data directory path
        name -- json file name
        """
        
        dataframe = []
        # go through data sources
        for f in os.listdir(path):
            f = Path(f)
            if f.suffix == ".json":
                data_source_name = Path(f.stem)
                # for each example from that source
                for file in os.listdir(explanations_path / data_source_name):
                    example_name = Path(file)
                    # add example to dataset
                    with open(explanations_path / data_source_name / example_name, encoding="utf8") as fpe:
                        with open(pcaps_path / data_source_name / example_name, encoding="utf8") as fpp:
                                input = f"{self.get_prompt()}\n{fpp.read()}"
                                output = fpe.read()
                                dataframe.append({"input": input, "output": output})
        # save data
        with open(path / Path(name), "w") as fp:
            json.dump(dataframe, fp)

    def prepare_prompt(self, path: Path):
        """
        Get prompt with pcap file content added.
        
        Arguments:
        path -- path to pcap file

        Return: prompt with pcap file content
        """
        
        with open(path) as fp:
            pcap = fp.read()
            # return prompt with pcap content concatenation
            return f"{self.get_prompt()}\n{pcap}"
        

    # Formatting function for prompts
    def formatting_func(self, example):
        """
        Format and join input and output of an entry from dataset on a same string.
        
        Arguments:
        example -- dataset entry with input and output

        Return: formatted prompt
        """
        return f"### Question: {example['input']}\n ### Answer: {example['output']}"

    # Tokenization and dataset preparation
    def generate_and_tokenize_prompt(self, tokenizer, prompt):
        """
        Tokenize prompt.
        
        Arguments:
        tokenizer -- tokenizer to use
        prompt -- input prompt

        Return: tokenized prompt
        """
        return tokenizer(self.formatting_func(prompt))

    def generate_and_tokenize_prompt2(self, tokenizer, prompt, max_length=512):
        """
        Tokenize prompt with padding and truncation if needed.
        
        Arguments:
        tokenizer -- tokenizer to use
        prompt -- input prompt

        Keyword arguments:
        max_length -- maximum length of prompt

        Return: tokenized prompt
        """
        
        result = tokenizer(self.formatting_func(prompt), truncation=True, max_length=max_length, padding="max_length")
        result["labels"] = result["input_ids"].copy()
        return result

    # Plot dataset entries lengths
    def plot_data_lengths(self, tokenized_train_dataset, tokenized_val_dataset):
        """
        Plot dataset entries lengths.
        
        Arguments:
        tokenized_train_dataset -- train dataset already tokenized
        tokenized_val_dataset -- validation dataset already tokenized
        """
        lengths = [len(x['input_ids']) for x in tokenized_train_dataset]
        lengths += [len(x['input_ids']) for x in tokenized_val_dataset]

        # Plotting the histogram
        plt.figure(figsize=(10, 6))
        plt.hist(lengths, bins=20, alpha=0.7, color='blue')
        plt.xlabel('Length of input_ids')
        plt.ylabel('Frequency')
        plt.title('Distribution of Lengths of input_ids')
        plt.savefig("data_lengths.png")