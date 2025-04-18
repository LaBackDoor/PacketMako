import pytest
import torch
from src.tokenization.hybrid_byt5_tokenizer import HybridByT5PCAPTokenizer
from transformers import T5ForConditionalGeneration


class TestTokenizerModelIntegration:

    @pytest.fixture
    def tokenizer_and_model(self):
        tokenizer = HybridByT5PCAPTokenizer(pcap_vocab_size=277)
        model = T5ForConditionalGeneration.from_pretrained("google/byt5-small")
        return tokenizer, model

    def test_model_input_compatibility(self, tokenizer_and_model):
        """Test tokenizer outputs are compatible with model inputs"""
        tokenizer, model = tokenizer_and_model

        # Create a simple tokenized input
        text = "This is a test"
        token_ids = tokenizer.encode_mixed_input(text=text)

        # Convert to tensor format expected by the model
        input_ids = torch.tensor([token_ids])
        attention_mask = torch.ones_like(input_ids)

        # Verify model accepts the input without errors
        try:
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            assert outputs is not None
        except Exception as e:
            pytest.fail(f"Model failed to process tokenizer output: {e}")