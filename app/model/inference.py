import torch
from transformers import AutoTokenizer, BartForConditionalGeneration

class TextSummarizer:
    def __init__(self, model_name: str = "facebook/bart-large-cnn", device: str = "cpu"):
        self.model_name = model_name
        self.device = torch.device("cpu") #device if device else self._set_device()
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = BartForConditionalGeneration.from_pretrained(self.model_name).to(self.device)

    def _set_device(self) -> torch.device:
        if torch.cuda.is_available():
            print("Using CUDA GPU")
            return torch.device("cuda")
        elif torch.backends.mps.is_available():
            print("Using MPS")
            return torch.device("mps")
        else:
            print("Using CPU")
            return torch.device("cpu")

    def summarize(self, text: str, 
                   max_input_length: int = 1024,
                   num_beams: int = 2,
                   min_length: int = 0,
                   max_length: int = 20) -> str:

        inputs = self.tokenizer(
            [text], 
            max_length=max_input_length, 
            return_tensors="pt", 
            truncation=True
        ).to(self.device)

        summary_ids = self.model.generate(
            inputs["input_ids"], 
            num_beams=num_beams, 
            min_length=min_length, 
            max_length=max_length
        )

        result = self.tokenizer.batch_decode(
            summary_ids, 
            skip_special_tokens=True, 
            clean_up_tokenization_spaces=False
        )[0]

        return result
