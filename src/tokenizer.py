from transformers import LlamaTokenizer

class UnlearningTokenizer:
    def __init__(self, model_path: str):
        self.tokenizer = LlamaTokenizer.from_pretrained(
            model_path,
            legacy=False  # Explicitly set legacy behavior
        )
        self.tokenizer.pad_token_id = 0
    
    def prepare_training_batch(self, prompt: str, response: str, max_length: int = 512):
        full_prompt = f"### Instruction:\n{prompt}\n\n### Response:\n{response}"
        
        tokens = self.tokenizer(
            full_prompt,
            max_length=max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        labels = tokens.input_ids.clone()
        labels[labels == self.tokenizer.pad_token_id] = -100
        
        return {
            "input_ids": tokens.input_ids,
            "attention_mask": tokens.attention_mask,
            "labels": labels
        }