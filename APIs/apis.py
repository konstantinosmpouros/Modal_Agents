import modal
from modal import App, Image

# Basic setup
MODEL_CONFIG = {
    "image": Image.debian_slim().pip_install('huggingface', 'torch', 'transformers', 'accelerate', 'bitsandbytes', 'sentencepiece'),
    "timeout": 600,
    "gpu": 'T4',
    "secrets": [modal.Secret.from_name("huggingface-secret")]
}

# Modal Llama API setup
LLAMA_MODEL = 'meta-llama/Meta-Llama-3.1-8B-Instruct'
llama_app = App('llama-app')
@llama_app.cls(**MODEL_CONFIG)
class Llama():
    @modal.build()
    def download_model(self):
        from huggingface_hub import snapshot_download
        
        # Download the model from huggingface
        snapshot_download(LLAMA_MODEL)

    @modal.enter()
    def setup(self):
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM
        from transformers import BitsAndBytesConfig
        
        # Quant configs
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type='nf4'
        )

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(LLAMA_MODEL)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = 'right'

        # Load model
        self.model = AutoModelForCausalLM.from_pretrained(LLAMA_MODEL,
                                                    device_map="cuda",
                                                    quantization_config=quant_config)
        self.model.generation_config.pad_token_id = self.tokenizer.pad_token_id
    
    @modal.method()
    def inference(self, history: list) -> any:
        import torch
        from threading import Thread
        from transformers import TextIteratorStreamer
        
        # Encode the prompt
        tokenized_message = self.tokenizer.apply_chat_template(history,
                                                    return_tensors="pt").to('cuda')
        attention_mask = torch.ones(tokenized_message.shape, device='cuda')

        # Initialize a stream to stream the response back
        streamer = TextIteratorStreamer(self.tokenizer, 
                                        skip_prompt=True,
                                        skip_special_tokens=True)

        # Generate response with in a thread
        generation_kwargs = {
            "input_ids": tokenized_message,  # Correctly pass input IDs
            "attention_mask": attention_mask,
            "streamer": streamer,
            "temperature": 0.85,
            "max_new_tokens": 1000,
        }
        thread = Thread(target=self.model.generate, kwargs=generation_kwargs)
        thread.start()

        for chunk in streamer:
            yield chunk

    @modal.method()
    def ping(self):
        return "pong"


# Modal Qwen API setup
QWEN_MODEL = 'Qwen/Qwen2.5-7B-Instruct'
qwen_app = App('qwen-app')
@qwen_app.cls(**MODEL_CONFIG)
class Qwen():
    @modal.build()
    def download_model(self):
        from huggingface_hub import snapshot_download
        
        # Download the model from huggingface
        snapshot_download(QWEN_MODEL)

    @modal.enter()
    def setup(self):
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM
        from transformers import BitsAndBytesConfig
        
        # Quant configs
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type='nf4'
        )

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(QWEN_MODEL)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = 'right'

        # Load model
        self.model = AutoModelForCausalLM.from_pretrained(QWEN_MODEL,
                                                    device_map="cuda",
                                                    quantization_config=quant_config)
        self.model.generation_config.pad_token_id = self.tokenizer.pad_token_id

    @modal.method()
    def inference(self, history: list) -> any:
        import torch
        from threading import Thread
        from transformers import TextIteratorStreamer
        
        # Encode the prompt
        tokenized_message = self.tokenizer.apply_chat_template(history,
                                                    return_tensors="pt").to('cuda')
        attention_mask = torch.ones(tokenized_message.shape, device='cuda')

        # Initialize a stream to stream the response back
        streamer = TextIteratorStreamer(self.tokenizer, 
                                        skip_prompt=True,
                                        skip_special_tokens=True)

        # Generate response with in a thread
        generation_kwargs = {
            "input_ids": tokenized_message,  # Correctly pass input IDs
            "attention_mask": attention_mask,
            "streamer": streamer,
            "temperature": 0.85,
            "max_new_tokens": 1000,
        }
        thread = Thread(target=self.model.generate, kwargs=generation_kwargs)
        thread.start()

        for chunk in streamer:
            yield chunk

    @modal.method()
    def ping(self):
        return "pong"


# Modal Gemma API setup
GEMMA_MODEL = 'google/gemma-2-9b-it'
gemma_app = App('gemma-app')
@gemma_app.cls(**MODEL_CONFIG)
class Gemma():
    @modal.build()
    def download_model(self):
        from huggingface_hub import snapshot_download
        
        # Download the model from Hugging Face
        snapshot_download(GEMMA_MODEL)

    @modal.enter()
    def setup(self):
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM
        from transformers import BitsAndBytesConfig
        
        # Quantization configuration for Gemma
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type='nf4'
        )

        # Load the tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(GEMMA_MODEL)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = 'right'

        # Load the model with quantization settings
        self.model = AutoModelForCausalLM.from_pretrained(
            GEMMA_MODEL,
            device_map="cuda",
            quantization_config=quant_config
        )
        self.model.generation_config.pad_token_id = self.tokenizer.pad_token_id

    @modal.method()
    def inference(self, history: list) -> any:
        import torch
        from threading import Thread
        from transformers import TextIteratorStreamer
        
        # Encode the prompt
        tokenized_message = self.tokenizer.apply_chat_template(
            history,
            return_tensors="pt"
        ).to('cuda')
        attention_mask = torch.ones(tokenized_message.shape, device='cuda')

        # Initialize a streamer to stream the response back
        streamer = TextIteratorStreamer(
            self.tokenizer,
            skip_prompt=True,
            skip_special_tokens=True
        )

        # Generate response in a separate thread
        generation_kwargs = {
            "input_ids": tokenized_message,  # Correctly pass input IDs
            "attention_mask": attention_mask,
            "streamer": streamer,
            "temperature": 0.85,
            "max_new_tokens": 1000,
        }
        thread = Thread(target=self.model.generate, kwargs=generation_kwargs)
        thread.start()

        # Yield the generated chunks from the streamer
        for chunk in streamer:
            yield chunk

    @modal.method()
    def ping(self):
        return "pong"


# Modal Phi API setup
PHI_MODEL = 'microsoft/phi-4'
phi_app = App('phi-app')
@phi_app.cls(**MODEL_CONFIG)
class Phi():
    @modal.build()
    def download_model(self):
        from huggingface_hub import snapshot_download
        
        # Download the model from Hugging Face
        snapshot_download(PHI_MODEL)

    @modal.enter()
    def setup(self):
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM
        from transformers import BitsAndBytesConfig
        
        # Quantization configuration for Gemma
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type='nf4'
        )

        # Load the tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(PHI_MODEL)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = 'right'

        # Load the model with quantization settings
        self.model = AutoModelForCausalLM.from_pretrained(
            PHI_MODEL,
            device_map="cuda",
            quantization_config=quant_config
        )
        self.model.generation_config.pad_token_id = self.tokenizer.pad_token_id

    @modal.method()
    def inference(self, history: list) -> any:
        import torch
        from threading import Thread
        from transformers import TextIteratorStreamer
        
        # Encode the prompt
        tokenized_message = self.tokenizer.apply_chat_template(
            history,
            return_tensors="pt"
        ).to('cuda')
        attention_mask = torch.ones(tokenized_message.shape, device='cuda')

        # Initialize a streamer to stream the response back
        streamer = TextIteratorStreamer(
            self.tokenizer,
            skip_prompt=True,
            skip_special_tokens=True
        )

        # Generate response in a separate thread
        generation_kwargs = {
            "input_ids": tokenized_message,  # Correctly pass input IDs
            "attention_mask": attention_mask,
            "streamer": streamer,
            "temperature": 0.85,
            "max_new_tokens": 1000,
        }
        thread = Thread(target=self.model.generate, kwargs=generation_kwargs)
        thread.start()

        # Yield the generated chunks from the streamer
        for chunk in streamer:
            yield chunk

    @modal.method()
    def ping(self):
        return "pong"


# Modal Mistral API setup
MISTRAL_MODEL = 'mistralai/Mistral-7B-Instruct-v0.3'
mistral_app = App('mistral-app')
@mistral_app.cls(**MODEL_CONFIG)
class Mistral():
    @modal.build()
    def download_model(self):
        from huggingface_hub import snapshot_download
        
        # Download the model from Hugging Face
        snapshot_download(MISTRAL_MODEL)

    @modal.enter()
    def setup(self):
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM
        from transformers import BitsAndBytesConfig
        
        # Quantization configuration for Mistral
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type='nf4'
        )

        # Load the tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(MISTRAL_MODEL, use_fast=True)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = 'right'

        # Load the model with quantization settings
        self.model = AutoModelForCausalLM.from_pretrained(
            MISTRAL_MODEL,
            device_map="cuda",
            quantization_config=quant_config
        )
        self.model.generation_config.pad_token_id = self.tokenizer.pad_token_id

    @modal.method()
    def inference(self, history: list) -> any:
        import torch
        from threading import Thread
        from transformers import TextIteratorStreamer
        
        # Encode the prompt
        tokenized_message = self.tokenizer.apply_chat_template(
            history,
            return_tensors="pt"
        ).to('cuda')
        attention_mask = torch.ones(tokenized_message.shape, device='cuda')

        # Initialize a streamer to stream the response back
        streamer = TextIteratorStreamer(
            self.tokenizer,
            skip_prompt=True,
            skip_special_tokens=True
        )

        # Generate response in a separate thread
        generation_kwargs = {
            "input_ids": tokenized_message,  # Correctly pass input IDs
            "attention_mask": attention_mask,
            "streamer": streamer,
            "temperature": 0.85,
            "max_new_tokens": 1000,
        }
        thread = Thread(target=self.model.generate, kwargs=generation_kwargs)
        thread.start()

        # Yield the generated chunks from the streamer
        for chunk in streamer:
            yield chunk

    @modal.method()
    def ping(self):
        return "pong"
