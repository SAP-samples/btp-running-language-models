import os
import sys
import torch
import transformers
import huggingface_hub

transformers.utils.logging.set_verbosity_error()
transformers.utils.logging.disable_progress_bar()

HUB_TOKEN = "hf_"
huggingface_hub.login(token=HUB_TOKEN)


os.environ["TRANSFORMERS_CACHE"] = "shared/IMR/llm2023/cache"

class Model:
    generator = None

    def setup():
        """model setup"""
        print("START LOADING SETUP", file=sys.stderr)   # somehow AI Cores logs only show the error stream :)
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        model_name = "meta-llama/Llama-2-7b-chat-hf"
        
        tokenizer = transformers.AutoTokenizer.from_pretrained(model_name, use_auth_token=True)
        
        pipeline = transformers.pipeline(
            "text-generation",
            model=model_name,
            tokenizer=tokenizer,
            torch_dtype=torch.bfloat16,
            device=device,
            trust_remote_code=True,
            use_auth_token=True
        )
        
        Model.generator = lambda prompt, args: pipeline(
            prompt,
            **{
                "max_length": 2000,
                "do_sample": True,
                "top_k": 10,
                "num_return_sequences": 1,
                "eos_token_id": tokenizer.eos_token_id,
                **args
            }
        )

        print("SETUP DONE", file=sys.stderr)

    def predict(prompt, args):
        """model setup"""
        return Model.generator(prompt, args) 

if __name__ == "__main__":
    # for local testing
    Model.setup()
    print(Model.predict("Hello, who are you?", {}))
