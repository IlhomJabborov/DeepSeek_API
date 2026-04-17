# requirements.txt
# fastapi
# uvicorn
# deepseek-ai

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

class PromptRequest(BaseModel):
    prompt: str

app = FastAPI()

# Load model and tokenizer (only once when server starts)

model_name = "deepseek-ai/deepseek-llm-7b-base"
tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir="./model_cache_2")
model = AutoModelForCausalLM.from_pretrained(
    model_name, 
    cache_dir="./model_cache_2",
    device_map="auto",
    torch_dtype=torch.float16,
    
).eval()

@app.post("/generate")
async def generate_response(request: PromptRequest):
    try:
        # Tokenize input
        inputs = tokenizer(request.prompt, return_tensors="pt").to(model.device)
        
        # Generate response
        outputs = model.generate(
            inputs.input_ids, 
            max_new_tokens=150,  # Limit output length
            do_sample=True,      # Sampling for diversity
            temperature=0.7,     # Adjust randomness
            top_k=50,            # Limits token selection
            top_p=0.9,           # Nucleus sampling
            eos_token_id=tokenizer.eos_token_id  # Stops generation properly
        )

        
        # Decode response
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(response)
        return {"response": response}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Run with: uvicorn main:app --reload