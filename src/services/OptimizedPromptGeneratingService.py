import requests
from textblob import TextBlob
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM

class OptimizedPromptGeneratingService:
    def __init__(self, keywords, use_deepseek=True):
        self.use_deepseek = use_deepseek  
        self.keywords = self.clean_keywords(keywords)  
        
        self.deepseek_api_url = "https://api.deepseek.com/v1/chat/completions"
        self.deepseek_api_key = "sk-69c7a88d0f254e348ad0fb9895e8f05d"  
        
        if not self.use_deepseek:
            self.hf_model = "EleutherAI/gpt-neo-1.3B"  
            self.hf_pipeline = pipeline("text-generation", model=self.hf_model)

    def clean_keywords(self, keywords):
        corrected_keywords = list(set([str(TextBlob(kw).correct()) for kw in keywords]))
        return corrected_keywords

    def generate_prompt(self):
        return (f"Write a well-structured and logical paragraph using these key ideas: "
                f"{', '.join(self.keywords)}. The text should be natural, detailed, and easy to understand.")

    def generate_with_deepseek(self):
        headers = {"Authorization": f"Bearer {self.deepseek_api_key}", "Content-Type": "application/json"}
        data = {
            "model": "deepseek-chat",
            "messages": [{"role": "user", "content": self.generate_prompt()}],
            "max_tokens": 200,
            "temperature": 0.7,
            "top_p": 0.9,
            "repetition_penalty": 1.2,
        }
        response = requests.post(self.deepseek_api_url, headers=headers, json=data)
        if response.status_code == 200:
            return response.json()["choices"][0]["message"]["content"]
        else:
            return f"Error: {response.status_code} - {response.text}"

    def generate_with_huggingface(self):
        prompt = self.generate_prompt()
        result = self.hf_pipeline(prompt, max_length=150, do_sample=True, temperature=0.7)
        return result[0]["generated_text"]

    def generate_text(self):
        if self.use_deepseek:
            return self.generate_with_deepseek()
        else:
            return self.generate_with_huggingface()





