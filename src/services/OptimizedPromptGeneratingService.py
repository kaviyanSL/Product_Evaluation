import requests
from textblob import TextBlob
from transformers import pipeline
import openai


class OptimizedPromptGeneratingService:
    def __init__(self, keywords, use_deepseek=False, use_openai=False, use_huggingface=True):
        """
        Initialize the prompt generation service with DeepSeek, OpenAI, or Hugging Face options.

        :param keywords: List of extracted keywords to generate prompts.
        :param use_deepseek: Boolean to use DeepSeek API (default True).
        :param use_openai: Boolean to use OpenAI API (default False).
        :param use_huggingface: Boolean to use Hugging Face's local model (default False).
        """
        self.use_deepseek = use_deepseek
        self.use_openai = use_openai
        self.use_huggingface = use_huggingface

        self.keywords = self.clean_keywords(keywords)

        if self.use_huggingface:
            self.hf_model = "EleutherAI/gpt-neo-1.3B"
            self.hf_pipeline = pipeline("text-generation", model=self.hf_model)

        if self.use_openai:
            openai.api_key = "YOUR_OPENAI_API_KEY"  

        if self.use_deepseek:
            self.deepseek_api_url = "https://api.deepseek.com/v1/chat/completions"
            self.deepseek_api_key = "sk-69c7a88d0f254e348ad0fb9895e8f05d"  

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

    def generate_with_openai(self):
        response = openai.Completion.create(
            engine="text-davinci-003",  # Use the engine that fits your needs
            prompt=self.generate_prompt(),
            max_tokens=150,
            temperature=0.7,
            top_p=1,
            n=1,
            stop=None
        )
        return response.choices[0].text.strip()

    def generate_with_huggingface(self):
        prompt = self.generate_prompt()
        result = self.hf_pipeline(prompt, max_length=150, do_sample=True, temperature=0.7)
        return result[0]["generated_text"]

    def generate_text(self):
        """Main function to switch between DeepSeek, OpenAI, and Hugging Face."""
        if self.use_deepseek:
            return self.generate_with_deepseek()
        elif self.use_openai:
            return self.generate_with_openai()
        elif self.use_huggingface:
            return self.generate_with_huggingface()
        else:
            return "Error: No model selected."


