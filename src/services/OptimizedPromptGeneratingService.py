from transformers import pipeline

class OptimizedPromptGeneratingService:
    def __init__(self,keywords):
        self.keywords = keywords

    def prompt_generator_huggingface(self):
        generator = pipeline("text-generation", model="EleutherAI/gpt-neo-1.3B")
        prompt = f"Write a paragraph using these keywords: {', '.join(self.keywords)}."
        result = generator(prompt, max_length=150, do_sample=True, temperature=0.7)
        return result[0]["generated_text"]


