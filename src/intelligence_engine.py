from transformers import pipeline
import torch

class IntelligenceEngine:
    def __init__(self):
        """Initializes the local LLM (GPT-2) for text generation."""
        print("Loading Intelligence Engine (GPT-2)...")
        device = 0 if torch.cuda.is_available() else -1
        self.generator = pipeline('text-generation', model='distilgpt2', device=device)
        print("Intelligence Engine Loaded.")

    def generate_analysis(self, context_rules, technical_signal, ticker, user_query):
        """
        Generates a natural language response based on RAG context and Technical Signal.
        """
        signal_map = {0: "Sell", 1: "Hold", 2: "Buy"}
        signal_text = signal_map.get(technical_signal, "Neutral")
        
        # Construct Prompt
        # Construct Prompt (Optimized for GPT-2 completion)
        prompt = f"""
Input Data for {ticker}:
- Signal: {signal_text}
- User Question: {user_query}

Analysis:
The stock {ticker} shows a {signal_text} signal. Based on the technical data, """
        
        try:
            print(f"Generating analysis for {ticker}...")
            # Generate Response - Tuned for stability
            response = self.generator(
                prompt, 
                max_new_tokens=60, 
                num_return_sequences=1, 
                temperature=0.1, # strict
                repetition_penalty=1.2, # reduce s/s/s/s
                truncation=True, 
                pad_token_id=50256
            )
            
            generated_text = response[0]['generated_text']
            print("\n--- Raw Gen ---")
            print(generated_text)
            print("---------------\n")
            
            # Robust Extraction
            if "Analysis:" in generated_text:
                result = generated_text.split("Analysis:")[-1].strip()
            else:
                result = generated_text.replace(prompt, "").strip()
            
            # Final Safety Check
            if not result or len(result) < 5:
                print("Output too short, using fallback.")
                return f"The technical indicators for {ticker} suggest a {signal_text} trend. Please review the RSI and MACD charts for confirmation."
                
            return result
            
        except Exception as e:
            print(f"LLM Error: {e}")
            return f"I am analyzing the latest data for {ticker}. The current signal is {signal_text}."
