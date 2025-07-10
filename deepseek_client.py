import httpx
import json
from typing import List, Dict, Any
from config import DEEPSEEK_API_URL, DEEPSEEK_API_KEY, DEEPSEEK_SYSTEM_PROMPT

class DeepSeekClient:
    def __init__(self):
        self.api_url = DEEPSEEK_API_URL
        self.api_key = DEEPSEEK_API_KEY
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        # Check if API key is set
        if not self.api_key or self.api_key == "your_deepseek_api_key_here":
            print("⚠️  DeepSeek API key not configured!")
            print("Please set your API key in config.py")
    
    def create_user_prompt(self, user_question: str, faq_matches: List[Dict[str, Any]]) -> str:
        """Create a formatted prompt with user question and FAQ matches"""
        prompt = f"User Question: {user_question}\n\n"
        prompt += "FAQ Matches:\n"
        
        for i, match in enumerate(faq_matches, 1):
            prompt += f"{i}. Question: {match['question']}\n"
            prompt += f"   Answer: {match['answer']}\n"
            if match.get('meta'):
                prompt += f"   Meta: {match['meta']}\n"
            prompt += f"   Score: {match['score']:.4f}\n\n"
        
        prompt += "Please provide the most relevant answer based on these FAQ matches."
        return prompt
    
    async def get_response(self, user_question: str, faq_matches: List[Dict[str, Any]]) -> str:
        """Send request to DeepSeek API and get response"""
        # Check if API key is configured
        if not self.api_key or self.api_key == "your_deepseek_api_key_here":
            return self.fallback_response(user_question, faq_matches)
        
        user_prompt = self.create_user_prompt(user_question, faq_matches)
        
        payload = {
            "model": "deepseek-chat",
            "messages": [
                {
                    "role": "system",
                    "content": DEEPSEEK_SYSTEM_PROMPT
                },
                {
                    "role": "user",
                    "content": user_prompt
                }
            ],
            "max_tokens": 1000,
            "temperature": 0.7
        }
        
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    self.api_url,
                    headers=self.headers,
                    json=payload,
                    timeout=30.0
                )
                response.raise_for_status()
                
                result = response.json()
                return result["choices"][0]["message"]["content"]
                
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 402:
                print("❌ DeepSeek API: Payment Required")
                print("Please check your DeepSeek account balance and API key")
                return self.fallback_response(user_question, faq_matches)
            elif e.response.status_code == 401:
                print("❌ DeepSeek API: Unauthorized - Check your API key")
                return self.fallback_response(user_question, faq_matches)
            else:
                print(f"❌ DeepSeek API HTTP Error: {e}")
                return self.fallback_response(user_question, faq_matches)
        except Exception as e:
            print(f"❌ Unexpected error: {e}")
            return self.fallback_response(user_question, faq_matches)
    
    def fallback_response(self, user_question: str, faq_matches: List[Dict[str, Any]]) -> str:
        """Provide a fallback response when DeepSeek API is unavailable"""
        if not faq_matches:
            return "Désolé, aucune réponse appropriée n'a été trouvée."
        
        # Use the best match (highest score)
        best_match = max(faq_matches, key=lambda x: x['score'])
        
        response = f"Basé sur les FAQ disponibles, voici la réponse la plus pertinente:\n\n"
        response += f"Question: {best_match['question']}\n"
        response += f"Réponse: {best_match['answer']}\n"
        
        if best_match.get('meta'):
            response += f"Plus d'informations: {best_match['meta']}\n"
        
        response += f"\n(Score de pertinence: {best_match['score']:.2f})"
        
        return response