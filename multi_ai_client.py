import httpx
import json
from typing import List, Dict, Any
from config import DEEPSEEK_SYSTEM_PROMPT

class MultiAIClient:
    def __init__(self):
        self.apis = {
            "deepseek": {
                "url": "https://api.deepseek.com/v1/chat/completions",
                "key": "your_deepseek_api_key_here",  # Replace with your actual API key sk-fb77dc749e634c8f93c0b78fd7e12381
                "model": "deepseek-chat"
            },
            "openai": {
                "url": "https://api.openai.com/v1/chat/completions",
                "key": "your_openai_api_key_here",  # Replace with your actual OpenAI API key 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
                "model": "gpt-3.5-turbo"
            }
        }
        
        # You can add more free APIs here
        #self.free_apis = {
        #    "ollama": {
        #        "url": "http://localhost:11434/api/generate",
        #        "model": "llama2"
        #    }
        #}
    
    def create_user_prompt(self, user_question: str, faq_matches: List[Dict[str, Any]]) -> str:
        """Create a formatted prompt with user question and FAQ matches"""
        prompt = f"Question de l'utilisateur: {user_question}\n\n"
        prompt += "Correspondances FAQ:\n"
        
        for i, match in enumerate(faq_matches, 1):
            prompt += f"{i}. Question: {match['question']}\n"
            prompt += f"   Réponse: {match['answer']}\n"
            if match.get('meta'):
                prompt += f"   Meta: {match['meta']}\n"
            prompt += f"   Score: {match['score']:.4f}\n\n"
        
        prompt += "Veuillez fournir la réponse la plus pertinente basée sur ces correspondances FAQ."
        return prompt
    
    async def try_deepseek(self, user_question: str, faq_matches: List[Dict[str, Any]]) -> str:
        """Try DeepSeek API"""
        api_config = self.apis["deepseek"]
        
        if api_config["key"] == "your_deepseek_api_key_here":
            return None
        
        user_prompt = self.create_user_prompt(user_question, faq_matches)
        
        payload = {
            "model": api_config["model"],
            "messages": [
                {"role": "system", "content": DEEPSEEK_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt}
            ],
            "max_tokens": 1000,
            "temperature": 0.7
        }
        
        headers = {
            "Authorization": f"Bearer {api_config['key']}",
            "Content-Type": "application/json"
        }
        
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    api_config["url"],
                    headers=headers,
                    json=payload,
                    timeout=30.0
                )
                response.raise_for_status()
                result = response.json()
                return result["choices"][0]["message"]["content"]
        except:
            return None
    
    async def try_ollama(self, user_question: str, faq_matches: List[Dict[str, Any]]) -> str:
        """Try local Ollama API (free)"""
        try:
            user_prompt = self.create_user_prompt(user_question, faq_matches)
            
            payload = {
                "model": "llama2",
                "prompt": user_prompt,
                "stream": False
            }
            
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    "http://localhost:11434/api/generate",
                    json=payload,
                    timeout=30.0
                )
                response.raise_for_status()
                result = response.json()
                return result["response"]
        except:
            return None
    
    def rule_based_response(self, user_question: str, faq_matches: List[Dict[str, Any]]) -> str:
        """Simple rule-based response when no AI API is available"""
        if not faq_matches:
            return "Désolé, aucune réponse appropriée n'a été trouvée."
        
        # Use the best match (highest score)
        best_match = max(faq_matches, key=lambda x: x['score'])
        
        # Simple intelligence: if score is very high, use exact answer
        if best_match['score'] > 15:
            response = f"Voici la réponse exacte à votre question:\n\n"
            response += f"{best_match['answer']}"
        else:
            # If score is lower, be more cautious
            response = f"Basé sur les informations disponibles, voici la réponse la plus pertinente:\n\n"
            response += f"Question similaire: {best_match['question']}\n"
            response += f"Réponse: {best_match['answer']}"
        
        if best_match.get('meta'):
            response += f"\n\nPour plus d'informations: {best_match['meta']}"
        
        return response
    
    async def get_response(self, user_question: str, faq_matches: List[Dict[str, Any]]) -> str:
        """Try different AI APIs in order of preference"""
        
        # Try DeepSeek first
        print("🔄 Tentative avec DeepSeek API...")
        result = await self.try_deepseek(user_question, faq_matches)
        if result:
            return result
        
        # Try Ollama (local, free)
        print("🔄 Tentative avec Ollama local...")
        result = await self.try_ollama(user_question, faq_matches)
        if result:
            return f"[Ollama] {result}"
        
        # Fallback to rule-based response
        print("🔄 Utilisation de la réponse basée sur les règles...")
        return self.rule_based_response(user_question, faq_matches)