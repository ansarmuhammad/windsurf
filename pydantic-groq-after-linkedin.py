#prompt given
# modify this code so that it has a way to determine the quality of the 
#answer and if the quality 
#is not good enough it would invoke the 
#same process again to get a better answer from groq.  
#it will continue for atleast 3 times before stopping.  
#at the end it should 
#print the answer and the quality score 
#for the answer so that we can see that it is improving

from pydantic import BaseModel
from groq import Groq
from typing import List, Tuple, Dict
import re
from statistics import mean
from collections import Counter

class GroqAPIRequest(BaseModel):
    api_key: str
    prompt: str = "Generate 5 phrases of wisdom"
    max_phrases: int = 5
    model: str = "mixtral-8x7b-32768"
    min_quality_score: float = 0.7
    max_attempts: int = 3

def assess_phrase_quality(phrase: str) -> Dict[str, float]:
    # Define wisdom-related keywords with weights
    wisdom_keywords = {
        'wisdom': 1.2, 'truth': 1.1, 'life': 1.0, 'knowledge': 1.1, 
        'understand': 1.0, 'learn': 1.0, 'patience': 1.1, 'peace': 1.0,
        'journey': 1.0, 'mind': 1.0, 'heart': 1.0, 'soul': 1.1, 'spirit': 1.0,
        'balance': 1.1, 'harmony': 1.1, 'growth': 1.0, 'insight': 1.2
    }
    
    # Initialize scores dictionary
    scores = {}
    
    # Length score (ideal length between 50-120 chars)
    length = len(phrase)
    if 50 <= length <= 120:
        length_score = 1.0
    else:
        length_score = 1.0 - (abs(85 - length) / 85)  # 85 is the midpoint
    scores['length'] = max(0.0, min(1.0, length_score))
    
    # Keyword quality score
    words = re.findall(r'\w+', phrase.lower())
    keyword_matches = [(word, wisdom_keywords[word]) 
                      for word in words if word in wisdom_keywords]
    if keyword_matches:
        keyword_score = sum(weight for _, weight in keyword_matches) / (len(keyword_matches) * 1.2)
    else:
        keyword_score = 0.0
    scores['keywords'] = max(0.0, min(1.0, keyword_score))
    
    # Structure complexity score
    structure_points = 0
    if ',' in phrase: structure_points += 0.3
    if any(word in phrase.lower() for word in ['and', 'but', 'because', 'therefore', 'however']): 
        structure_points += 0.3
    if any(char in phrase for char in ';":'): structure_points += 0.2
    if re.search(r'\b(if|when|while|unless)\b', phrase.lower()): structure_points += 0.2
    scores['structure'] = min(1.0, structure_points)
    
    # Uniqueness score (penalize common phrases)
    words = Counter(words)
    repetition_penalty = sum(count - 1 for count in words.values()) * 0.1
    scores['uniqueness'] = max(0.0, 1.0 - repetition_penalty)
    
    # Calculate weighted average
    weights = {'length': 0.2, 'keywords': 0.3, 'structure': 0.3, 'uniqueness': 0.2}
    final_score = sum(score * weights[metric] for metric, score in scores.items())
    
    return {'total': final_score, **scores}

def get_phrases_of_wisdom(request_data: GroqAPIRequest) -> Tuple[List[str], List[Dict[str, float]]]:
    best_phrases = []
    best_scores = []
    best_total_score = 0.0
    attempts = 0
    
    while attempts < request_data.max_attempts:
        attempts += 1
        try:
            client = Groq(api_key=request_data.api_key)
            
            # Adjust the prompt based on previous attempts
            system_message = (
                "You are a wise assistant. Provide profound, meaningful phrases of wisdom. "
                "Each phrase should be 50-120 characters, contain deep insights, "
                "and use sophisticated language structure. "
                "Focus on universal truths and philosophical insights. "
                "Separate each phrase with a newline."
            )
            
            completion = client.chat.completions.create(
                model=request_data.model,
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": request_data.prompt}
                ],
                temperature=0.7 + (attempts * 0.1),  # Increase temperature with each attempt
                max_tokens=150
            )
            
            if completion.choices and completion.choices[0].message.content:
                phrases = completion.choices[0].message.content.strip().split('\n')
                phrases = [p.strip() for p in phrases if p.strip()][:request_data.max_phrases]
                
                # Get detailed scores for each phrase
                phrase_scores = [assess_phrase_quality(phrase) for phrase in phrases]
                avg_total_score = mean(score['total'] for score in phrase_scores)
                
                print(f"\nAttempt {attempts}:")
                print(f"Overall Quality Score: {avg_total_score:.2f}")
                print("Individual Metrics:")
                for metric in ['length', 'keywords', 'structure', 'uniqueness']:
                    avg_metric = mean(score[metric] for score in phrase_scores)
                    print(f"- {metric.title()}: {avg_metric:.2f}")
                
                if avg_total_score > best_total_score:
                    best_phrases = phrases
                    best_scores = phrase_scores
                    best_total_score = avg_total_score
                
                if avg_total_score >= request_data.min_quality_score:
                    break
                    
            else:
                print(f"Attempt {attempts}: No valid response received")
                
        except Exception as e:
            print(f"Attempt {attempts} failed: {e}")
    
    return best_phrases, best_scores

if __name__ == "__main__":
    request_data = GroqAPIRequest(api_key="your api key here")
    phrases, scores = get_phrases_of_wisdom(request_data)
    
    print("\nFinal Results:")
    print("\nWisdom Phrases with Scores:")
    for i, (phrase, score) in enumerate(zip(phrases, scores), 1):
        print(f"\n{i}. {phrase}")
        print(f"   Quality Score: {score['total']:.2f}")
        print(f"   Metrics: Length={score['length']:.2f}, Keywords={score['keywords']:.2f}, "
              f"Structure={score['structure']:.2f}, Uniqueness={score['uniqueness']:.2f}")
