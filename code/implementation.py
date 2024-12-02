from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import torch
import matplotlib.pyplot as plt


class PromptAnalysis:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-1.3B")
        self.model = AutoModelForCausalLM.from_pretrained("EleutherAI/gpt-neo-1.3B")

        self.paraphrase_tokenizer = AutoTokenizer.from_pretrained("tuner007/pegasus_paraphrase")
        self.paraphrase_model = AutoModelForSeq2SeqLM.from_pretrained("tuner007/pegasus_paraphrase")

        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

        self.tokenizer.pad_token = self.tokenizer.eos_token

    def generate_response(self, prompt):
        # Tokenize input and add attention mask
        inputs = self.tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)
        
        # Generate output
        outputs = self.model.generate(
            inputs["input_ids"], 
            attention_mask=inputs["attention_mask"], 
            max_length=200,                
            temperature=0.7,          
            top_k=50,                     
            top_p=0.9,                    
            repetition_penalty=1.2,       
            no_repeat_ngram_size=3,    
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,  
        )
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        if not response.endswith(('.', '!', '?')):
            response += '...'
        
        return response


    def get_confidence(self, prompt):
        inputs = self.tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)
        outputs = self.model.generate(
            inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_length=100,
            return_dict_in_generate=True,
            output_scores=True,
        )
        scores = torch.stack(outputs.scores, dim=0)
        probs = torch.nn.functional.softmax(scores, dim=-1)
        confidences = probs.max(dim=-1).values
        average_confidence = confidences.mean().item()
        return average_confidence
    
    def get_entropy(self, prompt):
        inputs = self.tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)
        outputs = self.model.generate(
            inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_length=100,
            return_dict_in_generate=True,
            output_scores=True,
        )
        scores = torch.stack(outputs.scores, dim=0)
        probs = torch.nn.functional.softmax(scores, dim=-1)
        entropies = -(probs * torch.log(probs + 1e-9)).sum(dim=-1)
        average_entropy = entropies.mean().item()
        return average_entropy
    

    def qa_reversibility(self, prompt, response):
        # Reverse the response back to a question
        reverse_prompt = f"Based on this answer, what was the question: {response}"
        reverse_response = self.generate_response(reverse_prompt)
        
        # Compare embeddings
        model = SentenceTransformer('all-MiniLM-L6-v2')  # Small, fast model
        embeddings = model.encode([prompt, reverse_response])
        similarity = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
        return similarity
    
    def generate_paraphrases(self, question, num_paraphrases=5):
        inputs = self.paraphrase_tokenizer([question], truncation=True, padding=True, return_tensors="pt", max_length=60)
        outputs = self.paraphrase_model.generate(
            inputs["input_ids"], 
            max_length=60, 
            num_return_sequences=num_paraphrases, 
            num_beams=num_paraphrases
        )
        return [self.paraphrase_tokenizer.decode(output, skip_special_tokens=True) for output in outputs]
    
    def generate_responses(self, prompts):
        responses = []
        for prompt in prompts:
            inputs = self.tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)
            outputs = self.model.generate(inputs["input_ids"], max_length=100, temperature=0.7)
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            responses.append(response)
        return responses
    
    def calculate_variance(self, question):
        paraphrases = self.generate_paraphrases(question)
        responses = self.generate_responses(paraphrases)
        embeddings = self.embedding_model.encode(responses)
        similarity_matrix = cosine_similarity(embeddings, embeddings)
        
        upper_triangle_indices = np.triu_indices_from(similarity_matrix, k=1)
        variances = np.var(similarity_matrix[upper_triangle_indices])
        
        return {
            "paraphrases": paraphrases,
            "responses": responses,
            "variance": variances
        }


    def track_token_dynamics(self, prompt):
        inputs = self.tokenizer(prompt, return_tensors="pt", add_special_tokens=True)
        input_ids = inputs["input_ids"]
        num_tokens = input_ids.shape[1]
        
        # Store metrics for each token
        token_probabilities = []
        entropies = []
        tokens = self.tokenizer.convert_ids_to_tokens(input_ids[0])
        
        # Process tokens one by one
        for i in range(1, num_tokens):
            partial_input_ids = input_ids[:, :i]
            target_token = input_ids[:, i]
            
            with torch.no_grad():
                outputs = self.model(input_ids=partial_input_ids)
                logits = outputs.logits[:, -1, :]  # Logits for the last token
                probs = torch.softmax(logits, dim=-1).squeeze(0).cpu().numpy()
                
                # Track the probability of the true token
                true_token_prob = probs[target_token.item()]
                token_probabilities.append(true_token_prob)
                
                # Calculate entropy of the distribution
                entropy = -np.sum(probs * np.log(probs + 1e-12))  # Add small value for numerical stability
                entropies.append(entropy)
        
        # Exclude the first token as it's not evaluated
        tokens = tokens[1:]
        
        return tokens, token_probabilities, entropies
    
    def plot_token_influence(tokens, probabilities, entropies):
        x = list(range(len(tokens)))
        
        fig, ax1 = plt.subplots(figsize=(12, 6))
        
        # Plot true token probabilities
        ax1.plot(x, probabilities, 'b-o', label='True Token Probability')
        ax1.set_xlabel('Token Index')
        ax1.set_ylabel('True Token Probability', color='b')
        ax1.tick_params(axis='y', labelcolor='b')
        
        # Add token labels
        ax1.set_xticks(x)
        ax1.set_xticklabels(tokens, rotation=45, ha='right', fontsize=10)
        
        # Plot entropy on a secondary axis
        ax2 = ax1.twinx()
        ax2.plot(x, entropies, 'r--s', label='Entropy')
        ax2.set_ylabel('Entropy', color='r')
        ax2.tick_params(axis='y', labelcolor='r')
        
        # Add a title and legend
        plt.title("Token Influence on Model Predictions")
        ax1.legend(loc="upper left")
        ax2.legend(loc="upper right")
        plt.tight_layout()
        plt.show()