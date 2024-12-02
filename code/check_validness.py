import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

model_name = 'gpt2'
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)

def is_unclear(sentence, threshold=0.05):
    input_ids = tokenizer.encode(sentence, return_tensors='pt')
    with torch.no_grad():
        outputs = model(input_ids)
        logits = outputs.logits[:, -1, :]  
        probabilities = torch.softmax(logits, dim=-1)  
        max_prob = torch.max(probabilities)  
        print('max_prob=', max_prob)
    return max_prob.item() < threshold  

def generate_response(input_sentence):
    if is_unclear(input_sentence):
        return "Please clarify your prompt"
    
    input_ids = tokenizer.encode(input_sentence, return_tensors='pt')
    with torch.no_grad():
        output = model.generate(input_ids, max_length=50)
    
    response = tokenizer.decode(output[0], skip_special_tokens=True)
    return response

user_input = input("User's promt: ")
response = generate_response(user_input)
print("GPT-2:", response)
