import random
import json
import torch
import requests
from model import NeuralNet
from nltk_utils import bag_of_words, tokenize
import re
from train import preprocess_input

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

with open('intents.json', 'r') as json_data:
    intents = json.load(json_data)

FILE = "data.pth"
data = torch.load(FILE)

input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
all_words = data['all_words']
tags = data['tags']
model_state = data["model_state"]

model = NeuralNet(input_size, hidden_size, output_size).to(device)
model.load_state_dict(model_state)
model.eval()

bot_name = "Sam"
platform_api_url = "https://frequence.review-65.a.dev.frequence.rocks/"

# Global dictionary to store slots (e.g., strategy ID)
session_data = {}


# INPUT: user 'input' from ui chat
def get_response(input):
    preprocessed_input = preprocess_input(input)
    sentence = tokenize(preprocessed_input) #this is for intents
    X = bag_of_words(sentence, all_words)
    X = X.reshape(1, X.shape[0])
    X = torch.from_numpy(X).to(device)

    output = model(X)
    _, predicted = torch.max(output, dim=1)

    tag = tags[predicted.item()]

    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()]

    ret = {
        "answer": '',
        "redirect_url": None
    }

    # If AI model has a strong probability for a given intent in our intent set, then retrieve it
    if prob.item() > 0.60:
        for intent in intents['intents']:
            if tag == intent["tag"]:
                performAction(input, intent, ret)
                return ret
    
    print('\n[FAILED][prob]', prob.item(), '\n')

    ret["answer"] = "I do not understand..."
    return ret


def extract_pattern(input, intent):
    for pattern in intent["patterns"]:
        match = re.search(fr"{pattern}", input, re.IGNORECASE)
        if match:
            return match.group(1)
    return None  # No match found


def update_strategy_name_api_call(strategy_id, strategy_name):
    payload = {
        "strategy_id": strategy_id,
        "strategy_name": strategy_name
    }
    path = "strategy_manager/update_strategy_name"
    url = f"{platform_api_url}{path}"
    response = requests.post(url, json=payload)
    ret = response.json()
    print('\n\n==============', url, payload, ret, '==============\n\n')


def performAction(input, intent, ret):
    action = intent["action"] if 'action' in intent else None

    match action:
        case 'store_strategy_id':
            strategy_id = extract_pattern(input, intent)
            session_data['strategy_id'] = strategy_id
            ret["answer"] = random.choice(intent["responses"])
            return ret
        case 'update_strategy_name_api_call':
            strategyIdIsAvailable = 'strategy_id' in session_data and session_data['strategy_id']
            if strategyIdIsAvailable:
                strategy_name = extract_pattern(input, intent)
                strategy_id = session_data['strategy_id']
                update_strategy_name_api_call(strategy_id, strategy_name)
                session_data.clear()    # Clear session data after the process is done
            ret["answer"] = random.choice(intent["responses"])
            return ret
        case 'reset':
            ret["answer"] = random.choice(intent["responses"])
            ret["redirect_url"] = f'{platform_api_url}strategy_manager'
            return ret
        case _:
            ret["answer"] = random.choice(intent["responses"])
            return ret


def prettyPrintObj(obj):
    for key, value in obj.items():
        print(f"{key}: {value}")
    print('\n')


# Run chatbot on terminal
if __name__ == "__main__":
    print("Let's chat! (type 'quit' to exit)", '\n\n')

    while True:
        sentence = input("You: ")
        if sentence == "quit":
            break

        resp = get_response(sentence)
        print(resp, '\n\n')

