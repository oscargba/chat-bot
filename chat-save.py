import random
import json
import torch
import requests
from model import NeuralNet
from nltk_utils import bag_of_words, tokenize
import re
import wikipedia

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
    sentence = tokenize(input) #this is for intents
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
        "redirect": None
    }

    # If AI model has a strong probability for a given intent in our intent set, then retrieve it
    if prob.item() > 0.60:
        for intent in intents['intents']:

            if tag == intent["tag"]:    # Main tags: ask_cloning_help | provide_strategy_id | provide_strategy_name | clone_success

                print('\n[tag]', tag, input, '\n')

                strategyIdToCloneIsAvailable = 'strategy_id' in session_data and session_data['strategy_id']

                if tag == 'provide_strategy_id':
                    strategy_id = extract_strategy_id(input, intent)
                    session_data['strategy_id'] = strategy_id

                elif tag == 'provide_strategy_name' and strategyIdToCloneIsAvailable:
                    new_strategy_name = extract_strategy_id(input, intent)
                    new_strategy_id = clone_strategy_api_call(session_data['strategy_id'], new_strategy_name)
                    session_data['cloned_strategy_id'] = new_strategy_id

                elif tag == 'clone_success' and strategyIdToCloneIsAvailable:
                    obj = redirect_to_strategy(session_data['strategy_id'])
                    session_data.clear()    # Clear session data after the process is done

                    ret["redirect_url"] = obj["redirect_url"]

                ret["answer"] = random.choice(intent['responses'])


                prettyPrintObj(intent)
                return ret
    
    print('\n[FAILED][prob]', prob.item(), '\n')

    ret["answer"] = "I do not understand..."
    return ret

def extract_strategy_id(input, intent):
    for pattern in intent["patterns"]:
        print('[extract_strategy_id][---]', fr"{pattern}")
        match = re.search(fr"{pattern}", input, re.IGNORECASE)
        if match:
            print('[extract_strategy_id]', fr"{pattern}", match.group(1))
            return match.group(1)
    return None  # No match found

# Function to extract strategy name
def extract_strategy_name(input, intent):
    for pattern in intent["patterns"]:
        print('[extract_strategy_name][---]', fr"{pattern}")
        match = re.search(fr"{pattern}", input, re.IGNORECASE)
        if match:
            print('[extract_strategy_name]', fr"{pattern}", match.group(1))
            return match.group(1)
    return None

def clone_strategy_api_call(strategy_id, strategy_name):
    payload = {
        "strategy_id": strategy_id,
        "strategy_name": strategy_name
    }
    path = "strategy_manager/update_strategy_name"
    url = f"{platform_api_url}{path}"
    response = requests.post(url, json=payload)

    print('\n\n==============', url, payload, response, '==============\n\n')
    
    if response.status_code == 200:
        # Assume the API returns the new strategy ID in `response.data.message`
        new_strategy_id = response.json()["data"]["message"].split()[-1]
        return new_strategy_id
    else:
        return None

def redirect_to_strategy(new_strategy_id = '10eada8301b480c4b086efc75de42f9d'):
    redirect_url = f"{platform_api_url}{new_strategy_id}"
    return {
        "redirect_url": redirect_url
    }



# Run chatbot on terminal
if __name__ == "__main__":
    print("Let's chat! (type 'quit' to exit)", '\n\n')

    while True:
        sentence = input("You: ")
        if sentence == "quit":
            break

        resp = get_response(sentence)
        print(resp, '\n\n')


def prettyPrintObj(obj):
    for key, value in obj.items():
        print(f"{key}: {value}")
    print('\n')



'''
# Works:

import random
import json
import torch
import requests
from model import NeuralNet
from nltk_utils import bag_of_words, tokenize
import re
import wikipedia

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
    sentence = tokenize(input) #this is for intents
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
        "redirect": None
    }

    # If AI model has a strong probability for a given intent in our intent set, then retrieve it
    if prob.item() > 0.60:
        for intent in intents['intents']:

            if tag == intent["tag"]:    # Main tags: ask_cloning_help | provide_strategy_id | provide_strategy_name | clone_success

                print('\n[tag]', tag, input, '\n')

                strategyIdToCloneIsAvailable = 'strategy_id' in session_data and session_data['strategy_id']

                if tag == 'provide_strategy_id':
                    strategy_id = extract_strategy_id(input, intent)
                    session_data['strategy_id'] = strategy_id
                    ret["answer"] = random.choice(intent['responses']) if strategy_id else "I didn't catch the strategy ID. Could you please repeat it?"

                elif tag == 'provide_strategy_name' and strategyIdToCloneIsAvailable:
                    new_strategy_name = extract_strategy_id(input, intent)
                    new_strategy_id = clone_strategy_api_call(session_data['strategy_id'], new_strategy_name)
                    session_data['cloned_strategy_id'] = new_strategy_id
                    # session_data.clear()    # Clear session data after the process is done
                    ret["answer"] = random.choice(intent['responses'])

                elif tag == 'clone_success' and strategyIdToCloneIsAvailable:
                    obj = redirect_to_strategy(session_data['strategy_id'])
                    session_data.clear()    # Clear session data after the process is done

                    ret["answer"] = random.choice(intent['responses'])
                    ret["redirect_url"] = obj["redirect_url"]

                else:
                    ret["answer"] = random.choice(intent['responses'])


                prettyPrintObj(intent)
                return ret
    
    print('\n[FAILED][prob]', prob.item(), '\n')

    ret["answer"] = "I do not understand..."
    return ret

def extract_strategy_id(input, intent):
    for pattern in intent["patterns"]:
        print('[extract_strategy_id][---]', fr"{pattern}")
        match = re.search(fr"{pattern}", input, re.IGNORECASE)
        if match:
            print('[extract_strategy_id]', fr"{pattern}", match.group(1))
            return match.group(1)
    return None  # No match found

# Function to extract strategy name
def extract_strategy_name(input, intent):
    for pattern in intent["patterns"]:
        print('[extract_strategy_name][---]', fr"{pattern}")
        match = re.search(fr"{pattern}", input, re.IGNORECASE)
        if match:
            print('[extract_strategy_name]', fr"{pattern}", match.group(1))
            return match.group(1)
    return None

def clone_strategy_api_call(strategy_id, strategy_name):
    payload = {
        "strategy_id": strategy_id,
        "strategy_name": strategy_name
    }
    path = "ai_strategy_manager_controller/update_strategy_name"
    url = f"{platform_api_url}{path}"
    response = requests.post(url, json=payload)

    print('\n\n==============', url, payload, response, '==============\n\n')
    
    if response.status_code == 200:
        # Assume the API returns the new strategy ID in `response.data.message`
        new_strategy_id = response.json()["data"]["message"].split()[-1]
        return new_strategy_id
    else:
        return None

def redirect_to_strategy(new_strategy_id = '10eada8301b480c4b086efc75de42f9d'):
    redirect_url = f"{platform_api_url}{new_strategy_id}"
    return {
        "redirect_url": redirect_url
    }



# Run chatbot on terminal
if __name__ == "__main__":
    print("Let's chat! (type 'quit' to exit)", '\n\n')

    while True:
        sentence = input("You: ")
        if sentence == "quit":
            break

        resp = get_response(sentence)
        print(resp, '\n\n')


def prettyPrintObj(obj):
    for key, value in obj.items():
        print(f"{key}: {value}")
    print('\n')
'''