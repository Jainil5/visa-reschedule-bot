import random
import json
import torch
from model import NeuralNet
from nltk_utils import bag_of_words, tokenize

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


def filter_date(x):
    filtered = ""
    date = ""
    num_list = []
    for i in x:
        for j in range(0, 10):
            if i == str(j):
                filtered += str(i)
    for i in filtered:
        num_list.append(i)
    num_list.insert(2, "-")
    num_list.insert(5, "-")
    for i in num_list:
        date += str(i)
    return str(date)


dates_list = ["11-10-2024", "13-10-2024"]
num = 0
nums = [12345, 67890]
bot_name = "BOT"

print("Let's chat! (type 'quit' to exit)")
while True:
    # sentence = "do you use credit cards?"
    sentence = input("You: ")
    if sentence == "quit":
        break
    elif str(sentence).lower().strip() == "dates":
        if num == 0:
            print(f"{bot_name}: You first need to log in with your application number, type 'login'.")
        else:
            print(f"{bot_name}: Dates available for you are: ", str(dates_list))
            selected = input(f"{bot_name}: Enter the date from above to reschedule, in DD-MM-YYYY format: ")
            date = filter_date(selected)
            if len(selected) != 10:
                print(f"{bot_name}: Invalid format, format should be DD-MM-YYYY. To enter date again type 'date'.")
            else:
                if date in dates_list:
                    print(f"{bot_name}: Your rescheduling of dates is successfully completed.")
                else:
                    print(f"{bot_name}: Date you have entered is not available. To enter date again type 'date'.")

    elif str(sentence).lower().strip() == "login":
        num = int(input(f"{bot_name}: Enter your application number: "))
        if int(num) in nums:
            print(f"{bot_name}: Login successful. Now to check available dates, type 'dates'.")
        else:
            print(f"{bot_name}: Application number invalid. To login again, type 'login'.")
    elif str(sentence).lower().strip() == "logout":
        num = 0
        print(f"{bot_name}: Successfully logged out.")
    else:
        sentence = tokenize(sentence)
        X = bag_of_words(sentence, all_words)
        X = X.reshape(1, X.shape[0])
        X = torch.from_numpy(X).to(device)

        output = model(X)
        _, predicted = torch.max(output, dim=1)

        tag = tags[predicted.item()]

        probs = torch.softmax(output, dim=1)
        prob = probs[0][predicted.item()]
        if prob.item() > 0.75:
            for intent in intents['intents']:
                if tag == intent["tag"]:
                    print(f"{bot_name}: {random.choice(intent['responses'])}")
        else:
            print(f"{bot_name}: Sorry! I do not understand this...")
