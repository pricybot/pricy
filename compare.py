import json

with open("compare.json", "r") as read_file:
    data = json.load(read_file)

print(data)
