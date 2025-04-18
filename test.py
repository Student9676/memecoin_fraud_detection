import json
import os

files = os.listdir('data/transaction_data/brain')

for file in files:
    with open(os.path.join('data/transaction_data/brain', file), 'r') as f:
        data = json.load(f)
        if data["wallet"] == "3WfkwcQA3L8tGUXLsFMLNmt2hYbMJbk4fempPedmYBxx":
            print(json.dumps(data, indent=4))
            input("Press Enter to continue...")
