import json
from pathlib import Path

destination = Path(__file__).resolve().parents[2]/'data'/'raw'

with open(destination/"hello_frieda.json", mode="r", encoding="utf-8") as input_file:
    original_data = input_file.read()

json_data = json.loads(original_data)
mini_json = json.dumps(json_data, indent=None, separators=(",",":"))

with open(destination/"mini_frieda.json", mode="w", encoding="utf-8") as output_file:
    output_file.write(mini_json)

# dog_data = {
#     "name": "Frieda",
#     "is_dog": True,
#     "hobbies": ["eating", "sleeping", "barking"],
#     "age": 8,
#     "address": {
#         "work": None,
#         "home": ("Berlin", "Germany"),
#     },
#     "friends": [
#         {
#             "name": "Phillip",
#             "hobbies": ["eating", "sleeping", "barking"],
#         },
#         {
#             "name": "Mitch",
#             "hobbies": ["running", "snacking"],
#         }
#     ]
# }

# with open(destination/"hello_frieda.json", mode="w", encoding="utf-8") as write_file:
#     json.dump(dog_data, write_file)