import os
import re

dir_path = "/home/aryan/shl-assessment/sample_conversations/GenAI_SampleConversations"
files = [f for f in os.listdir(dir_path) if f.endswith(".md")]

name_to_type = {}
name_to_url = {}

# Match table rows like:
# | 1 | Occupational Personality Questionnaire OPQ32r | P | Personality & Behavior | 25 minutes | English International ... | URL |
pattern = re.compile(r"\|\s*\d+\s*\|\s*([^|]+?)\s*\|\s*([A-Z])\s*\|[^|]*\|[^|]*\|[^|]*\|\s*<([^>]+)>")

for file in files:
    with open(os.path.join(dir_path, file), "r") as f:
        content = f.read()
    matches = pattern.findall(content)
    for name, t_type, url in matches:
        name = name.strip()
        t_type = t_type.strip()
        url = url.strip()
        name_to_type[name] = t_type
        name_to_url[name] = url

print(f"Extracted {len(name_to_type)} mappings from traces:")
for name, t_type in sorted(name_to_type.items()):
    print(f"  '{name}' -> {t_type} ({name_to_url[name]})")
