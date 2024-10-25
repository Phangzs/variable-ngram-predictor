import json

# Define a function to read and extract text from the JSON file
def extract_text_from_json(file_path,maxLines=-1):
    texts = []
    counter = 0
    with open(file_path, 'r') as file:
        for line in file:
            if maxLines > 0 and counter > maxLines: break
            json_obj = json.loads(line.strip())
            texts.append(json_obj['text'])
            counter += 1
    return texts
