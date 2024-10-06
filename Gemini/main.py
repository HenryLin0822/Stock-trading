import google.generativeai as genai
from datasets import load_from_disk
from tqdm import tqdm

dataset = load_from_disk('../fine-tune-data/2330_val')
#print(dataset)
api_key = 'AIzaSyCaP4hP8CWF7xDsqUnJHuwYb9L5sxaa5QY'
genai.configure(api_key=api_key)
model = genai.GenerativeModel('gemini-pro')

total_response = []
for j in tqdm(range(len(dataset))):
    i = j+45
    if i > 55:
        break
    input = dataset[i]['instruction']+dataset[i]['input']
    response = model.generate_content(input)
    total_response.append(response.text)
    print('i:{i} response:{response}'.format(i=i, response=response.text))
#print(input)

print(total_response)
#print(response.text)