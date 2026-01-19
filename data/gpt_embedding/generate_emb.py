
import openai
import pandas as pd
import numpy as np
import os
import cs
import pickle
import csv
client = openai.OpenAI(api_key = "fill your api-key here") 
df = pd.read_csv("species_id_names.csv")
species_name_to_GPT_response = {}
species_completion_test = df['taxname']
for species in (species_completion_test):
    completion = client.chat.completions.create(
                    model="gpt-4o",
                    messages=[{"role": "user", "content": f"Please summarize the major function of species:{species}.Use academic language in one paragraph."}])
    species_name_to_GPT_response[species] = completion.choices[0].message.content
file_path = "species_name_to_GPT_response_gpt5.csv"
with open(file_path, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["Species", "GPT_Response"])  # Header
    for key, value in species_name_to_GPT_response.items():
        writer.writerow([key, value])

output_path = "/data/jmu27/data/GMrepo/gpt_embedding/species_name_to_GPT_response_gpt4o.pkl"
with open(output_path, "wb") as f:
    pickle.dump(species_name_to_GPT_response, f)

def get_gpt_embedding(text, model="text-embedding-ada-002"):
    text = text.replace("\n", " ")
    return np.array(client.embeddings.create(input = text, model=model).data[0].embedding)

def process_species_embeddings(species_name_to_GPT_response, get_gpt_embedding):
    result = {}
    for index, (species, response) in enumerate(species_name_to_GPT_response.items()):
        print(f"Processing {index}: {species}")
        result[species] = get_gpt_embedding(response)
    return result

result = process_species_embeddings(species_name_to_GPT_response,get_gpt_embedding)
file_path = "gpt5o_embedding_species.pkl"
with open(file_path, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["Species"] + [f"dim_{i}" for i in range(1536)])  # Header
    for key, value in result.items():
        writer.writerow([key] + list(value))

print(f"Dictionary saved to {file_path}")