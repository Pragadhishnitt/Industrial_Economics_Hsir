import os
import json
import csv

# Function to extract data from JSON-formatted text and append to CSV
def json_text_to_csv(text_file, csv_file):
    with open(text_file, 'r') as f:
        data = f.readlines()

    with open(csv_file, 'a', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        for line in data:
            item = json.loads(line)
            text = ' '.join(item['text'])
            user_id = item['user_id_str']
            writer.writerow([user_id, text])

# Path to the directory containing text files
folder_path = "/home/tharun/AEP"

# Path to CSV file to append data
csv_file = "/home/tharun/data.csv"

# Iterate over all files in the directory
for filename in os.listdir(folder_path):
    
        # Construct the full path to the text file
    text_file = os.path.join(folder_path, filename)
        
        # Call function to append data to CSV
    json_text_to_csv(text_file, csv_file)
