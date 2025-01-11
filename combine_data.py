import os
import csv
import json

# Directories
BASE_PATH ="D:/Study/DS201/PROJECT/data"

# Directories
vid_dir = "data_video/test_final"
spe_dir = "data_spec/test_final"
text_dir = "data_text/test_final.csv"
video_dir = os.path.join(BASE_PATH, vid_dir)
spec_dir = os.path.join(BASE_PATH, spe_dir)
text_csv_path = os.path.join(BASE_PATH, text_dir)


# Prepare the output dictionary
output_json = {}
missing_entries = []

# Read all data from the CSV file
csv_data = []
try:
    with open(text_csv_path, mode="r", encoding="utf-8") as csvfile:
        csvreader = csv.DictReader(csvfile)
        for row in csvreader:
            csv_data.append(row)
except FileNotFoundError:
    print(f"CSV file not found: {text_csv_path}")
# Process CSV data and directories
ID=0
for row in csv_data:
    name = row['name'].split(".")[0]
    label = row['label']
    text = row['text']
    
    video_path = os.path.join(video_dir, label, f"{name}.mp4")
    spec_path = os.path.join(spec_dir, label, f"{name}.PNG")
    # Check if video and spec files exist
    video_exists = os.path.exists(video_path)
    spec_exists = os.path.exists(spec_path)
    video_relative_path = f"{vid_dir}/{label}/{name}.mp4"
    spec_relative_path = f"{spe_dir}/{label}/{name}.mp4"
    if not video_exists and not spec_exists:
        missing_entries.append({
            "video_id": name,
            "missing_video": not video_exists,
            "missing_spec": not spec_exists,
            "label": label
        })

    output_json[ID] = {
        "video_path": video_relative_path if video_exists else None,
        "text": text,
        "spec_path": spec_relative_path if spec_exists else None,
        "label": label
    }
    ID+=1

# Save the JSON file
output_file = os.path.join(BASE_PATH, "test_final.json")
with open(output_file, "w", encoding="utf-8") as jsonfile:
    json.dump(output_json, jsonfile, indent=4, ensure_ascii=False)

# Save or print missing entries
missing_file = os.path.join(BASE_PATH, "missing_entries.json")
with open(missing_file, "w", encoding="utf-8") as jsonfile:
    json.dump(missing_entries, jsonfile, indent=4, ensure_ascii=False)

print(f"JSON data saved to {output_file}")
print(f"Missing entries saved to {missing_file}")
