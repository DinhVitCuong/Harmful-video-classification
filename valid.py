# Corrected file paths and task implementation
import os
import json
def update_spec_path_to_png(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
        
        # Update spec_path to replace .mp4 with .png
        for key, value in data.items():
            if "spec_path" in value:
                value["spec_path"] = value["spec_path"].replace(".mp4", ".png")
        
        # Save the updated JSON back to the file
        with open(file_path, 'w', encoding='utf-8') as file:
            json.dump(data, file, indent=4, ensure_ascii=False)
        
        print(f"Successfully updated {file_path}")
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
def remove_null_video_path_and_reorder(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
        
        # Filter out entries where video_path is null
        filtered_data = {
            str(index): entry
            for index, entry in enumerate(
                data.values()
            )
            if entry.get("video_path") is not None
        }
        
        # Save the updated JSON back to the file
        with open(file_path, 'w', encoding='utf-8') as file:
            json.dump(filtered_data, file, indent=4, ensure_ascii=False)
        
        print(f"Successfully cleaned and re-ordered {file_path}")
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        
def replace_text_from_source_to_target(source_file, target_file):
    try:
        # Read source JSON
        with open(source_file, 'r', encoding='utf-8') as source:
            source_data = json.load(source)
        
        # Extract texts from the source file
        source_texts = {key: entry.get("text", "") for key, entry in source_data.items()}
        
        # Read target JSON
        with open(target_file, 'r', encoding='utf-8') as target:
            target_data = json.load(target)
        
        # Replace "text" in the target file with the values from the source file
        for key in target_data:
            if key in source_texts:
                target_data[key]["text"] = source_texts[key]
        
        # Save the updated target JSON back to the file
        with open(target_file, 'w', encoding='utf-8') as target:
            json.dump(target_data, target, indent=4, ensure_ascii=False)
        
        print(f"Successfully replaced 'text' in {target_file} using {source_file}")
    except Exception as e:
        print(f"Error processing files: {e}")
def reindex_json(input_path, output_path):
    """
    Reindex the keys of a JSON file to be sequential integers starting from 0.

    Parameters:
    - input_path: str, path to the input JSON file.
    - output_path: str, path to save the reindexed JSON file.
    """
    # Load the JSON data
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Reindex the JSON data
    reindexed_data = {str(new_idx): value for new_idx, (old_key, value) in enumerate(data.items())}

    # Save the reindexed JSON data
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(reindexed_data, f, ensure_ascii=False, indent=4)

    print(f"Reindexed JSON saved to {output_path}")


# Process each JSON file
json_files = [
    r"D:\Study\DS201\PROJECT\data\data_dict\train.json",
    r"D:\Study\DS201\PROJECT\data\data_dict\val.json",
    r"D:\Study\DS201\PROJECT\data\data_dict\test.json"
]
for json_file in json_files:
    if os.path.exists(json_file):
        remove_null_video_path_and_reorder(json_file)
    else:
        print(f"File not found: {json_file}")

# Process each JSON file
for json_file in json_files:
    if os.path.exists(json_file):
        update_spec_path_to_png(json_file)
    else:
        print(f"File not found: {json_file}")
reindex_json(json_files[0], json_files[0])
reindex_json(json_files[1], json_files[1])
reindex_json(json_files[2], json_files[2])
