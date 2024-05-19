import json

# Define the number of folds for cross-validation
NUM_FOLDS = 5
MAX_SEQUENCE_LENGTH = 512
EPOCHS = 1
# load id to label convertion dictionary
with open('label_data/type_id_to_label.json', 'r') as json_file:
    type_label_to_id = json.load(json_file)

with open('label_data/category_id_to_label.json', 'r') as json_file:
    category_label_to_id = json.load(json_file)
    
num_labels_category = len(category_label_to_id)
num_labels_type = len(type_label_to_id)
