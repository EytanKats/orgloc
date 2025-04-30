import json

labels = json.load(open('/home/kats/storage/staff/eytankats/data/nako_10k/masks_aggregated_v2/labels.json'))

labels_dict = {'labels': {}}
for idx, label in enumerate(labels):
    labels_dict['labels'][label] = idx

with open('/home/kats/storage/staff/eytankats/data/nako_10k/labels_aggregated_v2.json', 'w') as f:
    json.dump(labels_dict, f, indent=4)
