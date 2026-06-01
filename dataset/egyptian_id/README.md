# combined_detect

Merged detection datasets (bbox-only COCO):
- egyptian_root (Detect Egyptian National ID)
- national_id_detect
- card_seg_det

## Split ratio
80% train / 10% valid / 10% test (stratified by source dataset).

Groups use `extra.name` or Roboflow stem before `.rf.<hash>` so augmentations stay in one split.

## Unified classes (only these ids in annotations)
| id | name |
|----|------|
| 1 | back_egy_id |
| 2 | front_egy_id |

Parent category **id 0** in source exports is ignored.

## Source category mapping
```json
{
  "egyptian_root": {
    "1": 1,
    "2": 2
  },
  "national_id_detect": {
    "1": 1,
    "2": 2
  },
  "card_seg_det": {
    "1": 1,
    "2": 2
  }
}
```

## Stats
```json
{
  "groups": 1605,
  "category_maps": {
    "egyptian_root": {
      "1": 1,
      "2": 2
    },
    "national_id_detect": {
      "1": 1,
      "2": 2
    },
    "card_seg_det": {
      "1": 1,
      "2": 2
    }
  },
  "splits": {
    "train": {
      "groups": 1284,
      "images": 4700,
      "annotations": 5304,
      "by_dataset": {
        "card_seg_det": 3391,
        "egyptian_root": 967,
        "national_id_detect": 342
      },
      "by_class": {
        "back_egy_id": 2541,
        "front_egy_id": 2763
      }
    },
    "valid": {
      "groups": 160,
      "images": 509,
      "annotations": 586,
      "by_dataset": {
        "egyptian_root": 120,
        "card_seg_det": 346,
        "national_id_detect": 43
      },
      "by_class": {
        "back_egy_id": 286,
        "front_egy_id": 300
      }
    },
    "test": {
      "groups": 161,
      "images": 536,
      "annotations": 599,
      "by_dataset": {
        "card_seg_det": 373,
        "national_id_detect": 43,
        "egyptian_root": 120
      },
      "by_class": {
        "back_egy_id": 339,
        "front_egy_id": 260
      }
    }
  }
}
```

Regenerate: `python combine_detection_datasets.py`
